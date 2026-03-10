"""
QUSR: Quality-Aware Prior with Uncertainty-Guided Noise Generation for Real-World Super-Resolution
"""

import os
import sys
import time
import random
import glob
import yaml
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import DDPMScheduler
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from diffusers.utils.import_utils import is_xformers_available
from peft import LoraConfig

_QUSR_ROOT = os.path.dirname(os.path.abspath(__file__))
if _QUSR_ROOT not in sys.path:
    sys.path.insert(0, _QUSR_ROOT)
from src.models.autoencoder_kl import AutoencoderKL
from src.models.unet_2d_condition import UNet2DConditionModel
from src.my_utils.vaehook import VAEHook


def find_filepath(directory, filename):
    matches = glob.glob(f"{directory}/**/{filename}", recursive=True)
    return matches[0] if matches else None


def read_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def initialize_unet(rank_sem, return_lora_module_names=False, pretrained_model_path=None):
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    unet.requires_grad_(False)
    unet.train()

    l_target_modules_encoder_sem, l_target_modules_decoder_sem, l_modules_others_sem = [], [], []
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]
    for n, p in unet.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n):
                l_target_modules_encoder_sem.append(n.replace(".weight", ""))
                break
            elif pattern in n and ("up_blocks" in n or "conv_out" in n):
                l_target_modules_decoder_sem.append(n.replace(".weight", ""))
                break
            elif pattern in n:
                l_modules_others_sem.append(n.replace(".weight", ""))
                break

    lora_conf_encoder_sem = LoraConfig(r=rank_sem, init_lora_weights="gaussian", target_modules=l_target_modules_encoder_sem)
    lora_conf_decoder_sem = LoraConfig(r=rank_sem, init_lora_weights="gaussian", target_modules=l_target_modules_decoder_sem)
    lora_conf_others_sem = LoraConfig(r=rank_sem, init_lora_weights="gaussian", target_modules=l_modules_others_sem)

    peft_cfg = getattr(unet, "peft_config", {})
    if "default_encoder_sem" not in peft_cfg:
        unet.add_adapter(lora_conf_encoder_sem, adapter_name="default_encoder_sem")
    if "default_decoder_sem" not in peft_cfg:
        unet.add_adapter(lora_conf_decoder_sem, adapter_name="default_decoder_sem")
    if "default_others_sem" not in peft_cfg:
        unet.add_adapter(lora_conf_others_sem, adapter_name="default_others_sem")

    if return_lora_module_names:
        return unet, l_target_modules_encoder_sem, l_target_modules_decoder_sem, l_modules_others_sem
    return unet


class UEM(nn.Module):
    """Uncertainty Estimation Module: predicts uncertainty map for UNG-guided noise generation."""

    def __init__(self, in_channels=3, hidden_channels=64, out_channels=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
            nn.ELU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, 3, 1, 1),
        )
        self.topk_ratio = 0.8
        self.enable_ranking = True
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)

    def forward(self, lr_image):
        features = self.encoder(lr_image)
        return self.decoder(features)


class CSDLoss(nn.Module):
    def __init__(self, args, accelerator):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path_csd, subfolder="tokenizer")
        self.sched = DDPMScheduler.from_pretrained(args.pretrained_model_path_csd, subfolder="scheduler")
        self.args = args

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        self.unet_fix = UNet2DConditionModel.from_pretrained(args.pretrained_model_path_csd, subfolder="unet")
        if args.enable_xformers_memory_efficient_attention and is_xformers_available():
            self.unet_fix.enable_xformers_memory_efficient_attention()
        self.unet_fix.to(accelerator.device, dtype=weight_dtype)
        self.unet_fix.requires_grad_(False)
        self.unet_fix.eval()

    def forward_latent(self, model, latents, timestep, prompt_embeds):
        return model(latents, timestep=timestep, encoder_hidden_states=prompt_embeds).sample

    def eps_to_mu(self, scheduler, model_output, sample, timesteps):
        alphas_cumprod = scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        alpha_prod_t = alphas_cumprod[timesteps]
        while len(alpha_prod_t.shape) < len(sample.shape):
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
        beta_prod_t = 1 - alpha_prod_t
        return (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5

    def cal_csd(self, latents, prompt_embeds, negative_prompt_embeds, args):
        bsz = latents.shape[0]
        min_dm_step = int(self.sched.config.num_train_timesteps * args.min_dm_step_ratio)
        max_dm_step = int(self.sched.config.num_train_timesteps * args.max_dm_step_ratio)

        timestep = torch.randint(min_dm_step, max_dm_step, (bsz,), device=latents.device).long()
        noise = torch.randn_like(latents)
        noisy_latents = self.sched.add_noise(latents, noise, timestep)

        with torch.no_grad():
            noisy_latents_input = torch.cat([noisy_latents] * 2)
            timestep_input = torch.cat([timestep] * 2)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            noise_pred = self.forward_latent(
                self.unet_fix,
                latents=noisy_latents_input.to(dtype=torch.float16),
                timestep=timestep_input,
                prompt_embeds=prompt_embeds.to(dtype=torch.float16),
            )
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + args.cfg_csd * (noise_pred_text - noise_pred_uncond)
            pred_real_latents = self.eps_to_mu(self.sched, noise_pred, noisy_latents, timestep)
            pred_fake_latents = self.eps_to_mu(self.sched, noise_pred_uncond, noisy_latents, timestep)

        weighting_factor = torch.abs(latents - pred_real_latents).mean(dim=[1, 2, 3], keepdim=True)
        grad = (pred_fake_latents - pred_real_latents) / weighting_factor
        return F.mse_loss(latents, (latents - grad).detach())


class QUSR(nn.Module):
    """QUSR: Quality-Aware Prior + Uncertainty-Guided Noise Generation for Real-World SR."""

    def __init__(self, args):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").cuda()
        self.args = args

        self.enable_uncertainty = getattr(args, 'enable_uncertainty', False)
        if self.enable_uncertainty:
            self.uem = UEM(
                in_channels=3,
                hidden_channels=getattr(args, 'uncertainty_hidden_channels', 64),
                out_channels=3
            ).cuda()
            self.uem.topk_ratio = getattr(args, 'topk_ratio', 0.8)
            self.uem.enable_ranking = getattr(args, 'enable_ranking', True)
            self.min_noise_level = getattr(args, 'min_noise', 0.1)
            self.uncertainty_weight = getattr(args, 'un', 0.1)
            self.kappa = getattr(args, 'kappa', 2.5)
            self.lambda_uncertainty = getattr(args, 'lambda_uncertainty', 0.1)
            self.lambda_consistency = getattr(args, 'lambda_consistency', 0.05)

        self.qap_path = getattr(args, 'quality_prompt_path', 'preset/lowlevel_prompt_q')

        if args.resume_ckpt is None:
            self.unet, lora_unet_modules_encoder_sem, lora_unet_modules_decoder_sem, lora_unet_others_sem = \
                initialize_unet(
                    rank_sem=args.lora_rank_unet_sem,
                    pretrained_model_path=args.pretrained_model_path,
                    return_lora_module_names=True,
                )
            self.lora_rank_unet_sem = args.lora_rank_unet_sem
            self.lora_unet_modules_encoder_sem = lora_unet_modules_encoder_sem
            self.lora_unet_modules_decoder_sem = lora_unet_modules_decoder_sem
            self.lora_unet_others_sem = lora_unet_others_sem
        else:
            print(f'====> resume from {args.resume_ckpt}')
            stage1_yaml = find_filepath(args.resume_ckpt.split('/checkpoints')[0], 'hparams.yml')
            stage1_args = SimpleNamespace(**read_yaml(stage1_yaml))
            self.unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_path, subfolder="unet")
            self.lora_rank_unet_sem = stage1_args.lora_rank_unet_sem
            ckpt = torch.load(args.resume_ckpt, map_location='cpu')
            self.load_ckpt_from_state_dict(ckpt)

        self.unet.to("cuda")
        self.vae_fix = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
        self.vae_fix.to('cuda')
        self.timesteps1 = torch.tensor([args.timesteps1], device="cuda").long()
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()
        self.vae_fix.requires_grad_(False)
        self.vae_fix.eval()

    def set_train_sem(self):
        self.unet.train()
        for n, _p in self.unet.named_parameters():
            if "sem" in n:
                _p.requires_grad = True
        if self.enable_uncertainty:
            self.uem.train()
            for _p in self.uem.parameters():
                _p.requires_grad = True

    def load_ckpt_from_state_dict(self, sd):
        self.lora_conf_encoder_sem = LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian", target_modules=sd["unet_lora_encoder_modules_sem"])
        self.lora_conf_decoder_sem = LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian", target_modules=sd["unet_lora_decoder_modules_sem"])
        self.lora_conf_others_sem = LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian", target_modules=sd["unet_lora_others_modules_sem"])

        peft_cfg = getattr(self.unet, "peft_config", {})
        if "default_encoder_sem" not in peft_cfg:
            self.unet.add_adapter(self.lora_conf_encoder_sem, adapter_name="default_encoder_sem")
        if "default_decoder_sem" not in peft_cfg:
            self.unet.add_adapter(self.lora_conf_decoder_sem, adapter_name="default_decoder_sem")
        if "default_others_sem" not in peft_cfg:
            self.unet.add_adapter(self.lora_conf_others_sem, adapter_name="default_others_sem")

        self.lora_unet_modules_encoder_sem = sd["unet_lora_encoder_modules_sem"]
        self.lora_unet_modules_decoder_sem = sd["unet_lora_decoder_modules_sem"]
        self.lora_unet_others_sem = sd["unet_lora_others_modules_sem"]

        for n, p in self.unet.named_parameters():
            if "lora" in n:
                p.data.copy_(sd["state_dict_unet"][n])

        if sd.get("enable_uncertainty", False) and self.enable_uncertainty:
            if "state_dict_uncertainty" in sd:
                self.uem.load_state_dict(sd["state_dict_uncertainty"])
            if "uncertainty_params" in sd:
                params = sd["uncertainty_params"]
                self.min_noise_level = params.get("min_noise_level", self.min_noise_level)
                self.uncertainty_weight = params.get("uncertainty_weight", self.uncertainty_weight)
                self.kappa = params.get("kappa", self.kappa)
                self.lambda_uncertainty = params.get("lambda_uncertainty", self.lambda_uncertainty)
                self.lambda_consistency = params.get("lambda_consistency", self.lambda_consistency)

    def encode_prompt(self, prompt_batch):
        with torch.no_grad():
            prompt_embeds = [
                self.text_encoder(
                    self.tokenizer(
                        caption, max_length=self.tokenizer.model_max_length,
                        padding="max_length", truncation=True, return_tensors="pt"
                    ).input_ids.to(self.text_encoder.device)
                )[0]
                for caption in prompt_batch
            ]
        return torch.concat(prompt_embeds, dim=0)

    def get_qap(self, image_name_or_index):
        """QAP: Quality-Aware Prior - get quality description from MLLM-generated prompts."""
        try:
            base_path = self.qap_path
            if isinstance(image_name_or_index, (int, str)) and str(image_name_or_index).isdigit():
                prompt_file = f"{int(image_name_or_index):05d}.txt"
            else:
                prompt_file = str(image_name_or_index)
                if not prompt_file.endswith('.txt'):
                    prompt_file += '.txt'
            prompt_path = os.path.join(base_path, prompt_file)
            if os.path.exists(prompt_path):
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            return "This image shows average quality with some details visible but could be improved."
        except Exception as e:
            print(f"Error reading QAP: {e}")
            return "This image shows average quality with some details visible but could be improved."

    def apply_ung(self, latents, uncertainty_map):
        """UNG: Uncertainty-Guided Noise Generation - adaptively adjusts noise injection intensity."""
        if not self.enable_uncertainty:
            return latents

        uncertainty_latent = self.vae_fix.encode(uncertainty_map).latent_dist.sample() * self.vae_fix.config.scaling_factor
        uncertainty_latent = uncertainty_latent.to(dtype=torch.float32)
        final_uncertainty = self.min_noise_level + (1 - self.min_noise_level) * uncertainty_latent
        noise = torch.randn_like(latents)
        noise_scale = torch.sqrt(torch.abs(uncertainty_latent) + 1e-8)
        guided_noise = noise * noise_scale
        return latents + guided_noise * 0.5

    def compute_uncertainty_loss(self, sr_pred, sr_gt, uncertainty_map):
        if not self.enable_uncertainty:
            return torch.tensor(0.0, device=sr_pred.device)

        uncertainty_normalized = torch.sigmoid(uncertainty_map)
        b, c_u, h_u, w_u = uncertainty_normalized.shape
        b, c_p, h_p, w_p = sr_pred.shape

        if (h_u, w_u) != (h_p, w_p):
            uncertainty_resized = F.interpolate(uncertainty_normalized, size=(h_p, w_p), mode='bilinear', align_corners=False)
        else:
            uncertainty_resized = uncertainty_normalized

        if uncertainty_resized.shape[1] != sr_pred.shape[1]:
            if uncertainty_resized.shape[1] == 1:
                uncertainty_resized = uncertainty_resized.repeat(1, sr_pred.shape[1], 1, 1)

        s = torch.exp(-uncertainty_resized)
        sr_weighted = torch.mul(sr_pred, s)
        hr_weighted = torch.mul(sr_gt, s)
        return F.l1_loss(sr_weighted, hr_weighted) + 2 * torch.mean(uncertainty_resized)

    def forward(self, c_t, c_tgt, batch=None, args=None):
        bs = c_t.shape[0]
        encoded_control = self.vae_fix.encode(c_t).latent_dist.sample() * self.vae_fix.config.scaling_factor
        encoded_control = encoded_control.to(dtype=torch.float32)

        uncertainty_map = None
        if self.enable_uncertainty:
            uncertainty_map = self.uem(c_t)

        if self.enable_uncertainty and uncertainty_map is not None and getattr(args, "use_uncertainty_now", True):
            encoded_control = self.apply_ung(encoded_control, uncertainty_map)

        default_prompt = "A high quality image with good details and clarity."
        default_neg_prompt = "low quality, blurry, noisy, distorted"
        default_null_prompt = "image"

        prompt_embeds = self.encode_prompt([default_prompt] * bs)
        neg_prompt_embeds = self.encode_prompt([default_neg_prompt] * bs)
        null_prompt_embeds = self.encode_prompt([default_null_prompt] * bs)

        if random.random() < args.null_text_ratio:
            pos_caption_enc = null_prompt_embeds
        else:
            pos_caption_enc = prompt_embeds

        if "quality_prompts" in batch:
            quality_prompts = batch["quality_prompts"]
        else:
            quality_prompts = [self.get_qap(i) for i in range(bs)]

        quality_prompt_embeds = self.encode_prompt(quality_prompts)

        model_pred = self.unet(
            encoded_control,
            self.timesteps1,
            encoder_hidden_states=quality_prompt_embeds.to(torch.float32),
        ).sample

        x_denoised = encoded_control - model_pred
        output_image = (self.vae_fix.decode(x_denoised / self.vae_fix.config.scaling_factor).sample).clamp(-1, 1)

        if torch.isnan(output_image).any() or torch.isinf(output_image).any():
            output_image = torch.nan_to_num(output_image, nan=0.0, posinf=1.0, neginf=-1.0)
            output_image = output_image.clamp(-1, 1)

        return output_image, x_denoised, prompt_embeds, neg_prompt_embeds, uncertainty_map

    def save_model(self, outf):
        sd = {}
        sd["unet_lora_encoder_modules_sem"] = self.lora_unet_modules_encoder_sem
        sd["unet_lora_decoder_modules_sem"] = self.lora_unet_modules_decoder_sem
        sd["unet_lora_others_modules_sem"] = self.lora_unet_others_sem
        sd["lora_rank_unet_sem"] = self.lora_rank_unet_sem
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k}
        sd["enable_uncertainty"] = self.enable_uncertainty
        if self.enable_uncertainty:
            sd["state_dict_uncertainty"] = self.uem.state_dict()
            sd["uncertainty_params"] = {
                "min_noise_level": self.min_noise_level,
                "uncertainty_weight": self.uncertainty_weight,
                "kappa": self.kappa,
                "lambda_uncertainty": self.lambda_uncertainty,
                "lambda_consistency": self.lambda_consistency
            }
        torch.save(sd, outf)


class QUSR_eval(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = "cuda"
        self.weight_dtype = torch.float16 if args.mixed_precision == "fp16" else (torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32)
        self.args = args

        self.enable_uncertainty = getattr(args, 'enable_uncertainty', False)
        if self.enable_uncertainty:
            self.uem = UEM(
                in_channels=3,
                hidden_channels=getattr(args, 'uncertainty_hidden_channels', 64),
                out_channels=3
            ).to(self.device)
            self.uem.topk_ratio = getattr(args, 'topk_ratio', 0.8)
            self.uem.enable_ranking = getattr(args, 'enable_ranking', True)

        self.qap_path = getattr(args, 'quality_prompt_path', 'preset/test_lowlevel_prompt_q_RealSR')

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").to(self.device)
        self.sched = DDPMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_path, subfolder="unet")

        self._load_pretrained_weights(args.pretrained_path)
        self._init_tiled_vae(
            encoder_tile_size=args.vae_encoder_tiled_size,
            decoder_tile_size=args.vae_decoder_tiled_size
        )

        set_weights_and_activate_adapters(self.unet, ["default_encoder_sem", "default_decoder_sem", "default_others_sem"], [1.0, 1.0, 1.0])
        self.unet.merge_and_unload()

        for model in [self.vae, self.unet, self.text_encoder]:
            model.to(self.device, dtype=self.weight_dtype)
            model.requires_grad_(False)
        if self.enable_uncertainty:
            self.uem.to(self.device, dtype=self.weight_dtype)

        self.timesteps1 = torch.tensor([1], device=self.device).long()

    def _load_pretrained_weights(self, pretrained_path):
        sd = torch.load(pretrained_path, map_location='cpu')
        self._load_ckpt_from_state_dict(sd)

    def _load_ckpt_from_state_dict(self, sd):
        self.lora_conf_encoder_sem = LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian", target_modules=sd["unet_lora_encoder_modules_sem"])
        self.lora_conf_decoder_sem = LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian", target_modules=sd["unet_lora_decoder_modules_sem"])
        self.lora_conf_others_sem = LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian", target_modules=sd["unet_lora_others_modules_sem"])

        peft_cfg = getattr(self.unet, "peft_config", {})
        if "default_encoder_sem" not in peft_cfg:
            self.unet.add_adapter(self.lora_conf_encoder_sem, adapter_name="default_encoder_sem")
        if "default_decoder_sem" not in peft_cfg:
            self.unet.add_adapter(self.lora_conf_decoder_sem, adapter_name="default_decoder_sem")
        if "default_others_sem" not in peft_cfg:
            self.unet.add_adapter(self.lora_conf_others_sem, adapter_name="default_others_sem")

        for name, param in self.unet.named_parameters():
            if "lora" in name:
                param.data.copy_(sd["state_dict_unet"][name])

        set_weights_and_activate_adapters(self.unet, ["default_encoder_sem", "default_decoder_sem", "default_others_sem"], [1.0, 1.0, 1.0])
        self.unet.merge_and_unload()

        if sd.get("enable_uncertainty", False) and self.enable_uncertainty and "state_dict_uncertainty" in sd:
            self.uem.load_state_dict(sd["state_dict_uncertainty"])
            self.uem.eval()
            self.uem.requires_grad_(False)

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        if self.enable_uncertainty:
            self.uem.eval()
            self.uem.requires_grad_(False)

    def encode_prompt(self, prompt_batch):
        with torch.no_grad():
            prompt_embeds = [
                self.text_encoder(
                    self.tokenizer(
                        caption, max_length=self.tokenizer.model_max_length,
                        padding="max_length", truncation=True, return_tensors="pt"
                    ).input_ids.to(self.text_encoder.device)
                )[0]
                for caption in prompt_batch
            ]
        return torch.concat(prompt_embeds, dim=0)

    def get_qap(self, image_name_or_index):
        try:
            base_path = self.qap_path
            if isinstance(image_name_or_index, (int, str)) and str(image_name_or_index).isdigit():
                prompt_file = f"{int(image_name_or_index):05d}.txt"
            else:
                prompt_file = str(image_name_or_index)
                if not prompt_file.endswith('.txt'):
                    prompt_file += '.txt'
            prompt_path = os.path.join(base_path, prompt_file)
            if os.path.exists(prompt_path):
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            return "This image shows average quality with some details visible but could be improved."
        except Exception as e:
            print(f"Error reading QAP: {e}")
            return "This image shows average quality with some details visible but could be improved."

    def apply_ung_inference(self, encoded_control, uncertainty_map):
        uncertainty_latent = self.vae.encode(uncertainty_map).latent_dist.sample() * self.vae.config.scaling_factor
        uncertainty_latent = uncertainty_latent.to(dtype=self.weight_dtype)
        min_noise_level = getattr(self.args, 'min_noise', 0.1)
        final_uncertainty = min_noise_level + (1 - min_noise_level) * uncertainty_latent
        noise = torch.randn_like(encoded_control)
        noise_scale = torch.sqrt(torch.abs(uncertainty_latent) + 1e-8)
        guided_noise = noise * noise_scale
        return encoded_control + guided_noise * 0.5

    @torch.no_grad()
    def forward(self, default, c_t, prompt=None, image_name=None):
        torch.cuda.synchronize()
        start_time = time.time()

        c_t = c_t.to(dtype=self.weight_dtype)

        if image_name is not None:
            try:
                base_name = os.path.basename(image_name)
                image_name_no_ext = os.path.splitext(base_name)[0]
                semantic_prompt = self.get_qap(image_name_no_ext)
            except Exception:
                semantic_prompt = "A high quality image with good details and clarity."
        else:
            semantic_prompt = "A high quality image with good details and clarity."

        prompt_embeds = self.encode_prompt([semantic_prompt]).to(dtype=self.weight_dtype)
        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
        encoded_control = encoded_control.to(dtype=self.weight_dtype)

        if self.enable_uncertainty:
            uncertainty_map = self.uem(c_t)
            encoded_control = self.apply_ung_inference(encoded_control, uncertainty_map)

        model_pred = self._process_latents(encoded_control, prompt_embeds, default)
        x_denoised = encoded_control - model_pred
        output_image = self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample.clamp(-1, 1)

        if torch.isnan(output_image).any() or torch.isinf(output_image).any():
            output_image = torch.nan_to_num(output_image, nan=0.0, posinf=1.0, neginf=-1.0)
            output_image = output_image.clamp(-1, 1)

        torch.cuda.synchronize()
        return time.time() - start_time, output_image

    def _process_latents(self, encoded_control, prompt_embeds, default):
        h, w = encoded_control.size()[-2:]
        tile_size, tile_overlap = self.args.latent_tiled_size, self.args.latent_tiled_overlap

        if h * w <= tile_size * tile_size:
            return self._predict_no_tiling(encoded_control, prompt_embeds, default)
        return self._predict_with_tiling(encoded_control, prompt_embeds, default, tile_size, tile_overlap)

    def _predict_no_tiling(self, encoded_control, prompt_embeds, default):
        return self.unet(encoded_control, self.timesteps1, encoder_hidden_states=prompt_embeds).sample

    def _predict_with_tiling(self, encoded_control, prompt_embeds, default, tile_size, tile_overlap):
        _, _, h, w = encoded_control.size()
        tile_weights = self._gaussian_weights(tile_size, tile_size, 1)
        tile_size = min(tile_size, min(h, w))
        grid_rows = 0
        cur_x = 0
        while cur_x < encoded_control.size(-1):
            cur_x = max(grid_rows * tile_size - tile_overlap * grid_rows, 0) + tile_size
            grid_rows += 1

        grid_cols = 0
        cur_y = 0
        while cur_y < encoded_control.size(-2):
            cur_y = max(grid_cols * tile_size - tile_overlap * grid_cols, 0) + tile_size
            grid_cols += 1

        input_list = []
        noise_preds = []
        for row in range(grid_rows):
            for col in range(grid_cols):
                if col < grid_cols - 1 or row < grid_rows - 1:
                    ofs_x = max(row * tile_size - tile_overlap * row, 0)
                    ofs_y = max(col * tile_size - tile_overlap * col, 0)
                if row == grid_rows - 1:
                    ofs_x = w - tile_size
                if col == grid_cols - 1:
                    ofs_y = h - tile_size

                input_tile = encoded_control[:, :, ofs_y:ofs_y + tile_size, ofs_x:ofs_x + tile_size]
                input_list.append(input_tile)

                if len(input_list) == 1 or col == grid_cols - 1:
                    input_list_t = torch.cat(input_list, dim=0)
                    model_out = self.unet(input_list_t, self.timesteps1, encoder_hidden_states=prompt_embeds).sample
                    input_list = []
                noise_preds.append(model_out)

        noise_pred = torch.zeros(encoded_control.shape, device=encoded_control.device)
        contributors = torch.zeros(encoded_control.shape, device=encoded_control.device)
        for row in range(grid_rows):
            for col in range(grid_cols):
                if col < grid_cols - 1 or row < grid_rows - 1:
                    ofs_x = max(row * tile_size - tile_overlap * row, 0)
                    ofs_y = max(col * tile_size - tile_overlap * col, 0)
                if row == grid_rows - 1:
                    ofs_x = w - tile_size
                if col == grid_cols - 1:
                    ofs_y = h - tile_size
                noise_pred[:, :, ofs_y:ofs_y + tile_size, ofs_x:ofs_x + tile_size] += noise_preds[row * grid_cols + col] * tile_weights
                contributors[:, :, ofs_y:ofs_y + tile_size, ofs_x:ofs_x + tile_size] += tile_weights
        return noise_pred / contributors

    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        from numpy import pi, exp, sqrt
        import numpy as np
        midpoint_x = (tile_width - 1) / 2
        midpoint_y = (tile_height - 1) / 2
        x_probs = [exp(-(x - midpoint_x) ** 2 / (2 * (tile_width ** 2) * 0.01)) / sqrt(2 * pi * 0.01) for x in range(tile_width)]
        y_probs = [exp(-(y - midpoint_y) ** 2 / (2 * (tile_height ** 2) * 0.01)) / sqrt(2 * pi * 0.01) for y in range(tile_height)]
        weights = np.outer(y_probs, x_probs)
        return torch.tensor(weights, device=self.device).repeat(nbatches, self.unet.config.in_channels, 1, 1)

    def _init_tiled_vae(self, encoder_tile_size=256, decoder_tile_size=256, fast_decoder=False, fast_encoder=False, color_fix=False, vae_to_gpu=True):
        encoder, decoder = self.vae.encoder, self.vae.decoder
        if not hasattr(encoder, 'original_forward'):
            encoder.original_forward = encoder.forward
        if not hasattr(decoder, 'original_forward'):
            decoder.original_forward = decoder.forward
        encoder.forward = VAEHook(encoder, encoder_tile_size, is_decoder=False, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)
        decoder.forward = VAEHook(decoder, decoder_tile_size, is_decoder=True, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)
