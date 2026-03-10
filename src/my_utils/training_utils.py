import os
import argparse
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from pathlib import Path

def parse_args(input_args=None):

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--is_module", default=False)
    parser.add_argument("--tracker_project_name", type=str, default="QUSR")

    # args for the loss function
    parser.add_argument("--lambda_lpips", default=2.0, type=float)
    parser.add_argument("--lambda_l2", default=1.0, type=float)
    parser.add_argument("--lambda_csd", default=1.0, type=float)

    # args for the csd training
    parser.add_argument("--pretrained_model_path_csd", default='preset/models/stable-diffusion-2-1-base', type=str)
    parser.add_argument("--min_dm_step_ratio", default=0.02, type=float)
    parser.add_argument("--max_dm_step_ratio", default=0.98, type=float)
    parser.add_argument("--neg_prompt_csd", default="painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth", type=str)
    parser.add_argument("--pos_prompt_csd", default="", type=str)
    parser.add_argument("--cfg_csd", default=1.0, type=float)

    # args for the `t` test
    parser.add_argument("--timesteps1", default=1, type=float)
    # details about the model architecture
    parser.add_argument("--pretrained_model_path", default='preset/models/stable-diffusion-2-1-base')
    # # unet lora setting
    parser.add_argument("--lora_rank_unet_pix", default=4, type=int)
    parser.add_argument("--lora_rank_unet_sem", default=4, type=int)

    # dataset options
    parser.add_argument("--dataset_txt_paths", default='/gt_path.txt', type=str)
    parser.add_argument("--highquality_dataset_txt_paths", default='/gt_selected_path.txt', type=str)
    parser.add_argument("--dataset_test_folder",
                        default="/testfolder")
    parser.add_argument("--null_text_ratio", default=0., type=float)
    parser.add_argument("--prob", default=0.5, type=float)
    parser.add_argument("--use_online_degradation", action="store_true", 
                        help="online dataset")
    parser.add_argument("--resolution_ori", type=int, default=512,)
    parser.add_argument("--resolution_tgt", type=int, default=512,)
    parser.add_argument("--use_consistent_crop", action="store_true", 
                        help="Use consistent crop strategy for DPO alignment")
    parser.add_argument("--consistent_crop_images", type=str, default=None,
                        help="Path to file containing image names for consistent cropping (one per line)")

    # resume
    parser.add_argument("--resume_ckpt", default=None, type=str)

    # training details
    parser.add_argument("--output_dir", default='experiments/oup')
    parser.add_argument("--seed", type=int, default=123, help="A seed for reproducible training.")
    parser.add_argument("--train_batch_size", type=int, default=2, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=10000)
    parser.add_argument("--max_train_steps", type=int, default=100000,)
    parser.add_argument("--pix_steps", type=int, default=10,)
    parser.add_argument("--checkpointing_steps", type=int, default=500,)
    parser.add_argument("--eval_freq", type=int, default=500, )
    parser.add_argument("--skip_eval", action="store_true", help="Skip evaluation during training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing", action="store_true",)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument("--dataloader_num_workers", type=int, default=0,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--report_to", type=str, default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true",)
    
    # Uncertainty estimation parameters
    parser.add_argument("--enable_uncertainty", action="store_true", 
                       help="Enable dynamic uncertainty estimation")
    parser.add_argument("--uncertainty_hidden_channels", type=int, default=64,
                       help="Hidden channels for uncertainty estimator network")
    parser.add_argument("--min_noise", type=float, default=0.4,
                       help="Minimum noise level for uncertainty-guided diffusion")
    parser.add_argument("--un", type=float, default=0.1,
                       help="Maximum uncertainty value")
    parser.add_argument("--kappa", type=float, default=2.5,
                       help="Noise scaling factor for uncertainty guidance")
    parser.add_argument("--lambda_uncertainty", type=float, default=0.1,
                       help="Weight for uncertainty loss")
    parser.add_argument("--lambda_consistency", type=float, default=0.15,
                       help="Weight for uncertainty consistency loss")
    parser.add_argument("--topk_ratio", type=float, default=0.8,
                       help="Ratio of top-k uncertain pixels for UDR-inspired ranking")
    parser.add_argument("--enable_ranking", action="store_true",
                       help="Enable UDR-inspired ranking mechanism")
    parser.add_argument("--uncertainty_channels", type=int, default=4,
                       help="Number of uncertainty channels (1 for grayscale, 3 for RGB, 4 for VAE latent space)")
    
    # Logging and monitoring parameters
    parser.add_argument("--log_steps", type=int, default=10,
                       help="Log every N steps")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=1000,
                       help="Evaluate every N steps")

    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--deg_file_path", default="params.yml", type=str)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='adain')
    
    # Dual attention parameters  
    parser.add_argument("--enable_dual_attention", action="store_true", 
                        help="Enable dual attention mechanism (semantic + quality prompts)")
    parser.add_argument("--quality_prompt_path", type=str, 
                        default="preset/lowlevel_prompt_q",
                        help="Path to QAP (Quality-Aware Prior) prompt files")
    parser.add_argument("--quality_prompt_gt_path", type=str,
                        default="preset/lowlevel_prompt_q_GT", 
                        help="Path to GT quality prompt files (more detailed)")
    parser.add_argument("--use_gt_quality_prompt", action="store_true",
                        help="Use GT quality prompts instead of regular ones")
    parser.add_argument("--gt_quality_prompt_ratio", type=float, default=0.0,
                        help="Ratio of using GT quality prompts vs LR quality prompts (0.0-1.0). If use_gt_quality_prompt is True, this is ignored.")



    if input_args is not None:
        args, unknown = parser.parse_known_args(input_args)
    else:
        args, unknown = parser.parse_known_args()

    return args