"""
Dual UNet for PISA-SR with XPSR-style dual attention mechanism.
Extends UNet2DConditionModel to support both semantic and image prompt conditions.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union, Tuple
from .unet_2d_condition import UNet2DConditionModel, UNet2DConditionOutput
from diffusers.models.attention_processor import AttnProcessor
from .dual_attention import DualCrossAttention


class SelfAttentionProcessor(AttnProcessor):
    """
    Wrapper processor for self-attention layers that filters out dual attention kwargs.
    """
    
    def __init__(self, original_processor):
        super().__init__()
        self.original_processor = original_processor
    
    def __call__(
        self,
        attn: nn.Module,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        scale: float = 1.0,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        """Process self-attention, filtering out quality_prompt_embeds."""
        # Remove quality_prompt_embeds as it's not needed for self-attention
        filtered_kwargs = {k: v for k, v in cross_attention_kwargs.items() 
                          if k != "quality_prompt_embeds"}
        
        return self.original_processor(
            attn=attn,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            temb=temb,
            scale=scale,
            **filtered_kwargs,
        )


class DualUNet2DConditionModel(UNet2DConditionModel):
    """
    Extended UNet2DConditionModel with dual attention mechanism.
    Supports both semantic (high-level text) and quality (low-level quality descriptions) conditioning.
    """
    
    def __init__(
        self,
        enable_dual_attention: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.enable_dual_attention = enable_dual_attention
        
        if self.enable_dual_attention:
            # Replace attention processors with dual attention
            self._setup_dual_attention_processors()
    
    def _setup_dual_attention_processors(self):
        """Replace standard attention processors with dual attention processors."""
        attn_processors = {}
        
        for name, processor in self.attn_processors.items():
            if "attn2" in name:  # Cross attention layers
                # Get the attention module to extract dimensions
                module_path = name.replace(".processor", "")
                try:
                    attn_module = self._get_module_by_path(module_path)
                except Exception as e:
                    print(f"Warning: Could not access module at path {module_path}: {e}")
                    continue
                
                # Safely extract attention parameters with fallbacks
                try:
                    query_dim = attn_module.to_q.in_features
                    cross_attention_dim = attn_module.to_k.in_features
                    
                    # Try different possible attribute names for heads and dim_head
                    if hasattr(attn_module, 'heads'):
                        heads = attn_module.heads
                    elif hasattr(attn_module, 'num_attention_heads'):
                        heads = attn_module.num_attention_heads
                    else:
                        heads = 8  # Default fallback
                    
                    if hasattr(attn_module, 'head_dim'):
                        dim_head = attn_module.head_dim
                    elif hasattr(attn_module, 'attention_head_dim'):
                        dim_head = attn_module.attention_head_dim
                    elif hasattr(attn_module, 'd_head'):
                        dim_head = attn_module.d_head
                    else:
                        # Calculate from query_dim and heads
                        dim_head = query_dim // heads if heads > 0 else 64
                        
                except Exception as e:
                    print(f"Warning: Could not extract attention parameters from {name}, using defaults: {e}")
                    # Use default values if extraction fails
                    query_dim = 1280
                    cross_attention_dim = 1280
                    heads = 8
                    dim_head = 64
                
                # Create dual cross attention processor
                dual_processor = DualCrossAttentionProcessor(
                    query_dim=query_dim,
                    cross_attention_dim=cross_attention_dim,
                    heads=heads,
                    dim_head=dim_head,
                )
                attn_processors[name] = dual_processor
            else:
                # Wrap self-attention processors to filter out quality_prompt_embeds
                attn_processors[name] = SelfAttentionProcessor(processor)
        
        self.set_attn_processor(attn_processors)
        
        # Print final setup summary
        dual_count = sum(1 for name, proc in self.attn_processors.items() if "attn2" in name and isinstance(proc, DualCrossAttentionProcessor))
        self_count = sum(1 for name, proc in self.attn_processors.items() if "attn1" in name and isinstance(proc, SelfAttentionProcessor))
        print(f"Dual attention processors setup complete: {dual_count} cross-attention, {self_count} self-attention layers modified.")
    
    def _get_module_by_path(self, path: str):
        """Get module by path string."""
        module = self
        for part in path.split('.'):
            module = getattr(module, part)
        return module
    
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        # New parameters for dual attention
        quality_prompt_embeds: Optional[torch.Tensor] = None,
        dual_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        """
        Forward pass with dual attention support.
        
        Additional Args:
            quality_prompt_embeds: [batch, seq_len, cross_attention_dim] - Low-level quality description embeddings
            dual_attention_kwargs: Additional arguments for dual attention
        """
        
        # Prepare cross attention kwargs with quality prompts
        if cross_attention_kwargs is None:
            cross_attention_kwargs = {}
        
        # Only add quality_prompt_embeds for dual attention if enabled and provided
        if self.enable_dual_attention and quality_prompt_embeds is not None:
            cross_attention_kwargs["quality_prompt_embeds"] = quality_prompt_embeds
            if dual_attention_kwargs is not None:
                cross_attention_kwargs.update(dual_attention_kwargs)
        
        # Call parent forward with extended kwargs
        return super().forward(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            class_labels=class_labels,
            timestep_cond=timestep_cond,
            attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
            down_intrablock_additional_residuals=down_intrablock_additional_residuals,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=return_dict,
        )


class DualCrossAttentionProcessor(AttnProcessor):
    """
    Attention processor that uses dual cross attention (semantic + quality prompts).
    """
    
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: int = 1280,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.dual_attention = DualCrossAttention(
            query_dim=query_dim,
            cross_attention_dim=cross_attention_dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            bias=bias,
        )
    
    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        """
        Custom attention processor call with dual attention support.
        """
        # Extract quality prompt embeds from cross_attention_kwargs
        quality_prompt_embeds = cross_attention_kwargs.get("quality_prompt_embeds", None)
        
        # Use dual attention if available, otherwise fallback to standard
        if hasattr(self, 'dual_attention'):
            # Ensure dual_attention is on the same device and dtype as hidden_states
            if self.dual_attention.to_q.weight.device != hidden_states.device or self.dual_attention.to_q.weight.dtype != hidden_states.dtype:
                self.dual_attention = self.dual_attention.to(hidden_states.device, hidden_states.dtype)
            
            return self.dual_attention(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                quality_prompt_embeds=quality_prompt_embeds,
                attention_mask=attention_mask,
            )
        else:
            # Fallback to standard cross attention
            return attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
            )


def create_dual_unet_from_pretrained(
    pretrained_model_path: str,
    enable_dual_attention: bool = True,
    **kwargs
) -> DualUNet2DConditionModel:
    """
    Create a DualUNet2DConditionModel from a pretrained UNet2DConditionModel.
    
    Args:
        pretrained_model_path: Path to pretrained model
        enable_dual_attention: Whether to enable dual attention
        **kwargs: Additional arguments for UNet
        
    Returns:
        DualUNet2DConditionModel instance
    """
    # Load pretrained UNet
    original_unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_path, 
        subfolder="unet",
        **kwargs
    )
    
    # Get config from original UNet
    config = original_unet.config
    
    # Create dual UNet with same config
    dual_unet = DualUNet2DConditionModel(
        enable_dual_attention=enable_dual_attention,
        **config
    )
    
    # Copy weights from original UNet (excluding dual attention components)
    dual_unet_state_dict = dual_unet.state_dict()
    original_state_dict = original_unet.state_dict()
    
    for name, param in original_state_dict.items():
        if name in dual_unet_state_dict and dual_unet_state_dict[name].shape == param.shape:
            dual_unet_state_dict[name].copy_(param)
    
    return dual_unet


class DualAttentionUNetBlock(nn.Module):
    """
    A UNet block with integrated dual attention mechanism.
    Can be used as a drop-in replacement for standard UNet blocks.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        cross_attention_dim: int = 1280,
        num_layers: int = 2,
        use_dual_attention: bool = True,
    ):
        super().__init__()
        
        self.use_dual_attention = use_dual_attention
        
        # Standard ResNet layers (simplified)
        self.resnets = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(32, in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, 3, padding=1),
                nn.GroupNorm(32, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
            ) for i in range(num_layers)
        ])
        
        if self.use_dual_attention:
            # Dual attention layers
            self.dual_attentions = nn.ModuleList([
                DualCrossAttention(
                    query_dim=out_channels,
                    cross_attention_dim=cross_attention_dim,
                ) for _ in range(num_layers)
            ])
        
        # Time embedding projection
        self.time_emb_proj = nn.Linear(temb_channels, out_channels)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        quality_prompt_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through dual attention UNet block.
        """
        for i, (resnet, dual_attn) in enumerate(zip(self.resnets, self.dual_attentions if self.use_dual_attention else [None]*len(self.resnets))):
            # ResNet layer
            residual = hidden_states
            hidden_states = resnet(hidden_states)
            
            # Add time embedding
            if temb is not None:
                time_emb = self.time_emb_proj(temb)
                hidden_states = hidden_states + time_emb[:, :, None, None]
            
            # Add residual connection
            if hidden_states.shape == residual.shape:
                hidden_states = hidden_states + residual
            
            # Dual attention
            if self.use_dual_attention and dual_attn is not None:
                # Reshape for attention (flatten spatial dimensions)
                b, c, h, w = hidden_states.shape
                hidden_states_flat = hidden_states.view(b, c, h*w).transpose(1, 2)  # [b, h*w, c]
                
                # Apply dual attention
                attended_states = dual_attn(
                    hidden_states=hidden_states_flat,
                    encoder_hidden_states=encoder_hidden_states,
                    quality_prompt_embeds=quality_prompt_embeds,
                )
                
                # Reshape back to spatial
                hidden_states = attended_states.transpose(1, 2).view(b, c, h, w)
        
        return hidden_states
