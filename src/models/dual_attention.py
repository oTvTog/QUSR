"""
Dual Attention Module for PISA-SR
Combines semantic attention (high-level text) and quality attention (low-level quality descriptions).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DualAttentionModule(nn.Module):
    """
    Dual Attention Module that combines semantic attention (high-level text) and quality attention (low-level quality descriptions).
    Follows XPSR-style dual attention design for super-resolution tasks.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
        cross_attention_dim: Optional[int] = None,
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim or dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.cross_attention_dim = cross_attention_dim or dim
        
        # Semantic attention (high-level text encoder features)
        self.semantic_to_q = nn.Linear(dim, dim, bias=bias)
        self.semantic_to_k = nn.Linear(self.cross_attention_dim, dim, bias=bias)
        self.semantic_to_v = nn.Linear(self.cross_attention_dim, dim, bias=bias)
        
        # Quality attention (low-level quality description features)  
        self.quality_to_q = nn.Linear(dim, dim, bias=bias)
        self.quality_to_k = nn.Linear(self.cross_attention_dim, dim, bias=bias)  # Quality text features have same dim as semantic
        self.quality_to_v = nn.Linear(self.cross_attention_dim, dim, bias=bias)
        
        # Fusion layer for combining semantic and quality attention
        self.fusion_proj = nn.Linear(dim * 2, dim, bias=bias)
        self.fusion_norm = nn.LayerNorm(dim)
        
        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=bias),
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        semantic_context: Optional[torch.Tensor] = None,
        quality_context: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, height*width, dim] - UNet feature maps
            semantic_context: [batch, seq_len, cross_attention_dim] - High-level text encoder features
            quality_context: [batch, seq_len, cross_attention_dim] - Low-level quality description features
            attention_mask: Optional attention mask
        
        Returns:
            Enhanced hidden states with dual attention
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Semantic Attention Branch (High-level text)
        semantic_output = None
        if semantic_context is not None:
            # Queries from hidden states, Keys/Values from semantic context
            q_semantic = self.semantic_to_q(hidden_states)
            k_semantic = self.semantic_to_k(semantic_context)
            v_semantic = self.semantic_to_v(semantic_context)
            
            semantic_output = self._compute_attention(
                q_semantic, k_semantic, v_semantic, attention_mask
            )
        
        # Quality Attention Branch (Low-level quality descriptions)
        quality_output = None
        if quality_context is not None:
            # Queries from hidden states, Keys/Values from quality context
            q_quality = self.quality_to_q(hidden_states)
            k_quality = self.quality_to_k(quality_context)
            v_quality = self.quality_to_v(quality_context)
            
            quality_output = self._compute_attention(
                q_quality, k_quality, v_quality, attention_mask
            )
        
        # Fusion of dual attention outputs
        if semantic_output is not None and quality_output is not None:
            # Combine both attention outputs
            fused_features = torch.cat([semantic_output, quality_output], dim=-1)
            fused_output = self.fusion_proj(fused_features)
            fused_output = self.fusion_norm(fused_output + hidden_states)
        elif semantic_output is not None:
            # Only semantic attention available
            fused_output = semantic_output + hidden_states
        elif quality_output is not None:
            # Only quality attention available
            fused_output = quality_output + hidden_states
        else:
            # No attention context available, return original
            fused_output = hidden_states
        
        # Final output projection
        output = self.to_out(fused_output)
        return output
    
    def _compute_attention(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute scaled dot-product attention.
        """
        # Reshape for multi-head attention
        batch_size, seq_len_q, _ = query.shape
        seq_len_kv = key.shape[1]
        
        query = query.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Apply softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_probs, value)
        
        # Reshape back to original dimensions
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len_q, -1)
        
        return attention_output


# ImagePromptEncoder is no longer needed since we use text quality prompts instead of image prompts


class DualCrossAttention(nn.Module):
    """
    Dual cross attention that can be integrated into UNet blocks.
    Combines semantic (high-level text) and quality (low-level quality descriptions) attention mechanisms.
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
        
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.scale = dim_head ** -0.5
        
        # Standard cross attention for semantic features (high-level text)
        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k_semantic = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v_semantic = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        
        # Quality prompt attention (low-level quality descriptions)
        self.to_k_quality = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v_quality = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        
        # Fusion and output
        self.fusion_weight = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))  # Learnable weights
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim, bias=bias),
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        quality_prompt_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, query_dim]
            encoder_hidden_states: [batch, seq_len_text, cross_attention_dim] - High-level text features
            quality_prompt_embeds: [batch, seq_len_quality, cross_attention_dim] - Low-level quality description features
            attention_mask: Optional attention mask
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Query from hidden states
        query = self.to_q(hidden_states)
        query = self._reshape_heads(query)  # [batch, heads, seq_len, dim_head]
        
        # Initialize outputs
        semantic_out = None
        quality_out = None
        
        # Semantic attention (high-level text encoder)
        if encoder_hidden_states is not None:
            key_semantic = self.to_k_semantic(encoder_hidden_states)
            value_semantic = self.to_v_semantic(encoder_hidden_states)
            
            key_semantic = self._reshape_heads(key_semantic)
            value_semantic = self._reshape_heads(value_semantic)
            
            semantic_out = self._compute_attention(query, key_semantic, value_semantic, attention_mask)
        
        # Quality prompt attention (low-level quality descriptions)
        if quality_prompt_embeds is not None:
            key_quality = self.to_k_quality(quality_prompt_embeds)
            value_quality = self.to_v_quality(quality_prompt_embeds)
            
            key_quality = self._reshape_heads(key_quality)
            value_quality = self._reshape_heads(value_quality)
            
            quality_out = self._compute_attention(query, key_quality, value_quality, attention_mask)
        
        # Fusion of attention outputs
        if semantic_out is not None and quality_out is not None:
            # Learnable weighted fusion
            weights = F.softmax(self.fusion_weight, dim=0)
            fused_attention = weights[0] * semantic_out + weights[1] * quality_out
        elif semantic_out is not None:
            # Only semantic attention available
            fused_attention = semantic_out
        elif quality_out is not None:
            # Only quality attention available
            fused_attention = quality_out
        else:
            # Fallback to identity - reshape query to match expected output shape
            fused_attention = self._reshape_heads_back(query)
            return self.to_out(fused_attention)
        
        # Reshape and project output
        fused_attention = self._reshape_heads_back(fused_attention)
        output = self.to_out(fused_attention)
        
        return output
    
    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape tensor for multi-head attention."""
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.heads, self.dim_head)
        x = x.transpose(1, 2)  # [batch, heads, seq_len, dim_head]
        return x
    
    def _reshape_heads_back(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape tensor back from multi-head attention."""
        if len(x.shape) == 4:
            batch, heads, seq_len, dim_head = x.shape
            x = x.transpose(1, 2)  # [batch, seq_len, heads, dim_head]
            x = x.reshape(batch, seq_len, heads * dim_head)
        else:
            # Handle case where tensor is already in correct shape
            pass
        return x
    
    def _compute_attention(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute scaled dot-product attention."""
        # Attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Softmax and apply to values
        attention_probs = F.softmax(attention_scores, dim=-1)
        if self.dropout > 0:
            attention_probs = F.dropout(attention_probs, p=self.dropout, training=self.training)
        
        attention_output = torch.matmul(attention_probs, value)
        return attention_output
