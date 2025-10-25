"""
This module implements the full Transformer model as described in
"Attention is All You Need" (Vaswani et al., 2017).

The Transformer is a sequence-to-sequence model that uses attention mechanisms
instead of recurrence, enabling parallel processing and better long-range dependencies.

High-Level Architecture:
------------------------
    Source Sequence → ENCODER → Context Representations
                                        ↓
    Target Sequence → DECODER → Output Predictions
    
The encoder processes the input sequence into contextualized representations.
The decoder generates the output sequence autoregressively, attending to both
its own previous outputs and the encoder's representations.

Model Components:
-----------------
**ENCODER:**
- Input Embeddings + Positional Encoding
- N stacked encoder blocks, each with:
  * Multi-head self-attention
  * Feed-forward network
  * Residual connections + layer normalization

**DECODER:**
- Output Embeddings + Positional Encoding  
- N stacked decoder blocks, each with:
  * Masked multi-head-attention (on target)
  * cross-attention (to encoder)
  * Feed-forward network
  * Residual connections + layer normalization

**OUTPUT:**
- Linear projection to vocabulary
- (Softmax applied during loss/inference)

Default Hyperparameters (Base Model):
--------------------------------------
- d_model = 512 (embedding dimension)
- N = 6 (number of encoder/decoder layers)
- h = 8 (number of attention heads)
- d_ff = 2048 (feed-forward hidden dimension)
- dropout = 0.1

Training Optimizations:
-----------------------
- Xavier initialization for stable gradients
"""

import torch.nn as nn

from .InputEmbeddings import InputEmbeddings
from .PositionalEncoding import PositionalEncoding
from .MultiHeadAttentionBlock import MultiHeadAttentionBlock
from .FeedForwardBlock import FeedForwardBlock
from .ProjectionLayer import ProjectionLayer
from .Encoder import EncoderBlock, Encoder
from .Decoder import DecoderBlock, Decoder


class Transformer(nn.Module):

    def __init__(
        self,
        src_vocab_size: int,      # Size of source vocabulary
        tgt_vocab_size: int,      # Size of target vocabulary
        src_seq_len: int,         # Maximum length of source sequence
        tgt_seq_len: int,         # Maximum length of target sequence
        model_dimension: int = 512,  # Embedding/hidden dimension (d_model)
        N: int = 6,               # Number of encoder/decoder layers
        h: int = 8,               # Number of attention heads
        dropout: float = 0.1,     # Dropout rate for regularization
        d_ff: int = 2048          # Hidden size of feed-forward network
    ) -> None:
        """
        Initialize the Transformer model with all components.
        
        Args:
            src_vocab_size (int): Number of unique tokens in source language
            tgt_vocab_size (int): Number of unique tokens in target language
            src_seq_len (int): Maximum source sequence length to support
            tgt_seq_len (int): Maximum target sequence length to support
            model_dimension (int): Dimension of embeddings and hidden states (d_model)
            N (int): Number of encoder and decoder layers to stack
            h (int): Number of parallel attention heads
            dropout (float): Dropout probability (0.0 to 1.0)
            d_ff (int): Inner dimension of feed-forward networks
        
        The model is initialized with:
        - Random embeddings (later optimized during training)
        - Xavier uniform initialization for all weight matrices
        - Weight sharing between embeddings and projection layer
        """
        super().__init__()

        # ============================================================
        # ENCODER COMPONENTS
        # ============================================================
        
        # Convert source token indices to dense vector representations
        # Shape: (batch, src_seq_len) → (batch, src_seq_len, d_model)
        src_embed = InputEmbeddings(model_dimension, src_vocab_size)
        
        # Add positional information to source embeddings
        # Injects sequence order using sinusoidal patterns
        src_pos = PositionalEncoding(model_dimension, src_seq_len, dropout)

        # Build N identical encoder blocks
        encoder_blocks = []
        for _ in range(N):
            # Self-attention: each source token attends to all source tokens
            encoder_self_attention_block = MultiHeadAttentionBlock(
                model_dimension, h, dropout
            )
            
            # Feed-forward: position-wise non-linear transformation
            feed_forward_block = FeedForwardBlock(
                model_dimension, d_ff, dropout
            )
            
            # Complete encoder block: self-attention + FFN + residuals + norms
            encoder_block = EncoderBlock(
                model_dimension,
                encoder_self_attention_block,
                feed_forward_block,
                dropout
            )
            encoder_blocks.append(encoder_block)
        
        # Wrap all encoder blocks in an Encoder module
        encoder = Encoder(model_dimension, nn.ModuleList(encoder_blocks))

        # ============================================================
        # DECODER COMPONENTS
        # ============================================================
        
        # Convert target token indices to dense vector representations
        # Shape: (batch, tgt_seq_len) → (batch, tgt_seq_len, d_model)
        tgt_embed = InputEmbeddings(model_dimension, tgt_vocab_size)
        
        # Add positional information to target embeddings
        tgt_pos = PositionalEncoding(model_dimension, tgt_seq_len, dropout)

        # Build N identical decoder blocks
        decoder_blocks = []
        for _ in range(N):
            # 1. Masked self-attention: target tokens attend to previous target tokens
            #    Prevents looking ahead during autoregressive generation
            decoder_self_attention_block = MultiHeadAttentionBlock(
                model_dimension, h, dropout
            )
            
            # 2. Cross-attention: target attends to encoder output
            #    This is how the decoder accesses source information
            decoder_cross_attention_block = MultiHeadAttentionBlock(
                model_dimension, h, dropout
            )
            
            # 3. Feed-forward: position-wise transformation
            feed_forward_block = FeedForwardBlock(
                model_dimension, d_ff, dropout
            )
            
            # Complete decoder block: masked self-attn + cross-attn + FFN + residuals + norms
            decoder_block = DecoderBlock(
                model_dimension,
                decoder_self_attention_block,
                decoder_cross_attention_block,
                feed_forward_block,
                dropout
            )
            decoder_blocks.append(decoder_block)
        
        # Wrap all decoder blocks in a Decoder module
        decoder = Decoder(model_dimension, nn.ModuleList(decoder_blocks))

        # ============================================================
        # OUTPUT PROJECTION
        # ============================================================
        
        # Project decoder outputs to target vocabulary logits
        projection_layer = ProjectionLayer(model_dimension, tgt_vocab_size)

        
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

        # ============================================================
        # WEIGHT SHARING (PARAMETER TYING)
        # ============================================================
        
        # Share embedding weights between source and target
        # source and target vocabularies are the same
        # Reduces parameters and can improve generalization
        self.src_embed.embedding.weight = self.tgt_embed.embedding.weight
        
        # Share target embedding weights with projection layer
        # The same matrix used to embed tokens is used to predict them
        # This enforces consistency: similar embeddings → similar predictions
        self.projection_layer.proj.weight = self.tgt_embed.embedding.weight
        
        # Apply Xavier uniform initialization to all parameter matrices
        # This helps with gradient flow and training stability
        self.init_with_xavier()

    # ================================================================
    # FORWARD PASS UTILITIES
    # ================================================================

    def encode(self, src, src_mask):
        """
        Encode the source sequence into contextualized representations.
        
        Args:
            src (torch.Tensor): Source token indices
                               Shape: (batch_size, src_seq_len)
            src_mask (torch.Tensor): Source mask to ignore padding
                                    Shape: (batch_size, 1, 1, src_seq_len)
        
        Returns:
            torch.Tensor: Encoded source representations (encoder memory)
                         Shape: (batch_size, src_seq_len, model_dimension)
        
        Process:
            1. Convert token indices to embeddings
            2. Add positional encodings
            3. Pass through encoder stack
        """
        # Step 1: Embed source tokens
        # (batch, src_seq_len) → (batch, src_seq_len, d_model)
        src = self.src_embed(src)
        
        # Step 2: Add positional information
        # (batch, src_seq_len, d_model) → (batch, src_seq_len, d_model)
        src = self.src_pos(src)
        
        # Step 3: Pass through encoder
        # (batch, src_seq_len, d_model) → (batch, src_seq_len, d_model)
        return self.encoder(src, src_mask)

    def decode(self, tgt, memory, src_mask, tgt_mask):
        """
        Decode the target sequence using encoder memory.
        
        Args:
            tgt (torch.Tensor): Target token indices
                               Shape: (batch_size, tgt_seq_len)
            memory (torch.Tensor): Encoder output (from encode())
                                  Shape: (batch_size, src_seq_len, model_dimension)
            src_mask (torch.Tensor): Source mask for cross-attention
                                    Shape: (batch_size, 1, 1, src_seq_len)
            tgt_mask (torch.Tensor): Target mask (causal + padding)
                                    Shape: (batch_size, 1, tgt_seq_len, tgt_seq_len)
        
        Returns:
            torch.Tensor: Decoded target representations
                         Shape: (batch_size, tgt_seq_len, model_dimension)
        
        Process:
            1. Convert target token indices to embeddings
            2. Add positional encodings
            3. Pass through decoder stack (with access to encoder memory)
        
        The output will be projected to vocabulary space for predictions.
        """
        # Step 1: Embed target tokens
        # (batch, tgt_seq_len) → (batch, tgt_seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        
        # Step 2: Add positional information
        # (batch, tgt_seq_len, d_model) → (batch, tgt_seq_len, d_model)
        tgt = self.tgt_pos(tgt)
        
        # Step 3: Pass through decoder (attending to memory)
        # (batch, tgt_seq_len, d_model) → (batch, tgt_seq_len, d_model)
        return self.decoder(tgt, memory, src_mask, tgt_mask)

    def project(self, x):
        """
        Project decoder outputs to target vocabulary logits.
        
        Args:
            x (torch.Tensor): Decoder output representations
                             Shape: (batch_size, seq_len, model_dimension)
        
        Returns:
            torch.Tensor: Vocabulary logits (unnormalized scores)
                         Shape: (batch_size, seq_len, tgt_vocab_size)
        """

        # Project to vocabulary space
        # (batch, seq_len, d_model) → (batch, seq_len, vocab_size)
        return self.projection_layer(x)

    def init_with_xavier(self):
        """
        Initialize all model parameters using Xavier uniform initialization.
        """
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)