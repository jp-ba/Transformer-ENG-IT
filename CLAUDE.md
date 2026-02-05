# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CSCI 440 coursework implementing core Transformer architecture components from scratch in Python using PyTorch. The code is heavily commented for educational purposes, explaining both the mathematics and PyTorch API usage.

## Running

Scripts are standalone Python files with no build system or dependency manager:

```bash
python "CSCI_440-03-m+c_1.7.py"
python "CSCI_440-04-m+c_1.1.py"
```

**Dependencies:** `torch` (PyTorch), `math` (stdlib)

## Architecture

Each file implements one or more Transformer modules as `nn.Module` subclasses:

- **`CSCI_440-03-m+c_1.7.py`** — Module 2: `PositionalEncoding`. Implements sine/cosine positional embeddings using log-space computation for numerical stability. Code is incomplete (marked "stopped here" at line 230) and has known syntax issues (line 205: `math.log(10000,0)` should be `math.log(10000.0)`; line 228: invalid unsqueeze multiplication; `forward` is defined outside the class).

- **`CSCI_440-04-m+c_1.1.py`** — Module 3: `LayerNormalization` with learnable alpha/bias parameters. Module 4: `FeedForwardBlock` (two linear layers with ReLU and dropout). Has syntax issues (lines 56-57: `dim * -1` should be `dim=-1`; line 84: stray `?` character).

- **`CSCI_440-07-m+c_1.3.py`** — Module 5: `MultiHeadAttentionBlock` (incomplete version). Contains `__init__` and partial `forward` method with Q, K, V matrix split logic. Code stops at "stopped here lecture 06" (line 109). This is an earlier lecture snapshot before the attention calculation was implemented.

- **`CSCI_440-08-m+c_1.4.py`** — Most complete file containing all core modules:
  - Module 1: `InputEmbeddings` (token embeddings with sqrt(d_model) scaling)
  - Module 2: `PositionalEncoding` (sine/cosine positional embeddings)
  - Module 3: `LayerNormalization` (learnable alpha/bias parameters)
  - Module 4: `FeedForwardBlock` (two linear layers with ReLU and dropout)
  - Module 5: `MultiHeadAttentionBlock` (complete with `attention` staticmethod and masking)
  - Module 6: `ResidualConnection` (skip connections / Add & Norm)

  Syntax issues: line 73: `nn.embedding` should be `nn.Embedding`; line 121: `torch.arrange` should be `torch.arange`; line 124: `math.log(10000,0)` should be `math.log(10000.0)`; line 144: invalid multiplication with shape comment; lines 150, 153: code outside class; lines 198-199: `dim * -1` should be `dim=-1`; line 226: stray `?`; line 289: missing `@staticmethod` decorator; line 297: `?*` should be `@`; line 352: incorrect assignment syntax; line 374+: `ResidualConnection` incorrectly indented inside `MultiHeadAttentionBlock`.

## Conventions

- Files follow the naming pattern `CSCI_440-{module}-m+c_{version}.py`
- Each file has two sections: `#I. Math` (theory/reference notes) and `#II. Code` (implementation)
- Standard Transformer dimension names: `d_model` (512), `d_ff` (2048), `seq_len`, `dropout`
- All neural network components inherit from `nn.Module` using the constructor/forward pattern

## Project Structure

Target file organization for the EN→IT (512D) transformer translator:

```
transformer-en-it/
├── CLAUDE.md                     # Project guidance
├── config.py                     # Hyperparameters (d_model=512, d_ff=2048, n_heads=8, n_layers=6, dropout=0.1)
├── modules/                      # Core Transformer components (nn.Module classes)
│   ├── __init__.py
│   ├── embeddings.py             # InputEmbeddings (Module 1) - HAVE
│   ├── positional_encoding.py    # PositionalEncoding (Module 2) - HAVE
│   ├── layer_norm.py             # LayerNormalization (Module 3) - HAVE
│   ├── feed_forward.py           # FeedForwardBlock (Module 4) - HAVE
│   ├── attention.py              # MultiHeadAttention (Module 5) - HAVE
│   ├── residual.py               # ResidualConnection (Module 6) - HAVE
│   ├── encoder.py                # EncoderBlock, Encoder stack - TODO
│   └── decoder.py                # DecoderBlock, Decoder stack - TODO
├── model.py                      # Full Transformer (assembles encoder + decoder + output projection) - TODO
├── tokenizer.py                  # BPE or SentencePiece wrapper for EN/IT - TODO
├── dataset.py                    # EN-IT parallel corpus loading, batching, padding, masks - TODO
├── train.py                      # Training loop (cross-entropy loss, Adam, LR scheduler, checkpointing) - TODO
├── inference.py                  # Translation (greedy decoding or beam search) - TODO
├── utils.py                      # Helpers (mask creation, LR scheduler, weight init) - TODO
├── data/                         # Data directory
│   ├── raw/                      # Raw parallel corpus files
│   └── processed/                # Tokenized/preprocessed data
├── checkpoints/                  # Saved model weights
└── vocab/                        # Vocabulary files (en.vocab, it.vocab)
```

## Current Work

Setting up the initial files for the transformer architecture. This involves creating the foundational structure and components needed for a complete Transformer implementation.

### Progress

- **Created `transformer.py`** — Combined main transformer file containing all three core modules:
  - `PositionalEncoding` (Module 2): Sine/cosine positional embeddings
  - `LayerNormalization` (Module 3): Layer normalization with learnable alpha/bias parameters
  - `FeedForwardBlock` (Module 4): Two-layer feed-forward network with ReLU activation

  Note: This file currently contains the same syntax bugs as the original files for reference. These will need to be fixed before the code can run properly.

- **New lecture files added** — `CSCI_440-08-m+c_1.4.py` is now the most complete source, containing:
  - `InputEmbeddings` (Module 1): Token embeddings with sqrt(d_model) scaling - NEW
  - `MultiHeadAttentionBlock` (Module 5): Complete multi-head attention with masking - NEW
  - `ResidualConnection` (Module 6): Skip connections / Add & Norm layer - NEW

  With these additions, we now have all core building blocks except EncoderBlock and DecoderBlock (which compose the above modules).

### Code Quality Assessment

**Purpose**: The original code was written primarily for educational purposes, likely during lecture with focus on understanding concepts rather than execution.

**Architecture Quality**: Despite syntax bugs, the implementation is architecturally sound and correctly follows the "Attention Is All You Need" paper:
- InputEmbeddings correctly scales by sqrt(d_model) per paper section 3.4
- PositionalEncoding uses proper sine/cosine formulas with log-space computation for numerical stability
- LayerNormalization implements the standard approach with learnable parameters (alpha/bias)
- FeedForwardBlock matches the paper's FFN structure (two linear layers with ReLU and dropout)
- MultiHeadAttentionBlock correctly implements scaled dot-product attention with head splitting/concatenation
- ResidualConnection implements skip connections with pre-norm (differs slightly from paper but is common practice)

**Current State**:
- `CSCI_440-08-m+c_1.4.py` has ~12 syntax errors preventing execution
- Code was never tested/run (errors would have been caught immediately)
- Once bugs are fixed, the modules would be production-quality implementations
- The mathematical concepts and PyTorch patterns are correct

**Environment**: Likely written in Jupyter Notebook or basic text editor with no linting/syntax checking enabled, during coursework.
