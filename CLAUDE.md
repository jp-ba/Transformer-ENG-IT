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
│   ├── embeddings.py             # InputEmbeddings (token embeddings) - TODO
│   ├── positional_encoding.py    # PositionalEncoding (Module 2) - HAVE
│   ├── layer_norm.py             # LayerNormalization (Module 3) - HAVE
│   ├── feed_forward.py           # FeedForwardBlock (Module 4) - HAVE
│   ├── attention.py              # MultiHeadAttention (Module 5) - TODO
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

### Code Quality Assessment

**Purpose**: The original code was written primarily for educational purposes, likely during lecture with focus on understanding concepts rather than execution.

**Architecture Quality**: Despite syntax bugs, the implementation is architecturally sound and correctly follows the "Attention Is All You Need" paper:
- PositionalEncoding uses proper sine/cosine formulas with log-space computation for numerical stability
- LayerNormalization implements the standard approach with learnable parameters (alpha/bias)
- FeedForwardBlock matches the paper's FFN structure (two linear layers with ReLU and dropout)

**Current State**:
- Has 5-6 syntax errors preventing execution
- Code was never tested/run (errors would have been caught immediately)
- Once bugs are fixed, the modules would be production-quality implementations
- The mathematical concepts and PyTorch patterns are correct

**Environment**: Likely written in Jupyter Notebook or basic text editor with no linting/syntax checking enabled, during coursework.
