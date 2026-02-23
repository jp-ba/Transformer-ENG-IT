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

Each file is a cumulative lecture snapshot — later files repeat earlier blocks and add new ones. The pattern across all files is `#I. Math` (theory) followed by `#II. Code` (implementation).

### Lecture File Catalog

- **`CSCI_440-02-m+c_1.1.py`** — Block 01 only: `InputEmbeddings`. Earlier version with more docstring commentary. Bug: `nn.module` (lowercase m).

- **`CSCI_440-03-m+c_1.7.py`** — Block 02: `PositionalEncoding`. Incomplete (marked "stopped here"). `forward` is outside the class.

- **`CSCI_440-04-m+c_1.1.py`** — Blocks 03–04: `LayerNormalization`, `FeedForwardBlock`. Bugs: `dim * -1`, stray `?`.

- **`CSCI_440-07-m+c_1.3.py`** — Block 05 (partial): `MultiHeadAttentionBlock` `__init__` only; attention calculation not yet implemented.

- **`CSCI_440-08-m+c_1.4.py`** — Blocks 01–05 + `ResidualConnection`. Most complete prior file; source used for `model.py` initial compilation. **`attention` method has `?*` bug (should be `@`).**

- **`CSCI_440-09-m+c_1.3.py`** — Blocks 01–05 + `ResidualConnection` + **`EncoderBlock` + `Encoder` stack** (NEW — Block 06 source). **`attention` method is FIXED here — uses correct `@` operator** (only clean version across all files). `ResidualConnection` and `EncoderBlock` are still nested inside `MultiHeadAttentionBlock` at wrong indentation level. `InputEmbeddings` missing `:` after class definition.

- **`CSCI_440-10-m+c_1.0.py`** — All above + **`DecoderBlock` + `Decoder` stack** (NEW — Block 07 source). `DecoderBlock` comment blocks between `class` line and methods are at 0 indent (outside class body). `?*` bug re-introduced in `attention` method. `InputEmbeddings` missing `:` after class definition.

- **`CSCI_440-10-m+c_1.1.py`** — Minor revision of `-1.0`. Adds `#todo` comment block near the top (instructor notes on session goals). **No bugs fixed** — all bugs from `-1.0` are identical: `nn.module`/`nn.embedding` lowercase, `torch.arrange`, `math.log(10000,0)`, `dim * -1`, stray `?`, `?*` in `attention`, missing `@staticmethod`, missing `=`, `ResidualConnection`/`EncoderBlock`/`Encoder` nested at 8-space inside MHA, `DecoderBlock` comment blocks at 0 indent. Not a better source than `-1.0` for any block.

- **`CSCI_440-11-m+c_1.3.py`** — All above + **`ProjectionLayer`** (NEW — Block 08 source). `ProjectionLayer` has no syntax bugs — clean implementation. `?*` bug persists. `InputEmbeddings` missing `:` (uses `nn.Module` correct case this time). **`class Transformer` (Block 09) is absent from all lecture files.**

- **`CSCI_440-12-m+c_1.0.py`** — All blocks 01–09 present. **`class Transformer` (Block 09) appears here for the first time — clean, no syntax bugs.** Same bugs as `-11` for blocks 01–05 (`nn.module`/`nn.embedding` lowercase, `torch.arrange`, `math.log(10000,0)`, `?*` bug in `attention`, missing `@staticmethod`, missing `=` on attention assignment, `PositionalEncoding` indentation + `forward` at module level, `dim * -1`). `ResidualConnection`, `EncoderBlock`, and `Encoder` still nested inside `MultiHeadAttentionBlock` at 8-space indent. `DecoderBlock` comment blocks are at 0 indent (outside class body); `Decoder` is at module level — correct. `ProjectionLayer` is clean. `build_transformer` factory function is absent — only truncated description comments exist (file ends mid-comment at line 540).

- **`CSCI_440-12-m+c_1.1.py`** — Minor revision of `-1.0`. Adds a `#I. Math` section at the top of the file with a `LogSoftmax` example and seaborn plotting code (requires `matplotlib` and `seaborn` imports not present in the `#II. Code` section). **No bugs fixed** — all bugs from `-1.0` are identical: `nn.module`/`nn.embedding` lowercase, `torch.arrange`, `math.log(10000,0)`, `dim * -1`, stray `?`, `?*` in `attention`, missing `@staticmethod`, missing `=`, `ResidualConnection`/`EncoderBlock`/`Encoder` nested at 8-space inside MHA, `DecoderBlock` comment blocks at 0 indent. `build_transformer` factory function still absent — file ends at the same mid-comment truncation point (line 571, `#hyperparam hidden layer (a feed fwd): d_ff =`). `class Transformer` remains clean. Not a better source than `-1.0` for any block.

### Best Source Per Block

| Block | Class | Best Source File | Notes |
|-------|-------|-----------------|-------|
| 01 | `InputEmbeddings` | `-08` | `-02` has more docs but same bugs |
| 02 | `PositionalEncoding` | `-08` | All files have same bugs |
| 03 | `LayerNormalization` | `-08` | All files identical |
| 04 | `MultiHeadAttentionBlock` | **`-09`** | Only file with correct `@` operator |
| 05 | `ResidualConnection` | `-09` | All files nest it inside MHA incorrectly |
| 06 | `EncoderBlock` + `Encoder` | `-09` | Both nested inside MHA incorrectly in all files |
| 07 | `DecoderBlock` + `Decoder` | `-10`/`-11`/`-12` | Comment blocks outside class body; `-10-1.0` and `-10-1.1` are identical for this block |
| 08 | `ProjectionLayer` | **`-11`** or **`-12`** | Clean — no bugs in either; `-12-1.0` and `-12-1.1` identical for this block |
| 09 | `class Transformer` | **`-12`** | First and only appearance; no syntax bugs; `build_transformer` absent in both `-1.0` and `-1.1` |

### Syntax Issues in `CSCI_440-08-m+c_1.4.py` (source for model.py compilation)

  | Line(s) | Block | Issue |
  |---------|-------|-------|
  | 73 | InputEmbeddings | `nn.embedding` → `nn.Embedding` (wrong case) |
  | 98–101 | PositionalEncoding | Only these 4 lines are inside `__init__`; everything after `self.dropout = nn.Dropout(dropout)` drops back to class body level |
  | 106–150 | PositionalEncoding | All at class body level (4-space indent) — `seq_len` and `d_model` are not in scope here; `self.register_buffer` at line 150 calls `self` outside any method |
  | 121, 124 | PositionalEncoding | `torch.arrange` → `torch.arange` (×2) |
  | 124 | PositionalEncoding | `math.log(10000,0)` → `math.log(10000.0)` (second arg is base; log base 0 is undefined) |
  | 144 | PositionalEncoding | `pe.unsqueeze(0) * (1, Seq_Len, d_model)` — can't multiply tensor by tuple; also `Seq_Len` capitalized (undefined; parameter is `seq_len`) |
  | 153–160 | PositionalEncoding | `def forward(self, x):` is at module level (0 indent) — a loose function, not a class method |
  | 198–199 | LayerNormalization | `dim * -1` → `dim=-1` (keyword argument, not multiplication) |
  | 203–210 | FeedForwardBlock | Comment block has 5-space indent — orphaned, not inside any class or function |
  | 226 | FeedForwardBlock | Stray `?` character starts the line — not a valid Python comment (`#` needed) |
  | 230–257 | MultiHeadAttentionBlock | Comment block has 5-space indent — orphaned, same issue as lines 203–210 |
  | 289 | MultiHeadAttentionBlock | `def attention(...)` is missing `@staticmethod` decorator |
  | 297 | MultiHeadAttentionBlock | `query ?* key.transpose(...)` — `?*` should be `@` (matmul operator) |
  | 352 | MultiHeadAttentionBlock | `x, self.attention_scores * MultiHeadAttentionBlock.attention(...)` — missing `=` (should be `x, self.attention_scores = ...`) |
  | 374–390 | ResidualConnection | Entire class is indented 8 spaces inside `MultiHeadAttentionBlock` — should be at module level (0 indent) |

## Conventions

- Files follow the naming pattern `CSCI_440-{module}-m+c_{version}.py`
- Each file has two sections: `#I. Math` (theory/reference notes) and `#II. Code` (implementation)
- Standard Transformer dimension names: `d_model` (512), `d_ff` (2048), `seq_len`, `dropout`
- All neural network components inherit from `nn.Module` using the constructor/forward pattern

## Assignment Deliverables

**Due: 02/24/2026 at 10:30am PST**

Submit two files on Canvas:
- `model_Lname_Fname.py` — all 9 blocks in a single file
- `model_Lname_Fname.docx` — screenshot of output for each block

**All code goes in one `model.py` file** — not split across a `modules/` directory.

## Required Blocks (from `Copy of template_CSCI_440_MC_Lname_Fname.md`)

| Block | Class | Status | model.py state |
|-------|-------|--------|----------------|
| 01 | `InputEmbeddings` | IN PROGRESS | Present; `nn.embedding` bug unfixed |
| 02 | `PositionalEncoding` | IN PROGRESS | Present; indentation + `torch.arrange` + log bug unfixed; `forward` at module level |
| 03 | `LayerNormalization` | IN PROGRESS | Present; `dim * -1` bug unfixed |
| 04 | `FeedForwardBlock` | IN PROGRESS | Present; stray `?` on line 173 unfixed |
| 04 | `MultiHeadAttentionBlock` | IN PROGRESS | Present; `?*` bug + missing `@staticmethod` + missing `=` unfixed |
| 05 | `ResidualConnection` | IN PROGRESS | Present but nested 8-space inside MHA — needs move to module level |
| 06 | `EncoderBlock` + `Encoder` | IN PROGRESS | Present (copied from `-09`); still nested 8-space inside MHA — needs move to module level |
| 07 | `DecoderBlock` + `Decoder` | IN PROGRESS | Present (copied from `-10`); `DecoderBlock` comment blocks at 0 indent (outside class body) — needs fix |
| 08 | `ProjectionLayer` | IN PROGRESS | Present (copied from `-12`); clean — no bugs |
| 09 | `class Transformer` | IN PROGRESS | Present (copied from `-12`); clean; `build_transformer` absent — must be written from scratch |

Note: `FeedForwardBlock` is not a standalone submission block but is required internally by `EncoderBlock` and `DecoderBlock`. Also needed: `Encoder` and `Decoder` stack classes (from `-09`/`-10`).

## Current Work

Building `model.py` — a single file containing all 9 required blocks for the CSCI 440 assignment.

### Progress (last updated: 2026-02-19)

- **`model.py`** exists as an untracked new file (not yet committed to git).
- Blocks 01–05 are present in `model.py`, sourced from `CSCI_440-08-m+c_1.4.py`. All syntax bugs from that source file are still present — none have been fixed yet.
- Block 05 (`ResidualConnection`) is nested at 8-space indent inside `MultiHeadAttentionBlock` in `model.py` — must be moved to module level.
- Blocks 06–09 were stub comments only at session start; all four blocks were added to `model.py` this session (2026-02-19), copied verbatim from their best source files. No bugs were fixed during the copy — all source file issues are present in `model.py` as copied.
  - Block 06 (`EncoderBlock` + `Encoder`) — from `-09`; still nested 8-space inside `MultiHeadAttentionBlock`
  - Block 07 (`DecoderBlock` + `Decoder`) — from `-10`; `DecoderBlock` comment blocks at 0 indent (outside class body)
  - Block 08 (`ProjectionLayer`) — from `-12`; clean, no bugs
  - Block 09 (`class Transformer`) — from `-12`; clean; ends with truncated `build_transformer` comment — must be written from scratch
- **`transformer.py`** (older combined file) — superseded by the assignment requirement to use `model.py`.
- **`CSCI_440-12-m+c_1.0.py`** discovered (2026-02-19) — most complete lecture file to date. Contains all 9 blocks. `class Transformer` (Block 09) appears here for the first time and is architecturally complete with no syntax bugs. `build_transformer` factory function is absent (truncated comments only).
- **`CSCI_440-10-m+c_1.1.py`** and **`CSCI_440-12-m+c_1.1.py`** discovered (2026-02-19) — minor revisions of their `-1.0` counterparts. Neither version fixes any bugs. `-10-1.1` adds only a `#todo` comment block. `-12-1.1` adds a `#I. Math` section with a LogSoftmax example and seaborn plot. `build_transformer` is still absent in `-12-1.1`. No change to best source recommendations for any block.

### Remaining Tasks

1. Fix all syntax bugs in `model.py` Blocks 01–05 (full bug list in the anomaly table above; none fixed yet)
2. Move `ResidualConnection` (Block 05) from 8-space indent inside MHA to module level (0 indent)
3. Fix `FeedForwardBlock` — stray `?` character must become a `#` comment [Done?]
4. Fix `EncoderBlock` + `Encoder` (Block 06) — move from 8-space inside MHA to module level (0 indent)
5. Fix `DecoderBlock` (Block 07) — move comment blocks from 0 indent to correct 4-space indent inside class body [Done?]
6. Write `build_transformer` factory function from scratch — absent from all lecture files; `class Transformer` (Block 09) is present and clean
7. Run each block and capture output screenshots for the `.docx` submission

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
