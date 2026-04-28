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

**Model file (assignment submission):**
```bash
python model_BeloAlmuete_JosephPaul.py
```

**Training (once pipeline files are created and bugs fixed):**
```bash
python train.py
```

**Dependencies:** `torch` (PyTorch), `math` (stdlib)

**Training pipeline dependencies (pip install):**
```
pip install torch datasets tokenizers tensorboard tqdm torchmetrics
```

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

- **`CSCI_440-12-m+c_1.1.py`** — Minor revision of `-1.0`. Adds a `#I. Math` section at the top of the file with a `LogSoftmax` example and seaborn plotting code (requires `matplotlib` and `seaborn` imports not present in the `#II. Code` section). **No bugs fixed** — all bugs from `-1.0` are identical. `build_transformer` still absent. Not a better source than `-1.0` for any block.

- **`CSCI_440-13-m+c_1.3.py`** — All blocks 01–09 with same bugs as `-12`, **PLUS `build_transformer` factory function (NEW — Block 10 source)**. `build_transformer` is complete and clean (lines 570–628): creates embeddings, PE, N encoder/decoder blocks, projection layer, initializes weights with Xavier uniform. Also includes lecture notes on dataset/training pipeline setup at the end (lines 631+). **Best source for `build_transformer`.**

- **`CSCI_440-15-m+c_train_1.1.py`** — First version of `train.py` content. Contains `greedy_decode`, `run_validation`, `get_all_sentences`, `get_or_build_tokenizer`, `get_ds`, `get_model`. `get_all_sentences` is indented inside `run_validation` (indentation bug). `WorldLevelTrainer` import name is wrong (should be `WordLevelTrainer`).

- **`CSCI_440-16_m+c_dataset_1.3.py`** — **`dataset.py` source**: `BilingualDataset` class + `causal_mask` function. Has 3 bugs (see dataset.py bugs table below). `CSCI_440-16_m+c_dataset_mylist_1.1.py` is a standalone Python demo of `__getitem__` — not part of the pipeline.

- **`CSCI_440-17-m+c_config_1.1.py`** — **`config.py` source**: `get_config()` dict and `get_weights_file_path()`. Bug: `get_weights_file_path` references `config['model_basename']` but `get_config()` has key `'model_filename'` — must be reconciled (either rename the key or fix the reference).

- **`CSCI_440-18-m+c_1.1.py`** — Cumulative train.py content. Adds `train_model` function stub with device setup, dataset load, model build, TensorBoard, Adam optimizer, preload block, and `loss_fn`. Two syntax bugs: `print(f'Using device {device}'')` (extra quote), `.get_vocab_size().to(device)` called on tokenizer instead of model.

- **`CSCI_440-19-m+c_1.1.py`** — Adds epoch/batch training loop to `train_model`. Adds `global_step`, `loss.backward()`, `optimizer.step()`, epoch-end `torch.save`. Bug: `time.sleep(180)` references unimported `time` module. `if __name__ == '__main__':` block is indented inside `train_model` (should be module-level).

- **`CSCI_440-20-m+c_1.1.py`** — Same content as `-19`.

- **`CSCI_440-21-m+c_1.1.py`** — **Cleanest version of `train_model`**. Fixes `.get_vocab_size().to(device)` bug. Still has: `loss_fn` inside `if config['preload']:` block (should be outside); `if __name__ == '__main__':` still inside `train_model`; missing `model.load_state_dict(state['model_state_dict'])` in preload section. **Best source for `train.py`.**

- **`CSCI_440-23-m+c_1.1.py`** — Adds updated `greedy_decode` / `run_validation` and moves `get_all_sentences` to module level (fixing the indentation bug from `-15`). Bug: `get_all_sentences` is still indented inside `run_validation` in this file (lines 175–179 — same bug persists). Use `-21` as the base and fix `get_all_sentences` manually.

### Best Source Per Block

| Block | Class / Function | Best Source File | Notes |
|-------|-----------------|-----------------|-------|
| 01 | `InputEmbeddings` | `-08` | `-02` has more docs but same bugs |
| 02 | `PositionalEncoding` | `-08` | All files have same bugs |
| 03 | `LayerNormalization` | `-08` | All files identical |
| 04 | `FeedForwardBlock` | `-08` | Clean after fixing stray `?` |
| 04 | `MultiHeadAttentionBlock` | **`-09`** | Only file with correct `@` operator |
| 05 | `ResidualConnection` | `-09` | All files nest it inside MHA incorrectly |
| 06 | `EncoderBlock` + `Encoder` | `-09` | Both nested inside MHA incorrectly in all files |
| 07 | `DecoderBlock` + `Decoder` | `-10`/`-11`/`-12` | Comment blocks outside class body; `-10-1.0` and `-10-1.1` identical |
| 08 | `ProjectionLayer` | **`-11`** or **`-12`** | Clean — no bugs in either |
| 09 | `class Transformer` | **`-12`** / **`-13`** | First appearance; no syntax bugs; identical in both |
| 10 | `build_transformer` | **`-13`** | First and only appearance; clean (lines 570–628) |

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

**Due: 02/24/2026 at 10:30am PST** (already passed — training pipeline is ongoing coursework)

Submit two files on Canvas:
- `model_Lname_Fname.py` — all 9 blocks in a single file → **`model_BeloAlmuete_JosephPaul.py`**
- `model_Lname_Fname.docx` — screenshot of output for each block

**All model code goes in one file** — `model_BeloAlmuete_JosephPaul.py`.

## Required Blocks (from `Copy of template_CSCI_440_MC_Lname_Fname.md`)

| Block | Class | Status | Current state in model file |
|-------|-------|--------|-----------------------------|
| 01 | `InputEmbeddings` | DONE | `nn.Embedding` fixed; clean |
| 02 | `PositionalEncoding` | DONE | `torch.arange`, log, indentation, `forward` all fixed; clean |
| 03 | `LayerNormalization` | DONE | `dim=-1` fixed; clean |
| 04 | `FeedForwardBlock` | DONE | Stray `?` fixed; clean |
| 04 | `MultiHeadAttentionBlock` | **BUG** | `*` instead of `@` (line 245); missing `@staticmethod` (line 237) |
| 05 | `ResidualConnection` | **BUG** | Class at 0 indent (correct) but methods indented 12-space instead of 4-space |
| 06 | `EncoderBlock` + `Encoder` | **BUG** | Classes at 0 indent (correct) but methods indented 8-space instead of 4-space |
| 07 | `DecoderBlock` + `Decoder` | DONE | Indentation fixed; clean |
| 08 | `ProjectionLayer` | DONE | Clean — no bugs |
| 09 | `class Transformer` | DONE | Clean |
| 10 | `build_transformer` | **MISSING** | Absent from model file; must be added from `-13` lines 570–628 |

Note: `FeedForwardBlock` is not a standalone submission block but is required internally by `EncoderBlock` and `DecoderBlock`.

## Training Pipeline

The training pipeline requires three additional files that **do not yet exist** and must be created from the lecture source files.

### Files to Create

| File | Source Lecture | Status |
|------|---------------|--------|
| `model_BeloAlmuete_JosephPaul.py` | Lectures 02–13 | Exists; bugs remain (see above) |
| `dataset.py` | `CSCI_440-16_m+c_dataset_1.3.py` | **Not yet created** |
| `config.py` | `CSCI_440-17-m+c_config_1.1.py` | **Not yet created** |
| `train.py` | `CSCI_440-21-m+c_1.1.py` (best) | **Not yet created** |

### Bugs to Fix When Creating `dataset.py`

Source: `CSCI_440-16_m+c_dataset_1.3.py`

| Line(s) | Issue | Fix |
|---------|-------|-----|
| 41–43 | `torch.Tensor([tokenizer_src.token_to_id(['[SOS]'])], dtype=...)` — wrong constructor, extra bracket wrapping the token | `torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)` (lowercase `tensor`, remove extra list wrapping the string) |
| 72 | `len(enc_input_tokens - 2)` — subtracts from the list, not from its length | `len(enc_input_tokens) - 2` |
| 75 | `len(dec_input_tokens - 1)` — same issue | `len(dec_input_tokens) - 1` |
| 97 | `[self.pad_token] * enc_num_padding_tokens` — `self.pad_token` is a tensor; multiplication fails | store pad as int: `self.pad_token = tokenizer_src.token_to_id('[PAD]')` (no `torch.tensor` wrapper), or call `.item()` at use site |

### Bugs to Fix When Creating `config.py`

Source: `CSCI_440-17-m+c_config_1.1.py`

| Issue | Fix |
|-------|-----|
| `get_weights_file_path` references `config['model_basename']` but `get_config()` dict has key `'model_filename'` | Either rename dict key to `'model_basename'` or change the reference in `get_weights_file_path` to `config['model_filename']` — must be consistent |

### Bugs to Fix When Creating `train.py`

Best source: `CSCI_440-21-m+c_1.1.py`; supplement `get_all_sentences` from anywhere (fix indentation manually).

| Location | Issue | Fix |
|----------|-------|-----|
| Import | `from tokenizers.trainers import WorldLevelTrainer` — class name wrong | `from tokenizers.trainers import WordLevelTrainer` |
| `train_model` line ~18 | `print(f'Using device {device}'')` — extra closing quote causes SyntaxError | `print(f'Using device {device}')` |
| `train_model` line ~22 | `loss_fn = ...` is inside the `if config['preload']:` block (wrong indentation) | Dedent `loss_fn` to be at the same level as the `if` block — it must be defined unconditionally |
| preload block | `model.load_state_dict(state['model_state_dict'])` is missing | Add this line after `state = torch.load(model_filename)` |
| `train_model` line ~24 | `print(f'Preloading model {model_filename}'')` — extra closing quote | `print(f'Preloading model {model_filename}')` |
| `get_all_sentences` | Function is indented inside `run_validation` in all lecture files | Move to module level (0 indent) — it is a standalone generator |
| `if __name__ == '__main__':` | Block is indented inside `train_model` function body | Move to module level (0 indent) |
| `time.sleep(180)` (Lecture 19 version) | `time` module not imported | Remove the line or add `import time` at the top |

## Current Work

Building the full training pipeline: `model_BeloAlmuete_JosephPaul.py` + `dataset.py` + `config.py` + `train.py`.

### Progress (last updated: 2026-04-28)

- **`model_BeloAlmuete_JosephPaul.py`** exists. Blocks 01–04 and 07–09 are clean. Remaining bugs: `*` vs `@` in `attention` (line 245), missing `@staticmethod` (line 237), `ResidualConnection` method indentation (12-space → 4-space), `EncoderBlock`/`Encoder` method indentation (8-space → 4-space). `build_transformer` is absent and must be added from `CSCI_440-13-m+c_1.3.py` lines 570–628.
- **`dataset.py`** — not yet created.
- **`config.py`** — not yet created.
- **`train.py`** — not yet created.
- **`transformer.py`** (older combined file) — superseded; ignore.
- Lecture files 13–23 discovered (2026-04-28): cover `build_transformer`, `dataset.py`, `config.py`, and `train.py` pipeline. All have syntax bugs documented above.

### Remaining Tasks

#### Step 1 — Fix `model_BeloAlmuete_JosephPaul.py`

1. Fix `attention` method: change `query * key.transpose(-2, -1)` → `query @ key.transpose(-2, -1)` (line ~245)
2. Add `@staticmethod` decorator above `def attention(query, key, value, mask, dropout: nn.Dropout):` (line ~237)
3. Fix `ResidualConnection` class body indentation: dedent all methods from 12-space to 4-space
4. Fix `EncoderBlock` and `Encoder` class body indentation: dedent all methods from 8-space to 4-space
5. Add `build_transformer` function from `CSCI_440-13-m+c_1.3.py` lines 570–628 (after `class Transformer`)

#### Step 2 — Create `dataset.py`

6. Create `dataset.py` from `CSCI_440-16_m+c_dataset_1.3.py`; fix all 4 bugs in the table above

#### Step 3 — Create `config.py`

7. Create `config.py` from `CSCI_440-17-m+c_config_1.1.py`; fix `model_basename` / `model_filename` key mismatch

#### Step 4 — Create `train.py`

8. Create `train.py` from `CSCI_440-21-m+c_1.1.py`; fix all 7 bugs in the table above

#### Step 5 — Install dependencies

9. `pip install torch datasets tokenizers tensorboard tqdm torchmetrics`

#### Step 6 — Run and validate

10. Run `python train.py` — expected behavior on first run: downloads opus_books en-it dataset, builds tokenizers, creates `weights/` folder, starts training epochs with tqdm progress bar showing loss, runs validation every epoch printing SOURCE / TARGET / PREDICTED translations
11. Monitor loss in TensorBoard: `tensorboard --logdir runs/`
12. Capture screenshots of each block's output for `.docx` submission

### Code Quality Assessment

**Purpose**: The original code was written primarily for educational purposes, likely during lecture with focus on understanding concepts rather than execution.

**Architecture Quality**: Despite syntax bugs, the implementation is architecturally sound and correctly follows the "Attention Is All You Need" paper:
- InputEmbeddings correctly scales by sqrt(d_model) per paper section 3.4
- PositionalEncoding uses proper sine/cosine formulas with log-space computation for numerical stability
- LayerNormalization implements the standard approach with learnable parameters (alpha/bias)
- FeedForwardBlock matches the paper's FFN structure (two linear layers with ReLU and dropout)
- MultiHeadAttentionBlock correctly implements scaled dot-product attention with head splitting/concatenation
- ResidualConnection implements skip connections with pre-norm (differs slightly from paper but is common practice)
- build_transformer: Xavier uniform initialization, correct N=6 layers, h=8 heads, d_ff=2048

**Training config defaults** (from `config.py`): batch_size=8, num_epochs=20, lr=1e-4, seq_len=350, d_model=512, en→it translation, opus_books dataset.

**Environment**: Likely written in Jupyter Notebook or basic text editor with no linting/syntax checking enabled, during coursework.
