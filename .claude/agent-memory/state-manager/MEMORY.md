# State Manager Memory — transformer project

## Project Identity
- CSCI 440 coursework, PyTorch Transformer from scratch
- Single deliverable: `model.py` (all 9 blocks), due 2026-02-24 at 10:30am PST
- Primary context file: `C:\Users\JP Oleb\Documents\Machine Learning\Projects\transformer\CLAUDE.md`

## CLAUDE.md Structure (stable sections, in order)
1. Project Overview
2. Running
3. Architecture
   - Lecture File Catalog (one bullet per file, cumulative)
   - Best Source Per Block (table)
   - Syntax Issues in `-08` (table, line-level detail — source file for model.py)
4. Conventions
5. Assignment Deliverables
6. Required Blocks (status table)
7. Current Work
   - Progress (last updated: date)
   - Remaining Tasks (numbered)
   - Code Quality Assessment

## Update Patterns
- Lecture File Catalog: append new entries at the bottom of the bullet list
- Best Source Per Block table: update the row for any block whose best source changes
- Required Blocks table: update Status and model.py state columns only
- Progress section: append new bullet; update "last updated" date in the heading
- Remaining Tasks: edit individual numbered items in place; do not renumber

## Key Project Facts (verified)
- `-09` is the only lecture file with the correct `@` matmul operator in `attention` — use for Block 04/05/06
- `-12` is the only lecture file containing `class Transformer` (Block 09) — clean, no syntax bugs
- `build_transformer` factory function is absent from ALL lecture files (confirmed in `-10-1.0`, `-10-1.1`, `-12-1.0`, `-12-1.1`) — must be written from scratch
- `ResidualConnection`, `EncoderBlock`, `Encoder` are nested at 8-space indent inside `MultiHeadAttentionBlock` in every lecture file — always needs structural fix when copied to model.py
- `model.py` has all bugs from `-08` unfixed as of 2026-02-19; Blocks 06–09 are stub comments only
- Minor version increments (e.g. `-1.0` to `-1.1`) in lecture files add only instructor notes or math examples — no bugs are ever fixed between minor versions

## User Preferences
- Surgical edits preferred — update only changed sections, preserve existing wording/style
- No emojis in files
- ISO 8601 dates (YYYY-MM-DD) in progress notes
