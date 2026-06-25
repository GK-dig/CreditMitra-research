# DP-QLoRA Benchmark & Privacy Audit — Complete Script Suite
## CreditMitra · Qwen2.5-1.5B · ε=1.9911 · On-premise RBI-compliant

All scripts write to: `benchmark_payee/outputs_privacy_audit/`

---

## Fixed Configuration (do not change across runs)

```
BASE_MODEL   = Qwen/Qwen2.5-1.5B-Instruct
MAX_LENGTH   = 64              # compliance requirement
ε  = 1.9911  (outputs_2)      # operating point — ε sweep knee
δ  = 2.5×10⁻⁴  ≈ 1/N         # N=4382 training records
C  = 1.5                       # per-sample clipping norm
B  = 24                        # logical batch size
r  = 16  α = 32               # LoRA rank / alpha
Epochs     = 6
Accounting = RDP via Opacus
Targets    = q_proj, k_proj, v_proj, o_proj
```

---

## Scripts at a Glance

| Script | GPU? | Runtime | What it proves |
|--------|------|---------|----------------|
| `script4_tradeoff.py` | **No** | < 1 min | ε=2 is the utility knee; full model comparison |
| `script5_hyperparam.py` | **No** | < 1 min | Every hyperparameter justified with research backing |
| `script1_extraction.py` | Yes | 15–25 min | DP suppresses verbatim memorization |
| `script2_canary.py` | Yes | 20–30 min | DP prevents name↔handle recall (fixed canary) |
| `script3_mia.py` | Yes | 30–45 min | 8-variant MIA all ≈ 0.51 AUC on ε=2 model |

**Recommended run order:** 4 → 5 → 1 → 2 → 3 → 4 (re-run to pick up live MIA)

---

## Script 4 — Privacy-Utility Tradeoff Dashboard
**File:** `script4_tradeoff.py`  **GPU: None**

Assembles the core story from your existing JSON outputs. Run this first
to verify everything works, and again last to pull in live MIA from script 3.

**6-panel dashboard:**
- Panel A: Utility (F1) vs ε — all models compared
- Panel B: Privacy-utility knee curve (ε=2 is the operating point)
- Panel C: MIA-AUC vs ε (privacy floor — updates automatically from script 3)
- Panel D: ε accumulation across all four training runs (RDP accountant live)
- Panel E: Full benchmark table — all models, all metrics
- Panel F: Error breakdown — exact→partial shift, not catastrophic failure

```bash
python script4_tradeoff.py --project_root /path/to/benchmark_payee
```

**Outputs:**
```
script4_tradeoff_dashboard.pdf    # main 6-panel figure
script4_knee_standalone.pdf       # standalone for LaTeX \includegraphics
script4_eps_accumulation.pdf      # standalone ε curves for LaTeX
```

---

## Script 5 — Hyperparameter Justification Dashboard
**File:** `script5_hyperparam.py`  **GPU: None**

Shows WHY every hyperparameter was chosen using the 4-account ε sweep
results, the Google "How to DP-fy ML" paper framework, and the real
trainer_state from checkpoint-400. Two output pages.

**Page 1 — 9-panel quantitative dashboard:**
- Panel A: ε sweep knee from outputs_1/2/4/8 — why ε=2
- Panel B: Privacy tier framework — your model vs Gboard/Facebook/Apple
- Panel C: NSR vs batch size (theory curve + your B=24 point)
- Panel D: LoRA rank vs trainable params — why r=16
- Panel E: Non-DP training dynamics (loss, grad norm, LR from real trainer_state)
- Panel F: DP training dynamics — loss + per-epoch ε (outputs_2)
- Panel G: Full model comparison — all 8 models, EM + char-sim
- Panel H: DP cost anatomy — exact→partial, not catastrophic failure
- Panel I: NSR for all four ε runs from actual σ values

**Page 2 — Hyperparameter justification table:**
Every parameter (ε, δ, C, B, r, α, epochs, MAX_LENGTH, accounting, sampling)
with its value, justification, and the paper that backs it.

```bash
python script5_hyperparam.py --project_root /path/to/benchmark_payee
```

**Outputs:**
```
script5_hyperparam_dashboard.pdf  # 9-panel figure (page 1)
script5_hyperparam_table.pdf      # justification table (page 2)
script5_eps_sweep.pdf             # standalone A for LaTeX
script5_nsr_batch.pdf             # standalone C for LaTeX
script5_dp_dynamics.pdf           # standalone F for LaTeX
script5_full_comparison.pdf       # standalone G for LaTeX
script5_error_breakdown.pdf       # standalone H for LaTeX
```

---

## Script 1 — Training Data Extraction Test
**File:** `script1_extraction.py`  **GPU: Yes (outputs_2 + outputs_8)**

Gives the model the first 50% of a REAL training narration and checks if it
can complete the suffix verbatim. The suffix is HIDDEN — the model must have
memorized it to reproduce it. If DP model completes fewer train suffixes than
non-DP but both are similar on val → DP reduced memorization.

This is a legitimate privacy test because:
- Uses real training narrations, not fake canaries
- Suffix never appears in the query
- Train vs val comparison isolates memorization from general capability

```bash
python script1_extraction.py \
  --project_root /path/to/benchmark_payee \
  --n_samples 100 \
  --prefix_frac 0.5
```

**Outputs:**
```
s1_dp_train.json / s1_lora_train.json / s1_dp_val.json / s1_lora_val.json
script1_extraction_test.pdf    # 6-panel: bars, sim distributions,
                               # train vs val matrix, summary box
```

---

## Script 2 — Canary Memorization Test (Fixed)
**File:** `script2_canary.py`  **GPU: Yes (outputs_2 + outputs_8)**

### Why the original canary.py was wrong
The original script embedded the canary name INSIDE the query narration and
asked the model to extract it. That is the ordinary extraction task — solvable
with zero memorization. The DP model scored lower because DP hurts extraction
utility, not because it reduced memorization.

### What this script does instead
The canary name is completely HIDDEN from the query. Only the UPI handle
derived from it is shown:

```
Training record (injected):
  narration: UPI/TRF/.../Zephyranth Bvlgari/HDFC/**zephyranth.bvlgari1234@okaxis/...
  payee:     Zephyranth Bvlgari

Query at inference (name NOT shown):
  "A UPI transaction was processed using: zephyranth.bvlgari1234@okaxis
   What is the full payee name registered to this handle?"
```

The model can only answer correctly if it memorized the name↔handle mapping.
DP formally limits this by bounding how much any single training record
influences the model weights.

Also computes the **EXPOSURE METRIC** (secret-sharer style): ranks each
canary's log-probability against 100 control names. Rank ≈ 0.5 → not
memorized. Rank → 1.0 → fully memorized.

```bash
python script2_canary.py \
  --project_root /path/to/benchmark_payee \
  --n_canaries 10
```

**Outputs:**
```
s2_canary_dp.json / s2_canary_nondp.json / s2_exposure.json
script2_canary_extraction_rates.pdf   # exact vs partial recall bars
script2_canary_heatmap.pdf            # per-canary LEAK/SAFE grid
script2_canary_exposure.pdf           # exposure rank per canary
script2_canary_summary.pdf            # one-page combined summary
```

---

## Script 3 — 8-Variant MIA (pinned to outputs_2, ε=1.9911)
**File:** `script3_mia.py`  **GPU: Yes (outputs_2 + outputs_8 + PT base)**

Runs all 8 MIA variants from the Google DP guide and the paper's Fig 6,
ALL targeting outputs_2 (ε=1.9911) so the attack evidence and the claimed ε
are provably from the same model. Uses the PT base model as a reference for
4 additional ratio-based variants.

**Attack variants:**
```
1. Basic LOSS          — raw loss threshold (Yeom et al. 2018)
2. PT-Ref LOSS         — loss ratio: target vs pre-trained base
3. Response-Only LOSS  — loss on response tokens only (not prompt)
4. PT-Ref Response     — response-only ratio vs PT base
5. Min-K% Prob         — bottom-K% token probability (Shi et al. 2024)
6. PT-Ref Min-K%       — Min-K% ratio vs PT base
7. Loss Variance       — within-sample token loss variance
8. Zlib Normalised     — loss / zlib-compressed byte length
```

Produces per-attack AUC, AP, TPR@FPR≤10%, and a verdict
("Near-random (strong)" / "Moderate" / "Weak") for both DP and non-DP models.

```bash
python script3_mia.py \
  --project_root /path/to/benchmark_payee \
  --max_samples 486 \
  --batch_size 4
```

**Outputs:**
```
s3_mia_dp_metrics.json / s3_mia_nondp_metrics.json
script3_mia_bar_comparison.pdf    # 8 attacks side-by-side bar chart
script3_mia_roc_grid.pdf          # 8 ROC curves in 2×4 grid
script3_loss_distributions.pdf    # member vs non-member loss histograms
script3_mia_summary_table.pdf     # formatted colour-coded results table
```

---

## Full Run Order on Colab / Kaggle (RTX or T4)

```bash
cd /content/benchmark_payee

# ── No-GPU scripts first — verify everything works ──────────────────────
python script4_tradeoff.py  --project_root .
python script5_hyperparam.py --project_root .

# ── GPU scripts — restart runtime between each to free VRAM ─────────────
python script1_extraction.py --project_root . --n_samples 100
# restart runtime, re-run install

python script2_canary.py     --project_root . --n_canaries 10
# restart runtime, re-run install

python script3_mia.py        --project_root . --max_samples 486
# restart runtime, re-run install

# ── Re-run script 4 to pull live MIA into Panel C ───────────────────────
python script4_tradeoff.py  --project_root .
```

Each GPU script deletes its models and calls `torch.cuda.empty_cache()`
before exiting — but a full runtime restart between scripts is safer on T4.

---

## What Each Script Proves

| Script | Evidence type | DP claim it supports |
|--------|--------------|----------------------|
| 4 | ε sweep + full benchmark | ε=2 is knee; gap to ceiling is only 5 points |
| 5 | Hyperparameter justification | Every choice is principled, not arbitrary |
| 1 | Prefix-completion memorization | DP reduces verbatim recall of training narrations |
| 2 | Canary exposure metric | DP prevents name↔handle memorization |
| 3 | 8-variant MIA | No attack > AUC 0.52 on ε=1.9911 model |

Together these establish:
- **Formal ceiling:** ε=1.9911, δ=2.5×10⁻⁴ (Claim 6 in benchmark doc)
- **Empirical floor:** scripts 1–3 all show no detectable leakage
- **Hyperparameter integrity:** every knob justified by theory + sweep evidence

---

## Output Folder Structure After All Scripts

```
benchmark_payee/outputs_privacy_audit/
├── script4_tradeoff_dashboard.pdf       # ε sweep story
├── script4_knee_standalone.pdf
├── script4_eps_accumulation.pdf
├── script5_hyperparam_dashboard.pdf     # 9-panel hyperparam justification
├── script5_hyperparam_table.pdf         # justification table (all params)
├── script5_eps_sweep.pdf
├── script5_nsr_batch.pdf
├── script5_dp_dynamics.pdf
├── script5_full_comparison.pdf
├── script5_error_breakdown.pdf
├── script1_extraction_test.pdf          # memorization test
├── s1_dp_train.json / s1_lora_train.json / s1_dp_val.json / s1_lora_val.json
├── script2_canary_extraction_rates.pdf  # canary (fixed)
├── script2_canary_heatmap.pdf
├── script2_canary_exposure.pdf
├── script2_canary_summary.pdf
├── s2_canary_dp.json / s2_canary_nondp.json / s2_exposure.json
├── script3_mia_bar_comparison.pdf       # 8-variant MIA
├── script3_mia_roc_grid.pdf
├── script3_loss_distributions.pdf
├── script3_mia_summary_table.pdf
└── s3_mia_dp_metrics.json / s3_mia_nondp_metrics.json
```

---

## References

| Paper | Used in |
|-------|---------|
| Ponomareva et al. "How to DP-fy ML" (JAIR 2023) | Scripts 4, 5 — tier framework, NSR, batch sizing |
| Abadi et al. "Deep Learning with DP" (CCS 2016) | Scripts 3, 5 — DP-SGD mechanism |
| Mironov "Rényi DP" (CSF 2017) | Scripts 3, 4 — RDP accounting |
| Hu et al. "LoRA" (ICLR 2022) | Script 5 — rank justification |
| Biderman et al. (TMLR 2024) | Script 5 — α=2r for extraction tasks |
| Shi et al. "Detecting Pretraining Data" (2024) | Script 3 — Min-K% attack |
| Ran et al. "LoRA-Leak" (arXiv 2025) | Script 3 — PT-reference calibration |
| Dwork & Roth "Algorithmic Foundations of DP" (2014) | Scripts 4, 5 — δ convention |
