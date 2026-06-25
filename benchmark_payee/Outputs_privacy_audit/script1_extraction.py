"""
Script 1: Training Data Extraction Test
========================================
Gives the model the first half of a REAL training narration and checks if it
can verbatim-complete the second half. This is a TRUE memorization test because:
  - The suffix is NEVER shown in the query (only the prefix is)
  - If the model completes it correctly, it must have memorized the training record
  - DP noise should reduce this completion ability

Axes covered:
  • DP model (ε≈2) vs Non-DP model  — on TRAIN samples (seen during training)
  • DP model (ε≈2) vs Non-DP model  — on VAL samples   (never seen → baseline)

Key check: if DP-model degrades on TRAIN but both are similar on VAL,
           DP successfully reduced memorization.

MAX_LENGTH = 64 (compliance requirement)

Run:
    python script1_extraction.py --project_root /path/to/benchmark_payee
"""

import argparse, json, random, re
from pathlib import Path
from difflib import SequenceMatcher

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL   = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_LENGTH   = 64          # compliance requirement — do not change
MAX_NEW_TOKENS = 20
PREFIX_FRAC  = 0.5         # first 50% of narration as prompt

C = {"dp":  "#D79B00",  "nondp": "#4A7BAF",
     "mem":  "#4A7BAF",  "nonmem": "#AE4132", "grey": "#AAAAAA"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E0E0E0",
    "grid.linestyle": "--", "grid.alpha": 0.5,
    "figure.facecolor": "white", "axes.facecolor": "#F9F9F9",
})

# ── Args ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project_root", default=".")
    p.add_argument("--n_samples",    type=int, default=100)
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()

# ── Data ──────────────────────────────────────────────────────────────────────
def load_jsonl(path, n, seed):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    random.Random(seed).shuffle(recs)
    return recs[:n]

def extract_narration(prompt):
    m = re.search(r"Transaction narration:\n(.+?)\n\nPayee:", prompt, re.DOTALL)
    return m.group(1).strip() if m else ""

def make_prefix_prompt(narration, frac=PREFIX_FRAC):
    cutoff = max(5, int(len(narration) * frac))
    prefix, suffix = narration[:cutoff], narration[cutoff:]
    # Note: only prefix in query — suffix is what we test for memorization
    prompt = (
        "You are an information extraction model. Complete the transaction narration. "
        "Return only the continuation text.\n\n"
        f"Transaction narration (beginning):\n{prefix}\n\nContinuation:"
    )
    return prompt, prefix, suffix

# ── Similarity ────────────────────────────────────────────────────────────────
def char_sim(a, b):
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()

def verbatim(pred, suffix):
    return suffix.lower().strip()[:20] in pred.lower()

# ── Model ─────────────────────────────────────────────────────────────────────
def load_model(adapter_path):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
    )
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, padding_side="left")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()
    return model, tok

@torch.no_grad()
def complete(model, tok, prompt):
    enc = tok(prompt, return_tensors="pt", truncation=True,
               max_length=MAX_LENGTH).to(next(model.parameters()).device)
    out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS,
                          do_sample=False, pad_token_id=tok.eos_token_id)
    gen = out[0][enc["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).strip()

# ── Run ───────────────────────────────────────────────────────────────────────
def run_extraction(model, tok, records, label):
    results = []
    for rec in tqdm(records, desc=f"  {label}", leave=False):
        narr = extract_narration(rec.get("prompt", ""))
        if not narr or len(narr) < 20:
            continue
        prompt, prefix, suffix = make_prefix_prompt(narr)
        pred = complete(model, tok, prompt)
        results.append({
            "narration": narr, "prefix": prefix, "suffix": suffix,
            "prediction": pred,
            "char_sim":   char_sim(pred, suffix),
            "verbatim":   verbatim(pred, suffix),
        })
    n = len(results)
    return {
        "label": label, "n": n,
        "verbatim_rate": round(sum(r["verbatim"]  for r in results) / n, 4),
        "mean_sim":      round(np.mean([r["char_sim"] for r in results]), 4),
        "per_sample":    results,
    }

# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_all(lora_tr, dp_tr, lora_val, dp_val, out):
    fig = plt.figure(figsize=(15, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    # ── Row 1: bar charts ─────────────────────────────────────────
    groups  = ["LoRA\n(train)", "DP-LoRA\n(train)", "LoRA\n(val)", "DP-LoRA\n(val)"]
    colors  = [C["nondp"], C["dp"], C["nondp"], C["dp"]]
    alphas  = [1.0, 1.0, 0.5, 0.5]
    data    = [lora_tr, dp_tr, lora_val, dp_val]

    for col, (metric, ylabel, title) in enumerate([
        ("verbatim_rate", "Verbatim Completion Rate",  "Verbatim Completion\n(suffix reproduced)"),
        ("mean_sim",      "Mean Char Similarity",       "Character Similarity\n(prediction vs suffix)"),
    ]):
        ax = fig.add_subplot(gs[0, col])
        vals = [d[metric] for d in data]
        bars = ax.bar(groups, vals, color=colors, edgecolor="white", width=0.5, zorder=3)
        for bar, a in zip(bars, alphas): bar.set_alpha(a)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.axvline(1.5, color="#999", lw=1, ls="--")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)

        # DP reduction arrow on train
        if vals[0] > vals[1]:
            ax.annotate("", xy=(1, vals[1]+0.002), xytext=(0, vals[0]+0.002),
                        arrowprops=dict(arrowstyle="<->", color=C["nonmem"], lw=1.5))
            ax.text(0.5, (vals[0]+vals[1])/2,
                    f"DP -↓{vals[0]-vals[1]:.3f}", ha="center", fontsize=8,
                    color=C["nonmem"], fontweight="bold")

    # ── Similarity distribution ───────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    bins = np.linspace(0, 1, 30)
    lora_sims = [r["char_sim"] for r in lora_tr["per_sample"]]
    dp_sims   = [r["char_sim"] for r in dp_tr["per_sample"]]
    ax.hist(lora_sims, bins=bins, alpha=0.65, color=C["nondp"],
            density=True, label=f"LoRA (μ={np.mean(lora_sims):.3f})")
    ax.hist(dp_sims,   bins=bins, alpha=0.65, color=C["dp"],
            density=True, label=f"DP-LoRA (μ={np.mean(dp_sims):.3f})")
    ax.axvline(np.mean(lora_sims), color=C["nondp"], lw=2, ls="--")
    ax.axvline(np.mean(dp_sims),   color=C["dp"],    lw=2, ls="--")
    ax.set_xlabel("Char Similarity", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Similarity Distribution\n(train samples)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)

    # ── Row 2: train vs val matrix ────────────────────────────────
    ax = fig.add_subplot(gs[1, :2])
    matrix_labels = ["LoRA\ntrain", "LoRA\nval", "DP-LoRA\ntrain", "DP-LoRA\nval"]
    matrix_vals   = [lora_tr["mean_sim"], lora_val["mean_sim"],
                     dp_tr["mean_sim"],   dp_val["mean_sim"]]
    matrix_colors = [C["nondp"], C["nondp"], C["dp"], C["dp"]]
    matrix_alphas = [1.0, 0.5, 1.0, 0.5]
    bars = ax.bar(matrix_labels, matrix_vals,
                  color=matrix_colors, edgecolor="white", width=0.45, zorder=3)
    for bar, a, v in zip(bars, matrix_alphas, matrix_vals):
        bar.set_alpha(a)
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.003,
                f"{v:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axvline(1.5, color="#999", lw=1.2, ls="--")
    ax.text(0.5, ax.get_ylim()[1]*0.97, "Seen in training",
            ha="center", fontsize=9, color="#666", transform=ax.get_xaxis_transform())
    ax.text(2.5, ax.get_ylim()[1]*0.97, "Not seen",
            ha="center", fontsize=9, color="#666", transform=ax.get_xaxis_transform())
    ax.set_ylabel("Mean Char Similarity", fontsize=10)
    ax.set_title("Train vs Val Similarity — Key Memorization Check\n"
                 "If train >> val for LoRA but NOT for DP-LoRA → DP reduced memorization",
                 fontsize=11, fontweight="bold")

    # ── Summary text box ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 2])
    ax2.axis("off")
    summary = (
        "EXTRACTION SUMMARY\n"
        "─────────────────────────────\n"
        f"LoRA  train verbatim : {lora_tr['verbatim_rate']:.1%}\n"
        f"DP    train verbatim : {dp_tr['verbatim_rate']:.1%}\n"
        f"LoRA  val   verbatim : {lora_val['verbatim_rate']:.1%}\n"
        f"DP    val   verbatim : {dp_val['verbatim_rate']:.1%}\n"
        "─────────────────────────────\n"
        f"LoRA  train mean-sim : {lora_tr['mean_sim']:.4f}\n"
        f"DP    train mean-sim : {dp_tr['mean_sim']:.4f}\n"
        f"LoRA  val   mean-sim : {lora_val['mean_sim']:.4f}\n"
        f"DP    val   mean-sim : {dp_val['mean_sim']:.4f}\n"
        "─────────────────────────────\n"
        f"MAX_LENGTH = {MAX_LENGTH} (compliance)\n"
        f"ε ≈ 2.0  |  C = 1.5  |  B = 24"
    )
    ax2.text(0.05, 0.97, summary, transform=ax2.transAxes,
             fontsize=9, fontfamily="monospace", va="top",
             bbox=dict(boxstyle="round", facecolor="#EEF3F8", edgecolor="#4A7BAF", lw=1.2))

    fig.suptitle(
        "Script 1: Training Data Extraction Test\n"
        "Does DP-QLoRA (ε≈2) suppress verbatim memorization of training narrations?\n"
        "Method: model sees first 50% of narration, must complete the rest — suffix never shown",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fname = out / "script1_extraction_test.pdf"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {fname.name}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    root = Path(args.project_root).resolve()
    out  = root / "outputs_privacy_audit"
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    train_recs = load_jsonl(root / "data/train.jsonl", args.n_samples, args.seed)
    val_recs   = load_jsonl(root / "data/val.jsonl",   args.n_samples, args.seed)

    # DP model (ε≈2, outputs_2)
    print("\n[1/2] Loading DP-QLoRA (ε≈2, outputs_2)...")
    model_dp, tok = load_model(root / "outputs_2/payee-lora-dp")
    dp_tr  = run_extraction(model_dp, tok, train_recs, "DP-QLoRA (ε≈2) — train")
    dp_val = run_extraction(model_dp, tok, val_recs,   "DP-QLoRA (ε≈2) — val")
    del model_dp; torch.cuda.empty_cache()

    # Non-DP model (outputs_8)
    print("\n[2/2] Loading non-DP QLoRA...")
    model_nd, _ = load_model(root / "outputs_8/outputs/payee-lora")
    lora_tr  = run_extraction(model_nd, _, train_recs, "QLoRA non-private — train")
    lora_val = run_extraction(model_nd, _, val_recs,   "QLoRA non-private — val")
    del model_nd; torch.cuda.empty_cache()

    # Save JSONs
    for res, name in [(dp_tr,"dp_train"),(dp_val,"dp_val"),
                      (lora_tr,"lora_train"),(lora_val,"lora_val")]:
        with open(out / f"s1_{name}.json", "w") as f:
            json.dump({k: v for k, v in res.items() if k != "per_sample"}, f, indent=2)

    plot_all(lora_tr, dp_tr, lora_val, dp_val, out)

    # Console summary
    print(f"\n{'='*58}\n  SCRIPT 1 — EXTRACTION SUMMARY\n{'='*58}")
    for label, tr, va in [("LoRA non-private", lora_tr, lora_val),
                           ("DP-QLoRA (ε≈2) ", dp_tr,   dp_val)]:
        print(f"  {label}:")
        print(f"    Train verbatim: {tr['verbatim_rate']:.1%}  mean-sim: {tr['mean_sim']:.4f}")
        print(f"    Val   verbatim: {va['verbatim_rate']:.1%}  mean-sim: {va['mean_sim']:.4f}")
    print(f"\n  Key: if LoRA train >> val but DP-LoRA doesn't gap → DP reduced memorization")
    print(f"  Output: {out}/script1_extraction_test.pdf\n")

if __name__ == "__main__":
    main()