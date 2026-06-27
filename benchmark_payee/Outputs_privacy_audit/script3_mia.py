"""
Script 3: 8-Variant Membership Inference Attack — outputs_2 (ε≈2)
===================================================================
Runs all eight attack variants from the MIA literature and from the
Google DP guide, ALL targeting the SAME outputs_2 (ε≈2) adapter
so the attack evidence and the claimed ε are provably from one model.

Attack variants (mirrors the paper's Fig 6 exactly):
  1. Basic LOSS            — raw loss threshold
  2. PT-Ref LOSS           — loss ratio vs the base (pre-trained) model
  3. Response-Only LOSS    — loss on response tokens only (not prompt)
  4. PT-Ref Response       — ratio on response tokens only
  5. Min-K% Prob           — min-K% token probability (Shi et al. 2024)
  6. PT-Ref Min-K%         — min-K% ratio vs PT model
  7. Loss Variance         — within-sample token loss variance
  8. Zlib Normalised       — loss / zlib-compressed length

For each attack:
  - Runs on DP model (outputs_2) AND non-DP model (outputs_8)
  - Reports: AUC, AP, TPR@FPR≤10%, accuracy
  - Generates: ROC curves, bar comparison, loss distributions, summary

All results pinned to outputs_2 (ε=1.9911).
MAX_LENGTH = 64 (compliance).

Run:
    python script3_mia.py --project_root /path/to/benchmark_payee
"""

import argparse, json, random, gc, zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_LENGTH = 64          # compliance
K_PERCENT  = 0.20        # Min-K%: bottom 20% of tokens

C = {"dp": "#D79B00", "nondp": "#4A7BAF", "random": "#AAAAAA",
     "mem": "#4A7BAF", "nonmem": "#AE4132"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E0E0E0",
    "grid.linestyle": "--", "grid.alpha": 0.5,
    "figure.facecolor": "white", "axes.facecolor": "#F9F9F9",
})

ATTACK_NAMES = [
    "Basic\nLOSS", "PT-Ref\nLOSS", "Resp-Only\nLOSS", "PT-Ref\nResp",
    "Min-K%\nProb", "PT-Ref\nMin-K%", "Loss\nVariance", "Zlib\nNorm",
]

# ── Args ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project_root", default=".")
    p.add_argument("--max_samples",  type=int, default=486)
    p.add_argument("--batch_size",   type=int, default=4)
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()

# ── Data ──────────────────────────────────────────────────────────────────────
def load_jsonl(path, n, seed):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    random.Random(seed).shuffle(recs)
    return recs[:n]

def fmt(rec):
    if "prompt" in rec and "response" in rec:
        return rec["prompt"].strip(), rec["response"].strip()
    if "input" in rec and "output" in rec:
        return rec["input"].strip(), rec["output"].strip()
    txt = " ".join(str(v) for v in rec.values() if isinstance(v, str))
    return txt, ""

class PairDataset(Dataset):
    """Returns (full_ids, prompt_len) for response-only attacks."""
    def __init__(self, records, tok, max_len):
        self.items = []
        for rec in records:
            prompt, response = fmt(rec)
            full   = tok(prompt + "\n" + response, max_length=max_len,
                         truncation=True, padding="max_length",
                         return_tensors="pt")["input_ids"].squeeze(0)
            prompt_ids = tok(prompt, max_length=max_len, truncation=True,
                              return_tensors="pt")["input_ids"].squeeze(0)
            prompt_len = int((prompt_ids != tok.pad_token_id).sum())
            raw_text = (prompt + "\n" + response).encode("utf-8")
            self.items.append((full, prompt_len, raw_text))

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

def collate(batch):
    ids       = torch.stack([b[0] for b in batch])
    plens     = [b[1] for b in batch]
    raw_texts = [b[2] for b in batch]
    return ids, plens, raw_texts

# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(path):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, padding_side="right")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(base, str(path))
    model.eval()
    return model, tok

def load_base_model(tok):
    """Load the bare base model as PT reference."""
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto", trust_remote_code=True)
    base.eval()
    return base

# ── Per-sample features ───────────────────────────────────────────────────────
@torch.no_grad()
def compute_features(model, loader, pad_id, desc="features"):
    """
    Returns dict of np.ndarray per sample:
        full_loss, resp_loss, min_k_prob, loss_variance, zlib_ratio
    """
    all_full_loss = []
    all_resp_loss = []
    all_mink      = []
    all_var       = []
    all_zlib      = []

    dev = next(model.parameters()).device

    for ids, plens, raw_texts in tqdm(loader, desc=f"  {desc}", leave=False):
        ids    = ids.to(dev)
        labels = ids.clone()
        labels[labels == pad_id] = -100
        out    = model(input_ids=ids, labels=labels)
        logits = out.logits

        shift_l = logits[:, :-1].contiguous()
        shift_r = labels[:, 1:].contiguous()
        tok_loss = F.cross_entropy(
            shift_l.view(-1, shift_l.size(-1)), shift_r.view(-1),
            reduction="none",
        ).view(shift_r.shape)

        mask = (shift_r != -100).float()

        # 1. Full loss
        full_loss = (tok_loss * mask).sum(1) / mask.sum(1).clamp(min=1)
        all_full_loss.extend(full_loss.cpu().tolist())

        # 2. Response-only loss
        for b_idx, p_len in enumerate(plens):
            resp_mask = mask[b_idx].clone()
            resp_mask[:max(0, p_len-1)] = 0
            denom = resp_mask.sum().clamp(min=1)
            rl = (tok_loss[b_idx] * resp_mask).sum() / denom
            all_resp_loss.append(rl.cpu().item())

        # 3. Min-K% token probability
        log_probs = F.log_softmax(shift_l, dim=-1)
        for b_idx in range(ids.size(0)):
            m = mask[b_idx]
            valid_ids = shift_r[b_idx][m.bool()]
            if len(valid_ids) == 0:
                all_mink.append(0.0); continue
            lp = log_probs[b_idx][m.bool(), valid_ids]
            k  = max(1, int(len(lp) * K_PERCENT))
            mink = lp.topk(k, largest=False).values.mean().item()
            all_mink.append(mink)

        # 4. Loss variance
        for b_idx in range(ids.size(0)):
            m   = mask[b_idx].bool()
            tl  = tok_loss[b_idx][m]
            v   = tl.var().item() if len(tl) > 1 else 0.0
            all_var.append(v)

        # 5. Zlib-normalised
        for b_idx, raw in enumerate(raw_texts):
            zlen = len(zlib.compress(raw))
            all_zlib.append(all_full_loss[-ids.size(0) + b_idx] / max(1, zlen) * 100)

    return {
        "full_loss":  np.array(all_full_loss),
        "resp_loss":  np.array(all_resp_loss),
        "min_k_prob": np.array(all_mink),
        "loss_var":   np.array(all_var),
        "zlib_ratio": np.array(all_zlib),
    }

# ── Attack scores ─────────────────────────────────────────────────────────────
def build_attack_scores(feat_m, feat_nm, feat_pt_m, feat_pt_nm):
    """Return dict: attack_name -> (y_true, scores)."""
    n_m  = len(feat_m["full_loss"])
    n_nm = len(feat_nm["full_loss"])
    y    = np.concatenate([np.ones(n_m), np.zeros(n_nm)])

    return {
        "Basic\nLOSS":      (y, np.concatenate([-feat_m["full_loss"],  -feat_nm["full_loss"]])),
        "PT-Ref\nLOSS":     (y, np.concatenate([feat_pt_m["full_loss"] - feat_m["full_loss"],
                                                 feat_pt_nm["full_loss"]- feat_nm["full_loss"]])),
        "Resp-Only\nLOSS":  (y, np.concatenate([-feat_m["resp_loss"],  -feat_nm["resp_loss"]])),
        "PT-Ref\nResp":     (y, np.concatenate([feat_pt_m["resp_loss"] - feat_m["resp_loss"],
                                                 feat_pt_nm["resp_loss"]- feat_nm["resp_loss"]])),
        "Min-K%\nProb":     (y, np.concatenate([feat_m["min_k_prob"],   feat_nm["min_k_prob"]])),
        "PT-Ref\nMin-K%":   (y, np.concatenate([feat_m["min_k_prob"] - feat_pt_m["min_k_prob"],
                                                  feat_nm["min_k_prob"]- feat_pt_nm["min_k_prob"]])),
        "Loss\nVariance":   (y, np.concatenate([-feat_m["loss_var"],   -feat_nm["loss_var"]])),
        "Zlib\nNorm":       (y, np.concatenate([-feat_m["zlib_ratio"], -feat_nm["zlib_ratio"]])),
    }

def compute_metrics(y_true, scores, label):
    auc = roc_auc_score(y_true, scores)
    ap  = average_precision_score(y_true, scores)
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    best_i = int(np.argmax(tpr - fpr))
    acc    = accuracy_score(y_true, (scores >= thresholds[best_i]).astype(int))
    mask10 = fpr <= 0.10
    tpr10  = float(tpr[mask10][-1]) if mask10.any() else 0.0
    return {"label":label,"mia_auc":round(float(auc),4),
            "avg_precision":round(float(ap),4),
            "accuracy":round(float(acc),4),
            "tpr_at_fpr10":round(tpr10,4),
            "verdict": ("Near-random (strong)" if auc<0.55
                        else "Moderate" if auc<0.65 else "Weak")}, fpr, tpr

# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_bar_comparison(dp_metrics, nd_metrics, out):
    atks = list(dp_metrics.keys())
    dp_aucs = [dp_metrics[k][0]["mia_auc"] for k in atks]
    nd_aucs = [nd_metrics[k][0]["mia_auc"] for k in atks]
    x, w = np.arange(len(atks)), 0.35

    fig, ax = plt.subplots(figsize=(13, 5))
    b1 = ax.bar(x-w/2, nd_aucs, w, label="QLoRA (non-private)",
                color=C["nondp"], edgecolor="white", zorder=3)
    b2 = ax.bar(x+w/2, dp_aucs, w, label="DP-QLoRA (ε≈2, outputs_2)",
                color=C["dp"],   edgecolor="white", zorder=3)
    for bars, vals in [(b1,nd_aucs),(b2,dp_aucs)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.axhline(0.5, color=C["random"], lw=1.5, ls="--", label="Random baseline (0.500)")
    ax.axhline(0.6, color="#FF7043", lw=0.8, ls=":", alpha=0.7, label="Concern threshold (0.600)")
    ax.set_xticks(x); ax.set_xticklabels(atks, fontsize=9)
    ax.set_ylabel("MIA-AUC", fontsize=11); ax.set_ylim(0.42, 0.75)
    ax.set_title(
        "Script 3: 8-Variant MIA — DP-QLoRA (ε≈2) vs Non-Private QLoRA\n"
        "All eight attacks from the Google DP guide | Lower AUC = stronger privacy",
        fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "script3_mia_bar_comparison.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  [SAVED] script3_mia_bar_comparison.pdf")

def plot_roc_grid(dp_metrics, nd_metrics, out):
    atks = list(dp_metrics.keys())
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("ROC Curves — All 8 MIA Variants\n"
                 "DP-QLoRA (ε≈2, orange) vs Non-Private (blue)",
                 fontsize=13, fontweight="bold")
    for ax, atk in zip(axes.flat, atks):
        ax.plot([0,1],[0,1], color=C["random"], lw=1, ls="--")
        _, fpr_nd, tpr_nd = nd_metrics[atk]
        _, fpr_dp, tpr_dp = dp_metrics[atk]
        ax.plot(fpr_nd, tpr_nd, lw=2, color=C["nondp"],
                label=f"Non-DP {nd_metrics[atk][0]['mia_auc']:.3f}")
        ax.plot(fpr_dp, tpr_dp, lw=2, color=C["dp"],
                label=f"DP {dp_metrics[atk][0]['mia_auc']:.3f}")
        ax.set_title(atk.replace("\n"," "), fontsize=9, fontweight="bold")
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.legend(fontsize=7, loc="lower right")
        ax.tick_params(labelsize=7)
    for ax in axes.flat: ax.set_xlabel("FPR",fontsize=8); ax.set_ylabel("TPR",fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "script3_mia_roc_grid.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  [SAVED] script3_mia_roc_grid.pdf")

def plot_loss_distributions(dp_m, dp_nm, nd_m, nd_nm, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, m_loss, nm_loss, tag, col in [
        (axes[0], nd_m, nd_nm, "QLoRA (non-private)",     C["nondp"]),
        (axes[1], dp_m, dp_nm, "DP-QLoRA (ε≈2, outputs_2)", C["dp"]),
    ]:
        lo = min(m_loss.min(), nm_loss.min())
        hi = max(m_loss.max(), nm_loss.max())
        bins = np.linspace(lo, hi, 50)
        ax.hist(m_loss,  bins=bins, alpha=0.6, color=C["mem"],   density=True,
                label=f"Members  (n={len(m_loss)})")
        ax.hist(nm_loss, bins=bins, alpha=0.6, color=C["nonmem"],density=True,
                label=f"Non-members  (n={len(nm_loss)})")
        ov = np.minimum(
            np.histogram(m_loss,  bins=bins, density=True)[0],
            np.histogram(nm_loss, bins=bins, density=True)[0],
        ).sum() * (bins[1]-bins[0])
        ax.set_title(f"{tag}\nDistribution overlap = {ov:.3f}  (higher = better privacy)",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Cross-Entropy Loss", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("Loss Distributions: Members vs Non-Members\n"
                 "More overlap → model can't distinguish training data",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out / "script3_loss_distributions.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  [SAVED] script3_loss_distributions.pdf")

def plot_summary_table(dp_metrics, nd_metrics, out):
    atks = list(dp_metrics.keys())
    dp_aucs = [dp_metrics[k][0]["mia_auc"] for k in atks]
    nd_aucs = [nd_metrics[k][0]["mia_auc"] for k in atks]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    col_labels = ["Attack", "DP AUC", "Non-DP AUC", "Gap", "Verdict"]
    rows = []
    for atk, da, na in zip(atks, dp_aucs, nd_aucs):
        verdict = "✅ Strong" if da < 0.55 else "⚠ Moderate" if da < 0.65 else "❌ Weak"
        rows.append([atk.replace("\n"," "), f"{da:.4f}", f"{na:.4f}",
                     f"{na-da:+.4f}", verdict])
    t = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.6)
    for j in range(len(col_labels)):
        t[0,j].set_facecolor("#1F3B73"); t[0,j].set_text_props(color="white", fontweight="bold")
    for i, row in enumerate(rows):
        color = "#E8F5E9" if "Strong" in row[-1] else "#FFF8E1" if "Moderate" in row[-1] else "#FFEBEE"
        for j in range(len(col_labels)):
            t[i+1, j].set_facecolor(color)
    ax.set_title("Script 3: MIA Results Summary Table\n"
                 "All attacks targeting outputs_2 (ε=1.9911)",
                 fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(out / "script3_mia_summary_table.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  [SAVED] script3_mia_summary_table.pdf")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    root = Path(args.project_root).resolve()
    out  = root / "outputs_privacy_audit"
    out.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    print(f"\nRunning 8-variant MIA | target: outputs_2 (ε≈2) | MAX_LENGTH={MAX_LENGTH}")

    members    = load_jsonl(root / "data/train.jsonl", args.max_samples, args.seed)
    nonmembers = load_jsonl(root / "data/val.jsonl",   args.max_samples, args.seed)
    n = min(len(members), len(nonmembers))
    members, nonmembers = members[:n], nonmembers[:n]
    print(f"Balanced: {n} members / {n} non-members")

    # Load DP model (outputs_2) + PT reference
    print("\n[1/3] Loading DP-QLoRA (outputs_2, ε≈2)...")
    model_dp, tok = load_model(root / "outputs_2/payee-lora-dp")
    pad_id = tok.pad_token_id or tok.eos_token_id

    ds_m  = PairDataset(members,    tok, MAX_LENGTH)
    ds_nm = PairDataset(nonmembers, tok, MAX_LENGTH)
    dl_m  = DataLoader(ds_m,  args.batch_size, shuffle=False, collate_fn=collate)
    dl_nm = DataLoader(ds_nm, args.batch_size, shuffle=False, collate_fn=collate)

    print("  Computing DP model features...")
    dp_feat_m  = compute_features(model_dp, dl_m,  pad_id, "DP members")
    dp_feat_nm = compute_features(model_dp, dl_nm, pad_id, "DP non-members")
    del model_dp; gc.collect(); torch.cuda.empty_cache()

    # Load non-DP model
    print("\n[2/3] Loading non-DP QLoRA (outputs_8)...")
    model_nd, _ = load_model(root / "outputs_8/outputs/payee-lora")
    nd_feat_m  = compute_features(model_nd, dl_m,  pad_id, "nonDP members")
    nd_feat_nm = compute_features(model_nd, dl_nm, pad_id, "nonDP non-members")
    del model_nd; gc.collect(); torch.cuda.empty_cache()

    # Load PT base as reference
    print("\n[3/3] Loading PT base model as reference...")
    base_model = load_base_model(tok)
    pt_feat_m  = compute_features(base_model, dl_m,  pad_id, "PT members")
    pt_feat_nm = compute_features(base_model, dl_nm, pad_id, "PT non-members")
    del base_model; gc.collect(); torch.cuda.empty_cache()

    # Build attack scores and metrics
    dp_attacks = build_attack_scores(dp_feat_m, dp_feat_nm, pt_feat_m, pt_feat_nm)
    nd_attacks = build_attack_scores(nd_feat_m, nd_feat_nm, pt_feat_m, pt_feat_nm)

    dp_metrics = {k: compute_metrics(y, s, f"DP {k}") for k,(y,s) in dp_attacks.items()}
    nd_metrics = {k: compute_metrics(y, s, f"nonDP {k}") for k,(y,s) in nd_attacks.items()}

    # Save
    with open(out / "s3_mia_dp_metrics.json", "w") as f:
        json.dump({k: v[0] for k,v in dp_metrics.items()}, f, indent=2)
    with open(out / "s3_mia_nondp_metrics.json", "w") as f:
        json.dump({k: v[0] for k,v in nd_metrics.items()}, f, indent=2)

    # Plots
    print("\nGenerating figures...")
    plot_bar_comparison(dp_metrics, nd_metrics, out)
    plot_roc_grid(dp_metrics, nd_metrics, out)
    plot_loss_distributions(
        dp_feat_m["full_loss"], dp_feat_nm["full_loss"],
        nd_feat_m["full_loss"], nd_feat_nm["full_loss"], out)
    plot_summary_table(dp_metrics, nd_metrics, out)

    # Console summary
    print(f"\n{'='*60}")
    print(f"  SCRIPT 3 — MIA RESULTS (outputs_2, ε=1.9911)")
    print(f"{'='*60}")
    print(f"  {'Attack':<20} {'DP AUC':>8} {'nonDP AUC':>10} {'Gap':>8}  Verdict")
    print(f"  {'-'*60}")
    for atk in dp_metrics:
        da = dp_metrics[atk][0]["mia_auc"]
        na = nd_metrics[atk][0]["mia_auc"]
        v  = dp_metrics[atk][0]["verdict"]
        print(f"  {atk.replace(chr(10),' '):<20} {da:>8.4f} {na:>10.4f} {na-da:>+8.4f}  {v}")
    best_atk = max(dp_metrics, key=lambda k: dp_metrics[k][0]["mia_auc"])
    print(f"\n  Strongest attack on DP model: {best_atk.replace(chr(10),' ')} "
          f"(AUC={dp_metrics[best_atk][0]['mia_auc']:.4f})")
    print(f"  Output: {out}/script3_*.pdf\n")

if __name__ == "__main__":
    main()