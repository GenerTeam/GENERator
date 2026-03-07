"""
Phase 1 — K-mer frequency baseline (SG1.1–1.4)

Builds per-family reference k-mer profiles, scores all sequences via cosine
similarity, computes ROC/AUC, and saves profiles + results.

Output:
  data/processed/kmer_profiles/profiles.npy   (n_families, 1024) float32
  data/processed/kmer_profiles/names.json      list of family names
  data/processed/kmer_results.npz             scores, labels, families
  plots/roc_kmer.png
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

ROOT      = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
PLOTS     = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)
(PROCESSED / "kmer_profiles").mkdir(exist_ok=True)

K = 5
ALPHABET = [a+b+c+d+e
            for a in "ACGT" for b in "ACGT" for c in "ACGT"
            for d in "ACGT" for e in "ACGT"]  # 1024 canonical 5-mers
KMER_INDEX = {km: i for i, km in enumerate(ALPHABET)}

# ── k-mer helpers ───────────────────────────────────────────────────────────

def kmer_vec(seq: str) -> np.ndarray:
    counts = Counter(seq[i:i+K] for i in range(len(seq) - K + 1)
                     if all(c in "ACGT" for c in seq[i:i+K]))
    vec = np.array([counts.get(km, 0) for km in ALPHABET], dtype=np.float32)
    total = vec.sum()
    return vec / (total + 1e-10)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

# ── load data ───────────────────────────────────────────────────────────────

sequences = np.load(PROCESSED / "sequences.npy", allow_pickle=True).tolist()
labels    = np.load(PROCESSED / "labels.npy").tolist()
families  = np.load(PROCESSED / "families.npy", allow_pickle=True).tolist()

labels_arr = np.array(labels)

# ── build per-family reference profiles (SG1.2) ─────────────────────────────

# Group pathogenic sequences by family
family_seqs: dict[str, list[str]] = {}
for seq, label, fam in zip(sequences, labels, families):
    if label == 1:
        family_seqs.setdefault(fam, []).append(seq)

pathogen_names = sorted(family_seqs.keys())
print(f"Pathogen families: {pathogen_names}")
for name, seqs in family_seqs.items():
    print(f"  {name}: {len(seqs)} sequences")

# Per-family profile = mean k-mer vector over all its sequences
pathogen_profiles: dict[str, np.ndarray] = {
    name: np.mean([kmer_vec(s) for s in seqs], axis=0).astype(np.float32)
    for name, seqs in family_seqs.items()
}

# Aggregate profile: mean over all pathogenic sequences
all_patho_vecs = [kmer_vec(s) for s, l in zip(sequences, labels) if l == 1]
aggregate_profile = np.mean(all_patho_vecs, axis=0).astype(np.float32)

# Save profiles
profiles_matrix = np.stack([pathogen_profiles[n] for n in pathogen_names])
np.save(PROCESSED / "kmer_profiles" / "profiles.npy", profiles_matrix)
with open(PROCESSED / "kmer_profiles" / "names.json", "w") as f:
    json.dump(pathogen_names, f, indent=2)
print(f"\nProfiles saved: {profiles_matrix.shape} for {pathogen_names}")

# ── screening function (SG1.3) ───────────────────────────────────────────────

def kmer_screen(seq: str, threshold: float = 0.80) -> dict:
    qvec = kmer_vec(seq)
    sims = {name: cosine(qvec, pvec) for name, pvec in pathogen_profiles.items()}
    top_name = max(sims, key=sims.get)
    top_score = sims[top_name]
    agg_score = cosine(qvec, aggregate_profile)
    return {
        "flagged": top_score > threshold,
        "top_hit": top_name,
        "score": top_score,
        "agg_score": agg_score,
        "all_sims": sims,
    }

# ── score all sequences ──────────────────────────────────────────────────────

print("\nScoring all sequences...")
agg_scores  = np.array([cosine(kmer_vec(s), aggregate_profile) for s in sequences])
top_scores  = np.array([max(cosine(kmer_vec(s), pathogen_profiles[n]) for n in pathogen_names)
                        for s in sequences])

# ── ROC / AUC (SG1.4) ───────────────────────────────────────────────────────

auc_agg = roc_auc_score(labels_arr, agg_scores)
auc_top = roc_auc_score(labels_arr, top_scores)
print(f"\nK-mer AUC (aggregate profile): {auc_agg:.3f}")
print(f"K-mer AUC (max per-family):    {auc_top:.3f}")

# Threshold at ≤5% FPR (small dataset — 2% FPR gives too few positives)
for score_arr, label, auc in [
    (agg_scores, "aggregate", auc_agg),
    (top_scores, "max-family", auc_top),
]:
    fpr, tpr, thresholds = roc_curve(labels_arr, score_arr)
    target_fpr = 0.05
    idx = np.searchsorted(fpr, target_fpr)
    if idx >= len(thresholds):
        idx = len(thresholds) - 1
    print(f"  [{label}] threshold={thresholds[idx]:.3f} "
          f"@ FPR={fpr[idx]:.1%}  TPR={tpr[idx]:.1%}")

# ── per-family stratified AUC ────────────────────────────────────────────────

print("\nPer-family AUC (pathogenic family vs all benign):")
benign_mask = labels_arr == 0
benign_agg  = agg_scores[benign_mask]
benign_top  = top_scores[benign_mask]

for fam in pathogen_names:
    fam_mask = np.array([f == fam for f in families])
    fam_agg  = agg_scores[fam_mask]
    fam_top  = top_scores[fam_mask]
    combined_labels = np.array([1] * fam_mask.sum() + [0] * benign_mask.sum())
    combined_agg    = np.concatenate([fam_agg, benign_agg])
    combined_top    = np.concatenate([fam_top, benign_top])
    auc_a = roc_auc_score(combined_labels, combined_agg)
    auc_t = roc_auc_score(combined_labels, combined_top)
    print(f"  {fam:20s}  agg={auc_a:.3f}  max-family={auc_t:.3f}  (n={fam_mask.sum()})")

# ── ROC plot ─────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 6))
for score_arr, name, color in [
    (agg_scores, f"K-mer aggregate (AUC={auc_agg:.3f})", "steelblue"),
    (top_scores, f"K-mer max-family (AUC={auc_top:.3f})", "darkorange"),
]:
    fpr, tpr, _ = roc_curve(labels_arr, score_arr)
    ax.plot(fpr, tpr, color=color, lw=2, label=name)

ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("K-mer Baseline ROC (k=5)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS / "roc_kmer.png", dpi=150)
print(f"\nPlot saved to {PLOTS}/roc_kmer.png")

# ── save results for phase 3 comparison ─────────────────────────────────────

np.savez(
    PROCESSED / "kmer_results.npz",
    agg_scores=agg_scores,
    top_scores=top_scores,
    labels=labels_arr,
    families=np.array(families, dtype=object),
)
print("Results saved to data/processed/kmer_results.npz")
