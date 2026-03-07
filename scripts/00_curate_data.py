"""
Phase 0 — Data curation (POC version, self-contained)

only csv stuff no gb for now

Pathogenic (label=1): patho/*.csv nucleotide_sequence columns
Benign    (label=0): nopatho/*.csv nucleotide_sequence columns

Everything comes from GeneBreaker/JailbreakDNABench — no external downloads.
"""
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT      = Path(__file__).resolve().parents[1]
BENCH     = ROOT / "GeneBreaker" / "JailbreakDNABench"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

TARGET_LEN  = 640
MIN_LEN     = 200
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ── helpers ────────────────────────────────────────────────────────────────

def normalise(seq: str, target: int = TARGET_LEN) -> str:
    seq = seq.upper().replace("\n", "").replace(" ", "")
    seq = ''.join(c for c in seq if c in "ACGTN")
    if len(seq) > target:
        seq = seq[:target]
    # left-pad to multiple of 6 (GENERator tokenizer requirement)
    pad = (6 - len(seq) % 6) % 6
    seq = "A" * pad + seq
    if len(seq) < target:
        seq = "A" * (target - len(seq)) + seq
    return seq

def is_valid(seq: str) -> bool:
    clean = seq.upper().replace("\n", "").replace(" ", "")
    clean = ''.join(c for c in clean if c in "ACGTN")
    if len(clean) < MIN_LEN:
        return False
    if clean.count("N") / max(len(clean), 1) > 0.05:
        return False
    return True

def read_csv_seqs(path: Path) -> list[str]:
    seqs = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = row.get("nucleotide_sequence", "").strip()
            if s and is_valid(s):
                seqs.append(s)
    return seqs

# ── collect sequences ──────────────────────────────────────────────────────

sequences, labels, families = [], [], []

for family_dir in sorted(BENCH.iterdir()):
    if not family_dir.is_dir():
        continue
    family = family_dir.name

    patho_dir   = family_dir / "patho"
    nopatho_dir = family_dir / "nopatho"

    for csv_file in sorted(patho_dir.glob("*.csv")) if patho_dir.exists() else []:
        for s in read_csv_seqs(csv_file):
            sequences.append(normalise(s))
            labels.append(1)
            families.append(family)

    for csv_file in sorted(nopatho_dir.glob("*.csv")) if nopatho_dir.exists() else []:
        for s in read_csv_seqs(csv_file):
            sequences.append(normalise(s))
            labels.append(0)
            families.append("benign")

# ── shuffle and save ───────────────────────────────────────────────────────

idx = list(range(len(sequences)))
random.shuffle(idx)
sequences = [sequences[i] for i in idx]
labels    = [labels[i]    for i in idx]
families  = [families[i]  for i in idx]

np.save(PROCESSED / "sequences.npy", np.array(sequences, dtype=object))
np.save(PROCESSED / "labels.npy",    np.array(labels,    dtype=np.int8))
np.save(PROCESSED / "families.npy",  np.array(families,  dtype=object))

# ── report ─────────────────────────────────────────────────────────────────

n_patho  = sum(labels)
n_benign = len(labels) - n_patho
print(f"Dataset saved to {PROCESSED}/")
print(f"  Total:      {len(sequences)}")
print(f"  Pathogenic: {n_patho}")
print(f"  Benign:     {n_benign}")
print(f"  Sequence length: all {TARGET_LEN} nt (normalised)")

from collections import Counter
print("\nPathogenic family counts:")
for fam, cnt in sorted(Counter(f for f, l in zip(families, labels) if l == 1).items()):
    print(f"  {fam:35s} {cnt}")
print("\nBenign sequences:", n_benign)

# Save pathogen family list for k-mer profile building
patho_families = sorted(set(f for f, l in zip(families, labels) if l == 1))
with open(PROCESSED / "pathogen_families.json", "w") as f:
    json.dump(patho_families, f, indent=2)
