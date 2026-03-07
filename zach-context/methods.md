# Anti-Jailbreaking Defenses for DNA Foundation Models
*Feasible approaches for a 10-hour hackathon — component-level novelty is established; the application to DNA FM biosecurity is new.*

---

## Context

GeneBreaker (Zhang et al., arXiv:2505.23839) demonstrated the first systematic jailbreak of DNA foundation models, achieving up to **60% attack success rate** on Evo2-40B — producing sequences with >90% nucleotide identity to SARS-CoV-2, HIV-1, and other select agents. The attack works by:

1. Using GPT-4o to retrieve non-pathogenic sequences homologous to the target pathogen
2. Constructing a few-shot prompt with a phylogenetic tag + homologous sequences
3. Running beam search guided by `f = PathoLM(x) + α · log p(x)` to steer generation toward pathogenic outputs

**The defense landscape is nearly empty.** Fewer than 3% of ~370 biological AI models include any safety safeguards, and Evo2 remains pip-installable with zero guardrails.

**Key framing:** every method below has prior use in bioinformatics. None has been applied as a real-time post-generation guardrail on a DNA FM. That gap is the contribution.

---

## Target Model: GENERator

GENERator is a LLaMA-based autoregressive DNA language model trained on RefSeq. It is the probe backbone used in our defense (see Method 4). The attack target is still Evo2 — GENERator is used for *detection*, not generation.

```
Input:  nucleotide string  →  "ATGTTTGTTTTTCTTGTT..."  (must be left-padded to multiple of 6)
Output: nucleotide string  →  continuation of the prompt (autoregressive, 6-mer tokens)
```

At the API level (standard HuggingFace — no custom engine):

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(
    "GenerTeam/GENERator-v2-eukaryote-1.2b-base", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "GenerTeam/GENERator-v2-eukaryote-1.2b-base",
    trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='cuda')

# With embeddings (needed for activation probes) — standard HuggingFace
outputs = model(**inputs, output_hidden_states=True, return_dict=True)
# outputs.hidden_states → tuple of (n_layers+1) tensors, each (1, seq_len, hidden_dim)
# Index: 0=embedding layer, -1=last transformer block
# Probe at n//4, n//2, 3n//4, -1 where n=model.config.num_hidden_layers (check at runtime)
```

Key properties relevant to defense:

- **6-mer tokenizer** (same as PathoLM) — vocabulary of 4096 tokens (4^6, not 4,104)
- **LLaMA decoder architecture** — actual layer count from `model.config.num_hidden_layers` at runtime (not hardcoded as 28)
- **eukaryote variant** trained with GCP: embeddings shaped around gene-centric CDS regions — matches JailbreakDNABench target sequences
- **~5 GB VRAM** in bf16 — fits alongside other components on any A100

---

## Method Independence and Shared I/O

All four defense methods take the **same input** and produce the **same output shape**. They can be run in any order or in parallel — there is no dependency between Methods 1–3. Method 4 (activation probes) is the one exception: it requires GENERator to be loaded, since it reads from GENERator's internals.

```
                    ┌─────────────────────────────────────────────────┐
                    │              SHARED INPUT                        │
                    │   generated DNA sequence (plain string,          │
                    │   e.g. "ATGTTTGTTTTTCTTGTTTTATTGCC...")          │
                    └───────────┬─────────────────────────────────────┘
                                │
           ┌────────────────────┼────────────────────┬──────────────────────┐
           ▼                    ▼                    ▼                      ▼
    [BLAST screen]       [K-mer screen]       [PathoLM screen]     [Activation probe]
    independent          independent          independent          requires Evo2 loaded
    ~5s latency          <1ms latency         <10ms latency        ~200ms latency
           │                    │                    │                      │
           ▼                    ▼                    ▼                      ▼
    {flagged: bool,      {flagged: bool,      {flagged: bool,      {flagged: bool,
     identity: float,    top_hit: str,        score: float}        score: float}
     organism: str}      score: float}
                                │
                    ┌───────────┴───────────────────────────────────────────┐
                    │             COMBINED OUTPUT (OR logic)                 │
                    │   {flagged: bool, layer_results: dict, latency: dict}  │
                    └───────────────────────────────────────────────────────┘
```

**BLAST, k-mer, and PathoLM are fully independent** — they don't call each other, don't share weights, and don't require GENERator to be running. You can test any one of them on any DNA string without any other component present.

**Activation probes are GENERator-dependent** — the probe classifier runs on embeddings extracted from GENERator's forward pass. GENERator uses the standard HuggingFace API (`output_hidden_states=True`), so no custom engine is required.

| Method | Depends on GENERator? | Depends on other methods? | Can run standalone? |
|---|---|---|---|
| BLAST | ❌ No | ❌ No | ✅ Yes |
| K-mer similarity | ❌ No | ❌ No | ✅ Yes |
| PathoLM | ❌ No | ❌ No | ✅ Yes |
| Activation probe | ✅ Yes (needs forward pass) | ❌ No | ✅ Yes (if GENERator loaded) |

---

## Method 1: BLAST Output Screening

### Input / Output
```
Input:  DNA string  e.g. "ATGTTTGTTTTTCTTGTT..."  (any length, ≥50 nt)
        + local BLAST database of pathogen reference genomes

Output: { flagged: bool,          # True if any hit exceeds identity threshold
           top_identity: float,    # % nucleotide identity to best hit (0–100)
           top_organism: str,      # e.g. "SARS-CoV-2 spike protein"
           hits: list[str] }       # raw BLAST tabular output lines
```

Requires nothing from Evo2. Can be called on any DNA string from any source.

### What it is
BLAST (Basic Local Alignment Search Tool) finds regions of local similarity between sequences. Given a generated DNA sequence, BLASTN queries it against a curated database of known pathogen genomes and reports the best-hit identity percentage.

### Why it works here
GeneBreaker's own success criterion is **>90% BLAST identity** to a target in JailbreakDNABench. A defensive BLAST screen using the same threshold directly counters the attack metric — any sequence that "succeeds" as a jailbreak is detectable by definition.

### Prior use
BLAST is used by IBBIS `commec` to screen synthetic DNA orders at companies like Twist Bioscience and IDT. It is also GeneBreaker's evaluation pipeline. **Not previously deployed as a real-time post-generation guardrail on a DNA FM.**

### Implementation sketch
```python
from Bio.Blast.Applications import NcbiblastnCommandline
import os

def blast_screen(sequence: str, db_path: str, threshold: float = 90.0) -> dict:
    # Write query to temp FASTA
    with open("query.fa", "w") as f:
        f.write(f">query\n{sequence}\n")
    
    # Run BLASTN against local pathogen DB
    cmd = NcbiblastnCommandline(
        query="query.fa", db=db_path,
        perc_identity=threshold, outfmt="6 qseqid sseqid pident stitle",
        out="blast_out.txt"
    )
    cmd()
    
    # Parse hits
    with open("blast_out.txt") as f:
        hits = f.readlines()
    
    return {"flagged": len(hits) > 0, "hits": hits}
```

Build local DB from JailbreakDNABench:
```bash
makeblastdb -in pathogens.fa -dbtype nucl -out pathogens_db
```

### Links
- BLAST+ standalone: https://www.ncbi.nlm.nih.gov/books/NBK153387/
- BioPython BLAST wrapper: https://biopython.org/docs/latest/api/Bio.Blast.Applications.html
- IBBIS commec (mature screening tool): https://github.com/ibbis-screening/common-mechanism
- JailbreakDNABench (pathogen reference sequences): https://github.com/zaixizhang/GeneBreaker

### Caveats
- ~5s per query against a small local DB (not suitable for high-throughput)
- Misses sequences below the identity threshold — adaptive attackers could target 85% identity and retain function
- The HMM-based layer in `commec` is more robust to evasion than BLAST alone

---

## Method 2: K-mer Frequency Cosine Similarity

### Input / Output
```
Input:  DNA string  e.g. "ATGTTTGTTTTTCTTGTT..."  (any length, ≥50 nt)
        + pre-built k-mer reference profiles (dict: pathogen_name → np.ndarray)

Output: { flagged: bool,          # True if cosine similarity exceeds threshold
           top_hit: str,          # e.g. "HIV-1"
           score: float }         # cosine similarity in [0, 1]
```

Requires nothing from Evo2. Reference profiles are built offline from any FASTA files.

### What it is
Decompose a DNA sequence into all substrings of length k (k-mers), build a frequency vector, and compute cosine similarity against pre-built reference profiles from known pathogen genomes. High similarity to a pathogen profile → flag.

### Why it works here
K-mer frequency profiles are organism-specific "fingerprints." Sequences with >90% identity to a pathogen will share the vast majority of their 5-mers with that pathogen, producing a high cosine similarity even if BLAST is not available. Runs in <1ms in pure Python — ideal as a fast pre-filter before slower methods.

### Prior use
Used in metagenomics classifiers (PaPrBaG, DCiPatho, Kraken2) for taxonomic classification of sequencing reads. **Not applied as a DNA FM defense.**

### Implementation sketch
```python
from collections import Counter
import numpy as np
from typing import Dict

def kmer_profile(seq: str, k: int = 5) -> np.ndarray:
    """Build normalised k-mer frequency vector (4^k = 1024 features for k=5)"""
    counts = Counter(seq[i:i+k] for i in range(len(seq) - k + 1))
    # Build fixed-length vector over all possible k-mers
    vocab = sorted(set(counts.keys()))  # or precompute all 4^k possibilities
    vec = np.array([counts.get(km, 0) for km in vocab], dtype=float)
    return vec / (vec.sum() + 1e-10)  # normalise

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

# Pre-build pathogen profiles from reference genomes
# At inference: compare generated sequence against all profiles
def screen_kmer(generated_seq: str, pathogen_profiles: Dict[str, np.ndarray],
                threshold: float = 0.85) -> dict:
    query_profile = kmer_profile(generated_seq)
    results = {name: cosine_similarity(query_profile, prof) 
               for name, prof in pathogen_profiles.items()}
    max_sim = max(results.values())
    return {"flagged": max_sim > threshold, "similarities": results}
```

### Links
- PaPrBaG (pathogen detection via k-mers): https://github.com/covid19-bh-bioinformatics/PaPrBaG
- Jellyfish (fast k-mer counting, C++ with Python bindings): https://github.com/gmarcais/Jellyfish
- Kraken2 (k-mer taxonomic classifier): https://github.com/DerrickWood/kraken2

### Caveats
- Cosine similarity threshold needs calibration against false positive rate on benign sequences
- k=5 (1,024 features) is a good balance; k=6 (4,096) increases discriminative power at minor compute cost
- Does not generalise to sequences below ~70% identity with any known pathogen

---

## Method 3: PathoLM Classification

### Input / Output
```
Input:  DNA string  (≤12,000 nt; truncated beyond that)

Output: { score: float,           # pathogenicity probability in [0, 1]
           flagged: bool }         # True if score > threshold (default 0.5)
```

Requires nothing from Evo2. PathoLM is a self-contained encoder model — load once, call on any DNA string.

### What it is
PathoLM is a fine-tuned **Nucleotide Transformer v2 50M** — an encoder-only BERT-style model trained to perform binary classification of DNA sequences as "pathogen" or "non-pathogen." It uses 6-mer tokenisation and produces a probability score.

### Why it works here
GeneBreaker uses PathoLM *offensively* — as the guidance signal in beam search, explicitly optimising generated sequences to have high PathoLM scores. A defensive screen using the same model at inference time will therefore catch GeneBreaker outputs targeting pathogens within PathoLM's training distribution (SARS-CoV-2, HIV-1). The attack literally optimises for detectability.

### Critical limitation — and why the GENERator probe is more general

PathoLM was fine-tuned on **~30 species of viruses and bacteria** (GeneBreaker §3.2). Its exact training species list is not published, so coverage of specific JailbreakDNABench families (Ebola, Variola, Poliovirus, Rabies, Measles) is unknown. Its knowledge is **hard-bounded by its labelled training set** — ask it about a viral family it never saw and it has no basis for judgement.

The GENERator activation probe is fundamentally different in kind:

- **PathoLM** is a *supervised* classifier. Pathogenicity knowledge was injected explicitly through labelled examples of ~30 species. Generalisation beyond those species is not guaranteed.
- **GENERator probe** is *unsupervised at the representation level*. GENERator was never shown a pathogenicity label. It learned the statistical grammar of functional DNA — codon usage, reading frame structure, evolutionary conservation patterns — from 386B nucleotides of raw sequence via next-token prediction. If the probe works, it means this structure is linearly separable across viral families the model never explicitly encountered, because the underlying biological signals (codon usage bias, high-entropy variable regions, protein-coding structure) are **family-agnostic**.

The expected result: PathoLM scores high on SARS-CoV-2 and HIV-1 (in its training distribution), but degrades on Ebola, Variola, Poliovirus. The GENERator probe, if the hypothesis holds, maintains performance across all 6 JailbreakDNABench categories — because it is detecting a structural property of pathogenic sequence, not pattern-matching against memorised families. **This cross-family generalisation is the core research claim.**

### Specs
| Property | Value |
|---|---|
| Parameters | ~50M |
| VRAM | <0.5 GB |
| Inference (640 nt) | <10 ms on GPU |
| Input | Raw nucleotide sequence (max ~12,000 nt) |
| Output | Probability score in [0, 1] |

### Implementation sketch
```python
# Clone repo and install: git clone https://github.com/Sajib-006/Patho-LM
# Download weights from Zenodo (see repo README)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class PathoLMScreener:
    def __init__(self, checkpoint_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
        self.model.eval().cuda()
    
    def score(self, sequence: str, threshold: float = 0.5) -> dict:
        inputs = self.tokenizer(sequence, return_tensors="pt", 
                                truncation=True, max_length=2048).to("cuda")
        with torch.no_grad():
            logits = self.model(**inputs).logits
        prob = torch.softmax(logits, dim=-1)[0, 1].item()  # pathogen probability
        return {"score": prob, "flagged": prob > threshold}
```

### Links
- PathoLM repo: https://github.com/Sajib-006/Patho-LM
- PathoLM paper (arXiv): https://arxiv.org/abs/2406.13133
- Nucleotide Transformer (base model): https://github.com/instadeepai/nucleotide-transformer
- GeneBreaker (offensive use of PathoLM): https://github.com/zaixizhang/GeneBreaker

### Caveats
- No published false positive rate on benign sequences (model organisms, synthetic biology constructs)
- Adaptive attacker can add a ceiling constraint to score just below the detection threshold
- GeneBreaker Google Drive checkpoint may differ from Zenodo release; use Zenodo as primary

---

## Method 4: GENERator Activation Space Linear Probes

### Input / Output
```
Input:  DNA string  (passed through GENERator's forward pass)
        + trained probe classifier (LogisticRegression fitted on embeddings)

Intermediate: GENERator forward pass  →  hidden_states tuple (n_layers+1, seq_len, hidden_dim)
              mean-pool over positions  →  vector (hidden_dim,)  ~2048-dim for 1.2B model

Output: { score: float,           # pathogenicity probability in [0, 1]
           flagged: bool,          # True if score > threshold
           layer: int }            # which GENERator layer index was used
```

**Requires GENERator to be loaded locally (~5 GB VRAM in bf16).** The probe classifier itself (LogisticRegression) is tiny — the cost is entirely the GENERator forward pass (~150ms on A100).

### What it is
Extract intermediate embeddings from GENERator's forward pass on a generated sequence, then apply a linear classifier (logistic regression / SVM) trained to distinguish pathogenic from benign sequences in that activation space. The probe is trained offline on ~500 pathogenic + ~500 benign sequences. GENERator uses the standard HuggingFace API — no custom engine.

### Why it works here
The hypothesis — drawn from analogous work on protein language models (ESM-2/ESM-3) — is that pathogenicity is **linearly encoded** in GENERator's intermediate representations. GENERator's training data (RefSeq) includes pathogenic bacterial and viral sequences, unlike Evo2 which excluded eukaryotic viruses. This makes GENERator's activation space a better candidate for pathogenicity probing. If the hypothesis holds, a linear probe trained on 1,000 sequences generalises to novel jailbreak outputs.

GENERator uses the standard HuggingFace `output_hidden_states` API:
```python
outputs = model(**inputs, output_hidden_states=True, return_dict=True)
# outputs.hidden_states: tuple of (n_layers+1,) tensors, each (1, seq_len, hidden_dim)
# Probe layers to try: 7 (early), 14 (mid), 21 (late), -1 (last)
```

### Why this is the novel contribution
Activation-space biosecurity probing has been done on **protein** language models (ESM models, NeurIPS 2025 BioSafe GenAI Workshop). It has **never been applied to a DNA language model.** Whether pathogenicity is linearly separable in GENERator's activation space is an open empirical question — answering it is the research finding.

### Implementation sketch
```python
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# Load once
tokenizer = AutoTokenizer.from_pretrained(
    "GenerTeam/GENERator-v2-eukaryote-1.2b-base", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "GenerTeam/GENERator-v2-eukaryote-1.2b-base",
    trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='cuda').eval()

def get_embedding(seq: str, layer_idx: int = 14) -> np.ndarray:
    # Left-pad to multiple of 6 (GENERator tokenizer requirement)
    pad = (6 - len(seq) % 6) % 6
    seq = 'A' * pad + seq
    inputs = tokenizer(seq, return_tensors='pt', add_special_tokens=True).to('cuda')
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    hidden = outputs.hidden_states[layer_idx]  # (1, seq_len, hidden_dim)
    # Mean-pool over sequence length
    mask = inputs['attention_mask'].unsqueeze(-1).float()
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
    return pooled.squeeze(0).cpu().float().numpy()

# Train probe (try layer indices 7, 14, 21, -1; pick best CV AUC)
X = np.stack([get_embedding(seq) for seq in sequences])
y = np.array(labels)  # 1=pathogenic, 0=benign

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
probe = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
cv_auc = cross_val_score(probe, X_scaled, y, cv=5, scoring='roc_auc')
print(f"Probe AUC: {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")
probe.fit(X_scaled, y)

def probe_screen(seq: str, threshold: float = 0.5) -> dict:
    emb = get_embedding(seq)
    prob = probe.predict_proba(scaler.transform(emb.reshape(1, -1)))[0, 1]
    return {"score": prob, "flagged": prob > threshold}
```

### Visualisation (for demo)
```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE(n_components=2, random_state=42, perplexity=40)
X_2d = tsne.fit_transform(X_scaled)

plt.scatter(X_2d[y==0, 0], X_2d[y==0, 1], c='steelblue', label='Benign', alpha=0.6)
plt.scatter(X_2d[y==1, 0], X_2d[y==1, 1], c='crimson', label='Pathogenic', alpha=0.6)
# Plot jailbreak output as a star
plt.scatter(*tsne_point, marker='*', s=300, c='black', label='GeneBreaker output')
plt.legend(); plt.title("GENERator Activation Space — Pathogenicity Probe")
```

### Links
- GENERator repo: https://github.com/GenerTeam/GENERator
- GENERator models (HuggingFace): https://huggingface.co/GenerTeam/
- "Steering Protein Language Models" (ESM-2 probe precedent, arXiv:2509.07983): https://arxiv.org/abs/2509.07983
- VFDB (virulence factor sequences for training data): http://www.mgc.ac.cn/VFs/main.htm
- NeurIPS 2025 BioSafe GenAI Workshop (activation-space biosecurity auditing on ESM): https://neurips.cc/virtual/2025/workshop/84714

### Caveats
- If linear probes achieve <65% AUC, pathogenicity may not be linearly separable — fall back to `SVC(kernel='rbf', C=10, probability=True)` or a small MLP
- Try both `GENERator-v2-prokaryote-1.2b-base` and `GENERator-v2-eukaryote-1.2b-base` — different training distributions may produce different probe AUCs
- The 3B model will likely give higher AUC at the cost of ~12 GB VRAM vs ~5 GB

---

## Combining the Methods: Three + One Layer Pipeline

```
Generated sequence
        │
        ▼
[Layer 1] K-mer cosine similarity      ← <1ms,  covers broad similarity
        │ flag if cosine > 0.85
        ▼
[Layer 2] PathoLM classification       ← <10ms, covers SARS-CoV-2, HIV-1
        │ flag if score > 0.5
        ▼
[Layer 3] BLAST against pathogen DB    ← ~5s,   definitive identity check
        │ flag if identity > 90%
        ▼
[Layer 4] GENERator activation probe   ← ~150ms, novel early-warning signal
        │ flag if probe score > 0.5
        ▼
  FLAG / PASS decision
```

**Use OR logic** — flag if *any* layer triggers. Tune thresholds independently per layer using ROC analysis on JailbreakDNABench positives vs. random genomic negatives.

**VRAM budget on a single A100 80GB:**

| Configuration | VRAM | Fits? |
|---|---|---|
| GENERator-v2-1.2b + PathoLM | ~5.5 GB | ✅ |
| GENERator-v2-3b + PathoLM | ~12.5 GB | ✅ |
| + local BLAST DB in memory | ~20–30 GB | ✅ |

---

---

## Proposed Solution: 6–8 Hour Build Plan

### The core insight

**Data curation is the shared bottleneck for every method.** You need ~500 pathogenic + ~500 benign sequences regardless of which defense you build. Do that first, once, and every subsequent method becomes fast. This is the strategic unlock.

The simplest highest-leverage starting point is therefore:

> **Activation probes on GENERator-v2-prokaryote-1.2b, with BLAST + k-mer + PathoLM as the scaffolding.**

Why probes first, not screener first? Because:
- Probes are the novel scientific claim — if they work, you have a publishable finding
- The data you curate for probes directly powers the k-mer profiles and BLAST DB
- Probe training is ~1 hour once embeddings are extracted — the bulk of time is data + model download (~5 GB, fast)
- If probes fail, you fall back to the screener, which is already half-built

### What "simplest" means in practice

Don't try to replicate the full GeneBreaker attack. Use the JailbreakDNABench sequences directly as your "known bad" set — they're already curated, already validated as not in Evo2's training data, and they're the exact sequences GeneBreaker targets. You don't need to run the attack to build the defense.

```
JailbreakDNABench sequences  →  pathogenic training set  (already exists, just download)
NCBI housekeeping genes       →  benign training set      (E. coli, S. cerevisiae, human)
```

That's your entire dataset. ~1 hour of curation. Everything flows from there.

---

### Hour-by-hour (solo or 2-person)

**Hour 0–1: Setup + data**

```bash
# Environment
conda create -n biosec python=3.10
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.49.0 huggingface_hub datasets
pip install biopython scikit-learn matplotlib

# Clone GeneBreaker — get JailbreakDNABench for free
git clone https://github.com/zaixizhang/GeneBreaker
# Pathogenic sequences are in GeneBreaker/JailbreakDNABench/

# PathoLM
git clone https://github.com/Sajib-006/Patho-LM
# Download Zenodo checkpoint (see repo README)

# GENERator is already cloned locally; models download from HuggingFace on first load (~5 GB)
```

While GENERator downloads (~5 GB, takes 5–10 min), curate benign sequences. Grab 500 CDS sequences from E. coli K-12 (NCBI accession U00096) and random intergenic regions from S. cerevisiae. Total: ~1,000 sequences, labels 0/1.

**Hour 1–3: Activation probe (the novel part)**

```python
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Load model once (~5 GB bf16)
MODEL_NAME = "GenerTeam/GENERator-v2-eukaryote-1.2b-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, trust_remote_code=True,
    torch_dtype=torch.bfloat16, device_map='cuda').eval()

def get_embedding(seq: str, layer_idx: int = 14) -> np.ndarray:
    pad = (6 - len(seq) % 6) % 6
    seq = 'A' * pad + seq
    inputs = tokenizer(seq, return_tensors='pt', add_special_tokens=True).to('cuda')
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    hidden = outputs.hidden_states[layer_idx]  # (1, seq_len, hidden_dim)
    mask = inputs['attention_mask'].unsqueeze(-1).float()
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
    return pooled.squeeze(0).cpu().float().numpy()

# Extract for all 1000 sequences (~150ms each → ~2.5 min total)
X = np.stack([get_embedding(seq) for seq in sequences])
y = np.array(labels)  # 1=pathogenic, 0=benign

# Train linear probe
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

probe = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
cv_auc = cross_val_score(probe, X_scaled, y, cv=5, scoring='roc_auc')
print(f"Probe AUC: {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")
probe.fit(X_scaled, y)  # fit on full set for deployment
```

Try four layer indices: `7` (early), `14` (mid), `21` (late), `-1` (last). Pick the one with best CV AUC — protein model literature suggests mid-to-late layers work best.

**Hour 3–5: Lightweight screener (build on top of probe data)**

Now that you have the pathogen sequences downloaded, building the screener is fast:

```python
from collections import Counter
from Bio.Blast.Applications import NcbiblastnCommandline
import subprocess

# --- K-mer profiles (reuse pathogenic sequences from JailbreakDNABench) ---
def kmer_vec(seq: str, k: int = 5) -> np.ndarray:
    all_kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
    counts = Counter(all_kmers)
    # Fixed 4^k alphabet
    alphabet = [a+b+c+d+e for a in 'ACGT' for b in 'ACGT'
                for c in 'ACGT' for d in 'ACGT' for e in 'ACGT']
    vec = np.array([counts.get(km, 0) for km in alphabet], dtype=float)
    return vec / (vec.sum() + 1e-10)

# Build one reference profile per pathogen by mean-pooling its sequences
pathogen_profiles = {}
for pathogen, seqs in pathogen_seqs.items():
    pathogen_profiles[pathogen] = np.mean([kmer_vec(s) for s in seqs], axis=0)

def kmer_screen(seq: str, threshold: float = 0.80) -> dict:
    qvec = kmer_vec(seq)
    sims = {p: float(np.dot(qvec, pvec) /
                     (np.linalg.norm(qvec) * np.linalg.norm(pvec) + 1e-10))
            for p, pvec in pathogen_profiles.items()}
    top = max(sims, key=sims.get)
    return {"flagged": sims[top] > threshold, "top_hit": top, "score": sims[top]}

# --- BLAST DB (build once from JailbreakDNABench FASTA) ---
# makeblastdb -in JailbreakDNABench/pathogens.fa -dbtype nucl -out pathogens_db
def blast_screen(seq: str, db: str = 'pathogens_db', threshold: float = 90.0) -> dict:
    with open('/tmp/query.fa', 'w') as f:
        f.write(f'>query\n{seq}\n')
    cmd = NcbiblastnCommandline(query='/tmp/query.fa', db=db,
                                perc_identity=threshold,
                                outfmt='6 qseqid sseqid pident stitle',
                                out='/tmp/blast_out.txt')
    cmd()
    with open('/tmp/blast_out.txt') as f:
        hits = f.readlines()
    return {"flagged": len(hits) > 0, "hits": hits[:3]}

# --- PathoLM (already cloned above) ---
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class PathoLMScreener:
    def __init__(self, ckpt: str):
        self.tok = AutoTokenizer.from_pretrained(ckpt)
        self.model = AutoModelForSequenceClassification.from_pretrained(ckpt).eval().cuda()

    def score(self, seq: str, threshold: float = 0.5) -> dict:
        inputs = self.tok(seq, return_tensors='pt',
                          truncation=True, max_length=2048).to('cuda')
        with torch.no_grad():
            prob = torch.softmax(self.model(**inputs).logits, dim=-1)[0, 1].item()
        return {"score": prob, "flagged": prob > threshold}
```

**Hour 5–7: Unified pipeline + evaluation**

```python
def screen(seq: str) -> dict:
    """Full 4-layer defense pipeline. OR logic — flag if any layer triggers."""
    results = {
        'kmer':    kmer_screen(seq),
        'patholm': patholm.score(seq),
        'blast':   blast_screen(seq),
        'probe':   {
            'score': float(probe.predict_proba(
                         scaler.transform(get_embedding(seq).reshape(1, -1)))[0, 1]),
            'flagged': None  # set below
        }
    }
    results['probe']['flagged'] = results['probe']['score'] > 0.5
    results['flagged'] = any(v['flagged'] for v in results.values() if isinstance(v, dict))
    return results

# Evaluate: run on JailbreakDNABench positives + benign set
# Compute per-layer precision/recall/F1 + combined
from sklearn.metrics import classification_report, roc_curve
```

**Hour 7–8: t-SNE visualisation + demo**

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_2d = tsne.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(X_2d[y==0, 0], X_2d[y==0, 1], c='steelblue', alpha=0.5,
           label='Benign', s=20)
ax.scatter(X_2d[y==1, 0], X_2d[y==1, 1], c='crimson', alpha=0.5,
           label='Pathogenic', s=20)
# Add generated sequence as a star
ax.scatter(*generated_2d, marker='*', s=400, c='black', zorder=5,
           label='GeneBreaker output')
ax.legend(); ax.set_title('GENERator Intermediate Activations — Pathogenicity Probe')
plt.savefig('probe_tsne.png', dpi=150)
```

This is your money shot for the demo.

---

### Decision tree: what to do if things break

```
Probe AUC < 0.65?
  └─ Try kernel SVM: SVC(kernel='rbf', C=10, probability=True)
  └─ Try different layer index (7 vs 14 vs 21 vs -1)
  └─ Try eukaryote model: GENERator-v2-eukaryote-1.2b-base
  └─ If still fails: pivot to ESM-2 cross-modal (translate → protein → virulence classifier)

PathoLM checkpoint broken?
  └─ Replace with k-mer random forest: RandomForestClassifier on k=6 freq vectors
  └─ Or DeePaC: github.com/rki-mf1/DeePaC

BLAST too slow (>10s per query)?
  └─ Switch to commec --skip-tx (HMM-only, <1 GB DB, much faster)
  └─ Or skip BLAST, rely on kmer + probe
```

---

### Minimum viable demo (3 minutes)

1. Show a JailbreakDNABench SARS-CoV-2 sequence being screened (or generated by Evo2 if available)
2. Show the sequence flagged by k-mer + PathoLM + BLAST + probe layers
3. Show the t-SNE — the sequence lands in the red (pathogenic) cluster
4. Show a benign E. coli sequence passing cleanly
5. Show the ROC curve comparing individual layers vs. combined

The **probe t-SNE** is the visual that wins the demo. It shows the model itself "knows" the sequence is pathogenic, even without explicit safety training.

---

### The one-sentence novelty claim

> We show that pathogenicity is linearly encoded in GENERator's intermediate activation space, and exploit this to build the first real-time multi-layer biosecurity guardrail for DNA foundation model outputs.

---

| Approach | Why not |
|---|---|
| **DPO fine-tuning of Evo2** | LoRA for StripedHyena doesn't exist; ~20h to implement |
| **Generation-time rejection sampling** | Requires deep Vortex engine modification; vulnerable to salami attacks |
| **Watermarking from scratch** | DNAMark already exists (arXiv:2509.18207); don't reinvent it |
| **commec full install** | BLAST taxonomy layer takes hours to download NCBI databases; use `--skip-tx` biorisk-only mode if you do use it |

---

## Key References

| Paper | Link |
|---|---|
| GeneBreaker (the attack) | https://arxiv.org/abs/2505.23839 |
| Evo2 (attack target model) | https://www.biorxiv.org/content/10.1101/2025.02.18.638918 |
| GENERator (probe backbone) | https://arxiv.org/abs/2502.07272 |
| PathoLM | https://arxiv.org/abs/2406.13133 |
| DNAMark/CentralMark (watermarking) | https://arxiv.org/abs/2509.18207 |
| Wang et al. Nature Biotech (defense roadmap) | https://www.nature.com/articles/s41587-025-02608-0 |
| IBBIS commec | https://github.com/ibbis-screening/common-mechanism |
| SecureDNA | https://securedna.org |
| NeurIPS 2025 BioSafe GenAI Workshop | https://neurips.cc/virtual/2025/workshop/84714 |