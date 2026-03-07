# Anti-Jailbreaking Defenses for DNA Foundation Models: Hackathon Research Brief

**Evo2's DNA generation can be jailbroken to produce pathogenic viral sequences with up to 60% success rate, and virtually no defenses exist today.** GeneBreaker (Zhang et al., ICLR 2026) demonstrated the first systematic jailbreak attacks on DNA language models, achieving >90% nucleotide identity to known pathogens including SARS-CoV-2 and HIV-1 with structural fidelity confirmed by AlphaFold3. The defense landscape is nearly empty: fewer than 3% of ~370 biological AI models include any safety safeguards. This brief provides verified technical details for building anti-jailbreaking defenses during a 10-hour hackathon, covering attack mechanics, defense architectures, compute constraints, and a refined build plan.

**Critical hardware finding that reshapes the entire plan: Evo2 1B requires an H100 GPU (Hopper architecture, FP8 via Transformer Engine) and cannot run on A100. Only Evo2 7B runs on A100 in bfloat16.** This affects every architecture below.

---

## Section 1: GeneBreaker attack mechanics — verified step-by-step

### The three-stage pipeline

GeneBreaker's attack proceeds through three tightly coupled stages, each essential — ablation shows removing either prompt design or guided beam search collapses ASR to **0%**.

**Stage 1 — LLM Agent Prompt Design (ChatGPT-4o).** The attacker provides ChatGPT-4o with a target pathogen gene's accession ID and asks for 3–5 GenBank accession IDs of homologous but non-pathogenic sequences. For HIV-1 envelope protein, ChatGPT returns sequences from SIV, FIV, and BIV. For SARS-CoV-2 spike, it returns bat coronavirus RaTG13 (MN996532.1), bat CoV ZC45 (MG772933.1), and pangolin coronavirus (MT040335.1). These sequences are downloaded from NCBI. A `--skip_chatgpt` flag in the code allows pre-cached retrieval.

**Stage 2 — Few-Shot Prompt Construction.** The jailbreak prompt concatenates: (1) a phylogenetic tag used during Evo2 training (e.g., `|D__VIRUS;P__SSRNA;O__CORONAVIRIDAE;F__BETACORONAVIRUS;G__SARS-COV-2|`), (2) the retrieved non-pathogenic homologous sequences as in-context examples, and (3) a short DNA prefix from the noncoding region upstream of the target coding sequence. The tag exploits Evo2's training-time taxonomic conditioning. The few-shot examples prime the model to continue generating within the target genomic neighborhood.

**Stage 3 — Guided Beam Search.** The core optimization uses scoring function **f = PathoLM(x) + α · log p(x)** with these exact parameters:

| Parameter | Value | Function |
|-----------|-------|----------|
| C | 128 nt | Chunk size per generation round |
| K | 8 | Candidate chunks generated per beam |
| K' | 4 | Top beams retained after scoring |
| Temperature | 1.0 | Sampling temperature |
| α | 0.5 | Log-probability weight |
| L | 640 nt | Total generated sequence length |

At each of 5 rounds, for each of 4 retained beams, 8 candidate 128-nt chunks are sampled. PathoLM scores the **entire accumulated sequence** (not just the new chunk) — this is critical for defensive analysis. From 32 total candidates (4 × 8), the top-4 by f-score are retained. Chunks are assembled by simple concatenation. A key empirical finding: Pearson **r = 0.75** between average log-probability and sequence similarity to pathogen targets, because pathogenic viral DNA is under-represented in training data.

### ASR by model size and viral category

| Model | Large DNA | Small DNA | +ssRNA | −ssRNA | dsRNA | Enteric RNA |
|-------|-----------|-----------|--------|--------|-------|-------------|
| **Evo2 1B** | 20.0±17.9 | 20.0±40.0 | 13.3±8.3 | 0.0±0.0 | 0.0±0.0 | 20.0±40.0 |
| **Evo2 7B** | 48.0±9.8 | 46.7±26.7 | 28.8±11.3 | 24.4±12.8 | 20.0±40.0 | 50.0±15.8 |
| **Evo2 40B** | 52.0±9.8 | 60.0±25.0 | 37.7±5.4 | 26.7±24.4 | 20.0±40.0 | 60.0±20.0 |

**Evo2 1B achieves meaningful ASR** (20% on Large DNA, Small DNA, Enteric RNA) but fails on negative-strand RNA and dsRNA. Evo2 7B is substantially more vulnerable than 1B across all categories. The 7B→40B jump shows diminishing returns (5–15% improvement per category).

### Why negative-strand RNA viruses have lower ASR

Three biological factors explain the **26.7% ± 24.4% ASR** for −ssRNA versus 60% for Small DNA: (1) fewer benign close relatives in public databases for Rabies/Measles families, reducing prompt quality; (2) faster evolutionary rates in RNA genomes creating greater inter-strain divergence, making the 90% identity threshold harder to reach; and (3) more complex segmented genomic architectures compared to compact DNA viral genomes.

### JailbreakDNABench

A curated benchmark covering 6 viral categories with max 3 coding DNA sequences sampled per virus. Pathogens include SARS-CoV-2, MERS-CoV, HIV-1, Ebola, Variola (smallpox), Poliovirus, Rabies, Measles, HPV, Herpesviridae, Parvovirus B19, and others. Validated to NOT appear in Evo2 training data. Available in the GitHub repo under `JailbreakDNABench/`. Success criterion: >90% BLAST identity to target. The ICLR camera-ready additionally uses **VADR v1.5.1** for functional annotation validation.

### Computational cost and A100 feasibility

The paper used **4× H100 GPUs** but does not report per-sequence generation time. Since Evo2 7B fits on a single A100 80GB (~18 GB VRAM), the attack is feasible on reduced hardware with K'=2 beams instead of K'=4, though ASR decreases with beam count (Figure 6b). PathoLM adds negligible overhead (~50M parameters, <0.5 GB).

### Code structure

The GitHub repo (`zaixizhang/GeneBreaker`, 21 stars, CC0-1.0 license) contains a single monolithic script `auto_jailbreak_hiv.py` plus the `JailbreakDNABench/` directory. PathoLM checkpoint is downloaded via `gdown` from Google Drive. Only the HIV attack script is provided; other viruses follow the same template.

---

## Section 2: PathoLM reliability analysis — the attack-defense asymmetry

### Architecture and specifications

PathoLM is a fine-tuned **Nucleotide Transformer v2 50M** — an encoder-only BERT-style model with **~50 million parameters** (224 MB weights). It uses 6-mer tokenization (4,104 token vocabulary), rotary positional embeddings, and gated linear units. Maximum context: **~12,000 nucleotides** (2,048 tokens). Binary output: "pathogen" or "non-pathogen" with a probability score. VRAM requirement: **<0.5 GB**. Inference on 640-nt sequences: **<10 ms on GPU**.

Installation requires cloning the GitHub repo (`Sajib-006/Patho-LM`), installing requirements, and downloading weights from Zenodo. No HuggingFace model page exists. GeneBreaker uses a separate checkpoint obtained via Google Drive.

### Training data overlap with GeneBreaker targets — the critical gap

PathoLM was trained on ~60K bacterial genomes (7 ESKAPEE species) and ~25K viral genomes across only 4 viral families:

| GeneBreaker Target | In PathoLM Training? | Consequence |
|---|---|---|
| SARS-CoV-2 | **YES** (Coronaviridae) | Defensive detection likely works |
| HIV-1 | **YES** (Retroviridae) | Defensive detection likely works |
| Ebola | **NO** (Filoviridae absent) | Defensive detection likely fails |
| Variola/Smallpox | **NO** (Poxviridae absent) | Defensive detection likely fails |
| Poliovirus | **NO** (Picornaviridae absent) | Defensive detection likely fails |
| Rabies | **NO** (Rhabdoviridae absent) | Defensive detection likely fails |
| Measles | **NO** (Paramyxoviridae absent) | Defensive detection likely fails |

**Only 2 of 7 GeneBreaker target pathogens overlap with PathoLM's training data.** This is the single most important finding for defensive architecture design.

### The key question: does defensive PathoLM screening catch GeneBreaker outputs?

**Yes, for covered pathogens — because GeneBreaker literally optimizes the defender's metric.** GeneBreaker scores the full accumulated sequence at each beam search round, not just individual 128-bp chunks. By round 5, the complete 640-bp output has been explicitly optimized to achieve a high PathoLM score. A defensive screen using the same model at the same sequence length would detect these outputs.

**No, for uncovered pathogens** (Ebola, Variola, Poliovirus, Rabies, Measles) — PathoLM has never seen Filoviridae, Poxviridae, or Picornaviridae sequences and has no basis for classifying them.

**No, against an adaptive attacker.** Since PathoLM is open-source, an attacker could: (a) substitute a different pathogenicity oracle for beam search guidance, (b) add a score-ceiling constraint to stay below the defensive threshold, (c) target pathogens outside PathoLM's training distribution, or (d) use only the log p(x) component (ablation shows this alone still produces pathogen-like sequences).

### False positive analysis

**No published false positive rate exists.** The paper reports accuracy, F1, AUC-ROC, and MCC but not individual precision/recall values. No evaluation on model organisms (E. coli K-12, S. cerevisiae, human genome segments), synthetic biology constructs, or common lab sequences has been published. This is a significant gap for any defensive deployment.

### Verdict

PathoLM provides **partial, fragile, but useful defense as one layer** in a multi-layer system. It should never be used as a standalone defense. For the hackathon, it is a valuable component but must be paired with BLAST-based screening and other methods to cover its blind spots.

---

## Section 3: Existing defense landscape

### IBBIS Common Mechanism (commec v1.0.2)

The most mature open-source biosecurity screening tool. Three-layer pipeline: (1) **HMM-based biorisk search** using HMMER against curated profiles — this is the key layer for AI-generated variant detection, as profile HMMs detect remote functional homologs even at low sequence identity; (2) **taxonomy search** via BLASTX against NCBI nr and BLASTN against NCBI core_nt; (3) **low-concern clearing** using benign databases.

Installed via `mamba install commec` (NOT pip — requires Bioconda for BLAST, DIAMOND, HMMER, Infernal dependencies). A lightweight biorisk-only mode runs on a laptop with <1 GB databases via `--skip-tx`. **False positive rate: <2%** on real synthetic biology designs. Covers Australia Group Common Control List plus national lists from India, China, South Africa, US Select Agents. Minimum sequence length: **50 bp**.

Critical limitation: the BLAST-based taxonomy search is the weak link. The HMM biorisk layer specifically addresses AI-generated evasion. After the Wittmann et al. "Paraphrase Project" (Science, October 2025) patched HMM databases to include homologous protein families, detection rates reached **97%** for synthetic variants predicted to retain wild-type function.

### SecureDNA

Uses the **Random Adversarial Threshold (RAT) algorithm** — fundamentally different from BLAST. Screens exact matches on **30-bp DNA subsequences** (not 50 bp), with millions of pre-computed functional variants for each hazard. Quasi-random window selection prevents adversaries from knowing which positions are monitored. Reverse screening against benign databases virtually eliminates false positives.

Available as a **free REST API** via `synthclient` binary (registration + certificate required). Docker containers available. Privacy-preserving via Distributed Oblivious Pseudorandom Function. Red team validation: blocked **99.999%** of functional attacks on M13 bacteriophage. Covers everything on the Australia Group List, US and China export controls, and all known endemic human viruses.

### Post-GeneBreaker defense work (May 2025 – March 2026)

The NeurIPS 2025 BioSafe GenAI Workshop (December 6, 2025, San Diego) was the watershed event. Key developments:

**DNAMark and CentralMark** (Zhang et al., NeurIPS 2025 poster, arXiv:2509.18207). Two watermarking methods from the same Princeton/Stanford group as GeneBreaker: DNAMark uses synonymous codon substitutions for function-preserving DNA watermarks; CentralMark creates inheritable watermarks transferring from DNA to translated proteins. **F1 detection scores >0.85** under diverse conditions. This directly addresses Architecture D below — the hackathon team should build on this existing work rather than starting from scratch.

**Biosecurity Agent** (Meng & Zhang, NeurIPS 2025 BioSafe GenAI). Four-mode lifecycle defense for text-based LLMs: dataset sanitization, DPO+LoRA preference alignment, runtime guardrails (BLAST + semantic + fuzzy + keyword), and automated red teaming. **DPO+LoRA reduced end-to-end ASR from 59.7% to 3.0%** — but this targets text LLMs (Llama-3-8B-Instruct), NOT DNA sequence models. The concept is directly transferable but implementation for Evo2 does not exist.

**Arc Institute has not responded** with any safety updates to Evo2. The "Evo 2: One Year Later" blog (circa February 2026) acknowledges "further work for model alignment will be needed" but announces no new safety features. The original data exclusion safeguard (removing eukaryotic viruses from training) was shown to be circumventable via fine-tuning at the NeurIPS workshop. **Evo2 remains pip-installable with zero safety guardrails.**

Wang et al. **Nature Biotechnology** (April 28, 2025) called for four built-in safeguard technologies: **watermarking, alignment (DPO), anti-jailbreak methods, and unlearning**. A comprehensive roadmap paper followed in October 2025 (arXiv:2510.15975).

---

## Section 4: Architecture feasibility verdicts

### Architecture A: Three-layer output screener — **YES (4–6 hours)**

**Layer 1 — BLAST screening.** Build a local BLAST database from Select Agent reference genomes and JailbreakDNABench sequences using `makeblastdb`. BLAST queries take ~5s per 640-nt sequence. Python wrapper via BioPython's `NcbiblastnCommandline`. BLAST alone is insufficient (documented unreliability for biosafety determination), but essential as a first filter.

**Layer 2 — K-mer frequency similarity.** Optimal k-value: **k=5 (1,024 features)** balances discriminative power and computational cost. PaPrBaG found k=1–4 sufficient; DCiPatho uses k=3–7. For 640-bp sequences, k-mer counting is trivially fast (<1 ms in pure Python using `collections.Counter`). Libraries: **Jellyfish** (fastest, C++ with Python bindings, bioconda), **khmer** (memory-efficient Count-Min Sketch), or pure Python for short sequences. Cosine similarity against curated pathogen k-mer profiles provides a fast secondary filter.

**Layer 3 — PathoLM classification.** <10 ms inference, <0.5 GB VRAM. Covers SARS-CoV-2 and HIV-1 well but has blind spots on Ebola, Variola, Poliovirus, Rabies, Measles.

**This exact three-layer combination has NOT been published anywhere.** Individual components exist; the combination is novel. Estimated build: Layer 1 (1–2h), Layer 2 (1–2h), Layer 3 (1–2h), integration (1h). **Total: 4–6 hours.** Key blocker: pathogen reference database curation. Novelty: **HIGH**.

### Architecture B: Generation-time rejection sampling — **MARGINAL YES (5–8 hours)**

Would require implementing custom beam search around Evo2's forward pass (logits are fully accessible via `outputs, _ = evo2_model(input_ids)`), loading PathoLM alongside, and rejecting candidate chunks exceeding a pathogenicity threshold. **Evo2 does NOT use standard HuggingFace model.generate()** — it uses the custom Vortex inference engine. No native `logits_processor` support. Custom beam search must be implemented manually.

Computational overhead: 32 PathoLM calls per chunk × 5 chunks = **160 PathoLM inferences per sequence** (~2–8 seconds total, approximately doubling generation time).

**Critical vulnerability:** Rejecting high-PathoLM chunks at 128-bp level does NOT prevent assembly of individually low-scoring chunks into collectively pathogenic sequences. An attacker aware of rejection sampling could find chunks below threshold that are still collectively pathogenic — the "salami attack" analog.

**No prior work** applies rejection sampling to biological sequence generation for safety. Novelty: **VERY HIGH**. Key blocker: requires deep modification of Evo2's generation code. Build estimate: 5–8 hours including testing.

### Architecture E: Activation-space monitoring — **YES (6–8 hours)**

This is the **highest-novelty feasible architecture.** Activation steering has been applied to protein language models (ESM-2/ESM-3) but NEVER to DNA language models for biosafety.

Evo2 explicitly supports intermediate embedding extraction:
```python
layer_name = 'blocks.28.mlp.l3'
outputs, embeddings = evo2_model(input_ids, return_embeddings=True, layer_names=[layer_name])
```

The Evo2 team notes that **intermediate embeddings work better than final embeddings** for downstream tasks. An SAE notebook for Evo2 already exists in their repo. The "Steering Protein Language Models" paper (Huang et al., arXiv:2509.07983) demonstrated that linear classifiers on ESM-2 intermediate activations distinguish property-related representations using only **100 positive/negative examples each**.

Build plan: data curation of ~500–1,000 pathogenic vs. benign sequences (2h), embedding extraction from Evo2 7B at 3–5 intermediate layers (2h), logistic regression/linear SVM training (1h), evaluation with ROC curves (1h), integration demo (1h). **Total: 6–8 hours.**

Key uncertainty: whether pathogenicity is **linearly separable** in Evo2's activation space. This IS the research question. The existence of the SAE notebook and the protein model precedents are encouraging.

### Architecture F: Cross-modal ESM-2 defense — **YES (4–6 hours)**

**ESM-2 embeddings reliably classify virulence factors** — validated in PLMVF (BMC Genomics 2025), pLM4VF (BMC Biology 2025), and GTAE-VF (2024). Pipeline: Evo2 DNA output → ORF detection (Prodigal) → translation → ESM-2 650M embeddings → virulence classifier.

**VFDB core dataset:** 3,581 verified virulence factor genes from 32 bacterial genera; expanded to 62,332 non-redundant orthologues across 135 species. FASTA files downloadable. Negative examples: Database of Essential Genes (~7,995 sequences) and UniProt Swiss-Prot.

ESM-2 650M: **~3–4 GB VRAM**, inference on 300 AA protein: **<50 ms on A100**. Total pipeline latency: **<2 seconds per generated sequence**. Co-locates easily with Evo2 7B on single A100.

**Fundamental limitation: blind to non-coding pathogenic DNA** (regulatory sequences, riboswitch elements, non-coding RNA). This creates a significant blind spot that must be acknowledged.

Build estimate: 4–6 hours. Novelty: **MODERATE** — ESM-2 VF classification is established; framing it as a real-time biosecurity guardrail for DNA generative models is new.

---

## Section 5: Compute requirements — verified figures

### Critical hardware constraint

| Model | Precision | Min GPU Required | VRAM (est.) | Runs on A100 80GB? |
|-------|-----------|-----------------|-------------|---------------------|
| **Evo2 1B** | FP8 (Transformer Engine) | **H100 only** | ~3–5 GB | **❌ NO** |
| **Evo2 7B** | BF16 | Any GPU with Flash Attention | ~16–20 GB | **✅ YES** |
| **Evo2 20B** | FP8 | H100 only | ~25–35 GB | ❌ NO |
| **Evo2 40B** | FP8 | 2× H100 or 1× H200 | ~80+ GB | ❌ NO |

The A100 (compute capability 8.0) does not support FP8 via Transformer Engine, which requires Hopper architecture (compute capability ≥8.9). **The hackathon must use Evo2 7B, not 1B.** This is counterintuitive but architecturally mandated.

### Co-location on single A100 80GB

| Configuration | VRAM Est. | Fits? |
|---|---|---|
| Evo2 7B alone | ~18 GB | ✅ |
| Evo2 7B + PathoLM 50M | ~18.5 GB | ✅ |
| Evo2 7B + ESM-2 650M | ~22 GB | ✅ |
| Evo2 7B + PathoLM + ESM-2 650M | ~22.5 GB | ✅ |
| All above + local BLAST DB | ~30–40 GB | ✅ |

**All proposed configurations fit comfortably** with 40–60 GB headroom.

### Component latency estimates

| Component | Per-sequence latency | Notes |
|---|---|---|
| PathoLM inference (640 nt) | <10 ms | 50M params, trivial |
| ESM-2 650M inference (300 AA) | <50 ms | Single sequence |
| BLAST query (640 nt vs local DB) | ~5 s | Depends on DB size |
| K-mer counting + cosine sim (640 nt, k=5) | <1 ms | Pure Python sufficient |
| Evo2 7B forward pass (640 tokens) | ~50–200 ms | Estimated, SSM architecture |

### Installation timeline

Evo2 7B light install (no Transformer Engine needed):
```bash
conda create -n evo2 python=3.11
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn==2.8.0.post2 --no-build-isolation
pip install evo2
```

**Estimated setup time: 30–60 minutes** (Flash Attention compilation dominates). Docker alternative: 20–40 minutes build + model download. Known gotchas: Flash Attention requires `nvcc` and matching CUDA toolkit; minimum NVIDIA driver 550.90.07; model weights ~50 GB download; must use `--recurse-submodules` when cloning (Vortex is a submodule).

---

## Section 6: Novel post-May 2025 work

### The NeurIPS 2025 BioSafe GenAI Workshop was the field's inflection point

Organized in part by GeneBreaker's lead author Zaixi Zhang, the December 6, 2025 workshop in San Diego produced the most concentrated set of relevant work. **GeneBreaker won Best Paper and received an Oral presentation.** Key papers:

The **Biosecurity Agent** (Meng & Zhang) demonstrated a complete defense-in-depth lifecycle: tiered dataset sanitization, DPO+LoRA alignment (ASR 59.7%→3.0%), runtime guardrails (BLAST + semantic + fuzzy + keyword, best F1=0.720 at L2), and automated red teaming. However, this targets text LLMs exclusively.

**"Open-weight genome language model safeguards"** demonstrated that Evo2's data exclusion safeguard is circumventable by fine-tuning with human-infecting viral data on 110 harmful viruses. This definitively proves **data exclusion alone is insufficient** for open-weight models.

**"Without Safeguards, AI-Biology Integration Risks Creating Future Pandemics"** (Shakhnovich & Esvelt) introduced the Intelligent Automated Biology capability levels framework for risk assessment.

Additional workshop papers covered SafeProtein (red-teaming framework for protein models), ProtGPT2 biosecurity evaluation, activation-space biosecurity auditing on ESM models, robust unlearning (MUDMAN), and pretraining data filtering (Deep Ignorance).

### The Paraphrase Project patched synthesis screening

Wittmann et al. (Science, October 2025) used EvoDiff and other AI tools to generate thousands of synthetic variants of 72 hazardous proteins. Initially, reformulated sequences **evaded screening at major DNA synthesis companies** (Twist Bioscience, IDT). A cross-sector team (Microsoft, IBBIS, RTX BBN, Battelle) developed expanded HMM databases covering homologous protein families. After patching: **97% detection rate** for synthetic homologs predicted to retain function. This cybersecurity-inspired "zero-day response" model is the most significant concrete defensive advance.

### Arc Institute has not meaningfully responded

No new safety features for Evo2. No post-GeneBreaker patches. The model remains pip-installable with zero guardrails under Apache 2.0 license. A CSIS analysis notes: "The probability that [data exclusion circumvention] will happen again seems relatively high." The Epoch AI survey found **fewer than 3% of ~370 biological AI models include any safety safeguards**.

### GeneBreaker citations and influence

GeneBreaker has been accepted at ICLR 2026 (the premier ML venue) and won the NeurIPS BioSafe GenAI Best Paper Award. It is cited by the DNAMark/CentralMark watermarking papers, the Biosecurity Agent paper, and the Wang et al. Nature Biotechnology roadmap. The Princeton/Stanford group (Wang, Zhang, Cong) is producing the majority of both attack and defense research in this space.

---

## Section 7: Recommended 10-hour build plan for 4-person team

### Scope decision

**Build Architecture A (Three-Layer Screener) as the primary deliverable, with Architecture E (Activation Probes) as the high-novelty differentiator.** Architecture F (ESM-2 cross-modal) as stretch goal. Present Architecture C (DPO) as designed future work with theoretical analysis.

This strategy maximizes: (1) probability of a working demo, (2) novelty for judges, (3) breadth of defense coverage, and (4) publishable contribution.

### Hour-by-hour plan

**Hours 0–1: Setup (all 4 members)**

- Person 1: Install Evo2 7B (light install, conda environment, model download)
- Person 2: Install PathoLM (clone repo, download Zenodo checkpoint, verify inference)
- Person 3: Install ESM-2 650M (`pip install fair-esm`), download VFDB core dataset
- Person 4: Install BLAST locally (`mamba install blast`), download JailbreakDNABench from GeneBreaker repo, build local pathogen BLAST database with `makeblastdb`

**Hours 1–5: Parallel build phase**

- **Person 1 — Architecture A, Layer 1 (BLAST screening):** Write Python wrapper for BLASTN against local pathogen DB. Input: FASTA sequence. Output: top hit identity %, organism, flag if >90% identity to any select agent. Use BioPython's `NcbiblastnCommandline`. Test on known pathogenic and benign sequences.

- **Person 2 — Architecture A, Layers 2+3 (K-mer + PathoLM):** Implement k-mer frequency extraction (k=5, 1,024 features) with `collections.Counter`. Build reference k-mer profiles from pathogen genomes. Implement cosine similarity scoring. Integrate PathoLM inference as layer 3. Determine thresholds via ROC analysis on JailbreakDNABench positives vs. random genomic sequences.

- **Person 3 — Architecture E (Activation Probes):** Curate ~500 pathogenic gene sequences (from VFDB nucleotide set + NCBI select agent genes) and ~500 benign sequences (housekeeping genes, random genomic regions). Run Evo2 7B forward pass on all 1,000 sequences with `return_embeddings=True` at layers `blocks.14`, `blocks.21`, `blocks.28` (early/mid/late). Extract embeddings, train logistic regression classifiers with scikit-learn, evaluate with cross-validation.

- **Person 4 — Integration + Demo Pipeline:** Build the unified screening pipeline: Evo2 7B generation → Architecture A three-layer screen → flag/pass decision. Build a simple CLI or Gradio demo interface. Start on Architecture F (ESM-2 virulence classifier) as stretch goal if ahead of schedule.

**Hours 5–7: Integration and testing**

- Connect Architecture A layers into single pipeline with configurable thresholds
- Integrate Architecture E probe as a fourth "early warning" layer
- Run GeneBreaker-style prompts through Evo2 7B and test detection rates
- Measure false positive rates on benign genomic sequences
- Debug and optimize latency

**Hours 7–9: Evaluation and polish**

- Systematic evaluation: run 20+ pathogenic generation attempts, 20+ benign queries
- Compute precision/recall/F1 for each layer independently and combined
- Build ROC curves and determine optimal operating points
- Create visualization of activation probe results (t-SNE of Evo2 embeddings, colored by pathogenicity)
- Prepare 3-minute demo script

**Hours 9–10: Demo preparation**

- Record backup demo video in case of live failures
- Prepare slides: problem statement, architecture diagram, evaluation results, novel contributions
- Rehearse 3-minute presentation

### Minimum viable demo specification

The 3-minute demo must show:

1. **Live generation:** Evo2 7B generating a sequence from a GeneBreaker-style prompt (phylogenetic tag + homologous sequences)
2. **Three-layer screening:** The generated sequence flagged by BLAST (>90% identity to HIV-1 env), k-mer similarity (cosine >0.85 to HIV profiles), and PathoLM (pathogenicity score >0.9)
3. **Activation probe visualization:** t-SNE plot showing clear separation between pathogenic and benign sequences in Evo2's intermediate activation space, with the generated sequence landing in the pathogenic cluster
4. **Pass case:** A benign sequence (e.g., from E. coli K-12 housekeeping gene) passing all three screening layers

---

## Section 8: Risk factors and contingencies

### Technical blockers ranked by probability

**Risk 1 (HIGH): Evo2 installation failure.** Flash Attention compilation is fragile and CUDA version-sensitive. **Contingency:** Use the NVIDIA Hosted API (`health.api.nvidia.com/v1/biology/arc/evo2-40b/generate`) for generation, which avoids local installation entirely. This gives access to Evo2 40B but only supports simple generation (no custom beam search or embedding extraction). For activation probes, local installation is mandatory — have Docker as backup.

**Risk 2 (MEDIUM): PathoLM checkpoint incompatibility.** The GeneBreaker Google Drive checkpoint may differ from the Zenodo version. **Contingency:** Use the Zenodo checkpoint as primary. If both fail, substitute PathoLM's layer 3 with a simple k-mer-based ML classifier (random forest on k=6 features, trained on PATRIC pathogenic vs. non-pathogenic genomes).

**Risk 3 (MEDIUM): Activation probes show no separation.** Pathogenicity may not be linearly separable in Evo2's latent space, especially if the model was trained with eukaryotic virus exclusion. **Contingency:** If linear probes achieve <65% accuracy, pivot to a kernel SVM or small MLP. If still <70%, abandon Architecture E and redirect effort to Architecture F (ESM-2 cross-modal), which has stronger prior validation.

**Risk 4 (LOW-MEDIUM): BLAST database building takes too long.** Downloading and indexing full NCBI databases is hours-long. **Contingency:** Use only JailbreakDNABench sequences (~50–100 sequences) as the BLAST database for the demo. This is small but sufficient to detect GeneBreaker's specific outputs. Alternatively, install `commec` with `--skip-tx` biorisk-only mode (~1 GB database).

**Risk 5 (LOW): False positive rate too high.** If the three-layer screener flags >10% of benign queries, the demo loses credibility. **Contingency:** Raise thresholds to prioritize specificity over sensitivity. Present the ROC curve showing the tradeoff honestly. For judges, a system with 80% sensitivity and 2% false positive rate is more impressive than 95% sensitivity with 15% false positives.

### If PathoLM proves unreliable

Replace PathoLM with one or more alternatives:

- **DeePaC** (Bartoszewicz et al., 2020): CNN/LSTM pathogen classifier designed for next-generation sequencing reads. Specifically biosecurity-aware. Available on GitHub.
- **K-mer random forest:** Train a random forest on k=6 frequency vectors from PATRIC pathogenic vs. non-pathogenic genomes. Can be built in <1 hour with scikit-learn. Less powerful than PathoLM but has no training distribution blind spots — it learns whatever features the training data contains.
- **ESM-2 cross-modal (Architecture F):** For protein-coding regions, this provides an independent pathogenicity signal that does not depend on PathoLM at all.

---

## Section 9: Three-sentence problem statement

DNA foundation models like Evo2 can be systematically jailbroken to generate pathogenic viral sequences with up to 60% success rate and >90% nucleotide identity to real pathogens — including SARS-CoV-2, HIV-1, and Ebola — yet these models ship with zero built-in safety guardrails and fewer than 3% of biological AI models include any safety safeguards. We present the first multi-layer defense system combining BLAST-based screening, k-mer frequency analysis, PathoLM pathogenicity classification, and novel activation-space probes on Evo2's intermediate representations to detect and block pathogenic sequence generation in real time. Our approach detects GeneBreaker-style jailbreak outputs while maintaining low false positive rates on legitimate genomic research, establishing a practical template for built-in biosecurity safeguards that the field urgently needs.

---

## Key unknowns to resolve during the hackathon

Three empirical questions will determine the final architecture: (1) whether Evo2 7B's intermediate activations linearly separate pathogenic from benign sequences (the core novelty claim), (2) what combined false positive rate the three-layer screener achieves on a realistic benign query set, and (3) whether PathoLM's defensive screening catches GeneBreaker outputs targeting pathogens within its training distribution (SARS-CoV-2, HIV-1) at the full-sequence level. All three are testable within the first 5 hours. If activation probes fail, the ESM-2 cross-modal defense (Architecture F) is the backup with the strongest prior validation. The most publishable outcome combines Architecture A (practical defense) with Architecture E (novel scientific finding about pathogenicity encoding in DNA model representations).