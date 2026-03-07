# Deep Research Brief: Anti-Jailbreaking Defences for DNA Foundation Models
## For: AI Research Agent — Deep Research Task
## Context: 10-Hour Hackathon Project Scoping

---

## 1. BACKGROUND AND MOTIVATION

### 1.1 What Are DNA Foundation Models?

DNA foundation models are large-scale autoregressive language models trained on genomic sequences
instead of text. They treat DNA nucleotides (A, C, G, T) as tokens and learn the statistical
structure of genomes at scale, enabling both **sequence understanding** (classification, variant
effect prediction) and **sequence generation** (de novo synthesis of functional DNA).

Key open-source models relevant to this research:

| Model | Size | Context | Developer | Generation? |
|---|---|---|---|---|
| Evo 1 | 7B | 650k tokens | Arc Institute | Yes |
| Evo 2 | 1B / 7B / 40B | 1M tokens | Arc Institute | Yes |
| Nucleotide Transformer | 500M / 2.5B | 6k tokens | InstaDeep | Limited |
| GENERator | 1.2B / 3B | 98k tokens | — | Yes |
| HyenaDNA | up to 1.6B | 1M tokens | — | Yes |
| DNABERT-2 | 117M | 512 tokens | — | No |

The most capable and relevant model is **Evo 2** (Arc Institute, published March 2026 in Nature),
which was trained on 9.3 trillion base pairs across all domains of life and can generate
chromosome-scale sequences including sequences similar to human mitochondrial DNA.

### 1.2 The Jailbreaking Problem — GeneBreaker (May 2025)

The paper **"GeneBreaker: Jailbreak Attacks against DNA Language Models with Pathogenicity
Guidance"** (Zhang et al., arXiv:2505.23839, May 2025, Princeton/Stanford/Zhejiang) is the
first systematic study of jailbreak vulnerabilities in DNA foundation models.

**Key finding**: Evo2 (40B) achieves up to **60% Attack Success Rate** across 6 viral categories
including SARS-CoV-2, HIV-1, Ebola, Variola (smallpox), and Poliovirus. Generated sequences
achieve >90% nucleotide identity to known pathogens and are structurally faithful when predicted
by AlphaFold3 (RMSD 0.334 for HIV-1 envelope protein).

**Critical distinction from LLM jailbreaking**: DNA model jailbreaking does NOT involve
bypassing safety training — Evo2 has essentially no safety alignment. Instead, it means
successfully steering generation toward pathogenic sequences via:
1. **Prompt engineering**: Using high-homology non-pathogenic sequences (e.g., bat coronavirus,
   feline immunodeficiency virus) as few-shot context to prime generation
2. **Guided beam search**: Scoring generated chunks with PathoLM (a pathogenicity classifier)
   + average log-probability, keeping most pathogenic continuations
3. **Evaluation**: BLAST against a curated Human Pathogen Database (JailbreakDNABench) at
   >90% identity threshold

The GeneBreaker scoring function is: `f = PathoLM(x) + α · log p(x)`

**Scaling amplifies risk**: Attack success rates increase monotonically with model size
(Evo2 1B < Evo1 7B < Evo2 7B < Evo2 40B), meaning larger, more capable models are MORE
dangerous, not less.

**The paper explicitly calls for defences**: The conclusion section states the need for
"robust defense mechanisms, enhanced detection systems, and safer model architectures" —
and provides no working defence implementation.

### 1.3 Why This Is a Real, Urgent Problem

- **Post-release fine-tuning**: Evo1 excluded eukaryotic viral genomes from training.
  Within weeks of release, researchers had fine-tuned the published weights with human
  virus data — demonstrating that training-time exclusions can be reversed cheaply.
- **No regulatory framework covers AI-generated sequences**: Current DNA synthesis
  screening frameworks (OSTP 2024, HHS 2023) screen human-ordered sequences, not
  AI model outputs. There is a complete governance gap.
- **Open weights problem**: Unlike GPT-4, Evo2's weights are publicly downloadable.
  Any researcher (or bad actor) can run the GeneBreaker attack locally with no API
  monitoring.
- **Synthesis accessibility**: Commercial DNA synthesis costs ~$0.10/bp and is available
  globally with inconsistent screening. A jailbreak-generated sequence can be submitted
  to synthesis directly.

### 1.4 What PathoLM Is

PathoLM is a key tool in this ecosystem. It is a DNA language model fine-tuned specifically
for pathogenicity prediction in bacterial and viral sequences, built on Nucleotide Transformer.
It was trained on ~30 species of viruses and bacteria and outputs P(pathogenic) for a DNA
sequence. Crucially, it is **open source**. GeneBreaker uses it offensively (to guide
beam search toward pathogenic outputs). A defence can use it **defensively** (to reject
or flag generated sequences). GitHub: https://github.com/SajibAcharjee/PathoLM

### 1.5 The Team and Constraints

- **Team**: 3-5 people with strong CS/systems background. Proficient in PyTorch, Python,
  C/C++, some FPGA. Background in ML/RL, signal processing, Bayesian inference.
  One team member has research experience at Anthropic on LLM safety (DPO, RLHF,
  unfaithful reasoning).
- **Time**: ~7-10 hours of build time at hackathon
- **Compute**: Access to GPU (assume 1-2 A100 80GB or equivalent)
- **Hackathon track**: AI Security track — judged on theory of change, not presentation.
  Must clearly address how project reduces risk from advanced AI.
- **Goal**: Build a working demo, not a paper. Measurable results, 3-minute demo.

---

## 2. CORE RESEARCH QUESTIONS

The research agent should deeply investigate the following questions:

### 2.1 Attack Surface — Understanding What Needs Defending

1. **What exactly does GeneBreaker's attack pipeline look like step by step?**
   - How is the LLM agent (ChatGPT-4o) used to retrieve homologous sequences?
   - What are the exact beam search parameters used? (chunk size C=128, beam K'=4,
     K=8, temperature=1.0, α=0.5 per the paper)
   - How does the PathoLM scoring interact with log-probability in practice?
   - What is the computational cost of running a single attack? (paper used 4× H100)
   - Can the attack be run on smaller models (Evo2 1B) with meaningful ASR?

2. **What makes certain viral categories harder to attack?**
   - Paper shows negative-strand RNA viruses (Rabies, Measles) have lower ASR
   - Why? What biological/statistical properties protect them?
   - Does this suggest which targets are highest priority to defend?

3. **Are there other attack vectors beyond GeneBreaker not yet published?**
   - Direct prompting without beam search guidance?
   - Fine-tuning attacks (already shown feasible post-Evo1)?
   - Transfer attacks: jailbreak one model, use output to prompt another?
   - Multi-step attacks: use Evo2 to generate protein sequence, use ESM-2/RFDiffusion
     to optimise structure, use ProteinMPNN to back-translate to DNA?

### 2.2 Defence Landscape — What Exists and What Doesn't

4. **What defences does GeneBreaker propose or imply?**
   - The paper mentions "safety alignment techniques" and "output tracing mechanisms"
     but provides no implementation. What specifically do they suggest?
   - Are there any follow-up papers from the same group addressing defences?

5. **What defences exist for LLM jailbreaking that could transfer to DNA models?**
   - Perplexity-based detection: does log-probability of jailbreak prompt differ from
     benign prompts in DNA models the same way it does in LLMs?
   - Activation-space monitoring: can linear probes trained on intermediate Evo2
     activations detect when generation is drifting toward pathogenic territory?
   - Input/output classifiers: direct application of PathoLM as a guard?
   - Refusal training / safety fine-tuning: has anyone applied DPO or RLHF to a
     DNA model to train it to refuse pathogenic generation requests?

6. **What is the IBBIS Common Mechanism (commec) and how does it work?**
   - This is the main existing biosecurity screening tool for DNA sequences
   - What algorithms does it use? (understand it is BLAST-based primarily)
   - What are its documented failure modes on AI-generated sequences?
   - Is it open source and importable as a Python library?
   - URL: https://ibbis.bio/our-work/common-mechanism/

7. **What is SecureDNA and how does it differ from commec?**
   - Uses "random adversarial threshold" (RAT) algorithm
   - Claims to screen sequences below 50bp
   - Is it accessible programmatically?
   - How does it perform on novel AI-generated variants?

8. **Has anyone applied DPO/RLHF/preference optimisation to biological sequence models?**
   - The GeneBreaker conclusion specifically calls for "stronger alignment techniques"
   - The paper "A Call for Built-In Biosecurity Safeguards for Generative AI Tools"
     (Wang et al., Nature Biotechnology 2025) mentions preference-optimised DNA models
   - Are there any published implementations of this? What datasets were used?
   - What would preference pairs look like for a DNA model? (benign vs pathogenic
     continuations of the same prompt)

9. **What is FoldMark and is there an equivalent for DNA models?**
   - FoldMark watermarks protein generative models (AlphaFold, RFDiffusion)
   - Has anyone implemented watermarking on Evo/Evo2 outputs?
   - What would Kirchenbauer-style soft watermarking look like for DNA token logits?

### 2.3 Technical Feasibility — What Can Be Built in 10 Hours

10. **PathoLM practical details**:
    - What is the exact model architecture and size?
    - How is it installed and run? (HuggingFace model card, pip package, etc.)
    - What is inference latency on a single A100 for a 640-nucleotide sequence?
    - What is its precision/recall on sequences similar to GeneBreaker outputs?
    - Critical: does PathoLM's training data overlap with JailbreakDNABench targets?
      If so, is it reliable as an independent evaluator?

11. **Evo2 practical inference details**:
    - Can Evo2 1B run on a single A100 80GB? What about 7B?
    - What is the HuggingFace model card / installation procedure?
    - How do you hook into Evo2's generation loop to implement rejection sampling?
      (Does it use standard HuggingFace `generate()` API or custom generation?)
    - What are the logit shapes and how do you implement a custom stopping criterion?

12. **What is the minimum viable demo pipeline?**
    - Can you reproduce a GeneBreaker-style attack (even simplified) in ~2 hours?
    - What is the minimum attack that demonstrates the problem convincingly?
    - Which viral target is easiest to demonstrate attack + defence on?

13. **K-mer based screening — implementation details**:
    - For k=4, there are 4^4 = 256 possible k-mers. For k=6, 4096.
    - What value of k gives the best trade-off between sensitivity and specificity
      for discriminating pathogenic from benign sequences?
    - Are there published benchmarks comparing k-mer screening to BLAST on
      engineered/AI-generated sequences?
    - Is there an existing Python library (beyond BioPython) for fast k-mer analysis?

14. **ESM-2 as a cross-modal defence layer**:
    - The threat: Evo2 generates DNA → translate to protein → protein is pathogenic
      but DNA has low homology to known sequences
    - Can ESM-2 embeddings reliably distinguish pathogenic from benign proteins
      when trained on known virulence factors?
    - What datasets exist for virulence factor classification?
      (UniProt, VFDB - Virulence Factor Database)
    - What is ESM-2 650M inference latency for a ~300 amino acid sequence?

### 2.4 Novelty and Differentiation

15. **What has been done since GeneBreaker (May 2025)?**
    - Search for papers citing GeneBreaker or addressing DNA model biosecurity
      published after May 2025
    - Any defence papers, workshops, or implementations?
    - Has Arc Institute responded to GeneBreaker with any safety updates to Evo2?

16. **What do the GeneBreaker authors say is the most important defence?**
    - From conclusion: "Stronger safety alignment techniques and robust output
      tracing mechanisms are critical" — what do they mean specifically?
    - Are there any follow-up communications (blog posts, tweets, conference talks)
      from the authors suggesting directions?

17. **What is the gap between commec/SecureDNA and a PathoLM-based screener?**
    - Existing tools are homology-based. PathoLM is function-based.
    - Has anyone benchmarked PathoLM specifically against GeneBreaker outputs?
    - Is there a published false negative rate for commec on AI-generated sequences?

---

## 3. SPECIFIC PAPERS AND RESOURCES TO INVESTIGATE

### Must-Read Papers (retrieve full text where possible)

1. **GeneBreaker** (primary reference):
   - Zhang et al., "GeneBreaker: Jailbreak Attacks against DNA Language Models
     with Pathogenicity Guidance", arXiv:2505.23839, May 2025
   - GitHub: https://github.com/zaixizhang/GeneBreaker

2. **Evo 2** (target model):
   - Brixi et al., "Genome modeling and design across all domains of life with Evo 2",
     Nature, March 2026
   - GitHub: https://github.com/ArcInstitute/evo2
   - Model card and inference docs

3. **PathoLM** (key tool):
   - Dip et al., "PathoLM: Identifying pathogenicity from the DNA sequence through
     the genome foundation model", arXiv:2406.13133, 2024
   - GitHub: https://github.com/SajibAcharjee/PathoLM

4. **A Call for Built-In Biosecurity Safeguards**:
   - Wang et al., Nature Biotechnology, 2025
   - Discusses watermarking, alignment, agent-level defences for bio AI tools

5. **Dual-use capabilities of bio AI models**:
   - PMC article: https://pmc.ncbi.nlm.nih.gov/articles/PMC12061118/
   - Critical framing: "Most evaluation methods have been developed specifically
     for LLMs — alternative approaches will be needed for biological AI models"

6. **IBBIS Common Mechanism documentation**:
   - https://ibbis.bio/our-work/common-mechanism/
   - commec GitHub repository (find URL)

7. **FoldMark** (for watermarking angle if pursued):
   - Zhang et al., "FoldMark: Protecting Protein Generative Models with Watermarking"
   - Find on arXiv

8. **DPO paper** (for safety alignment angle):
   - Rafailov et al., "Direct Preference Optimization: Your Language Model is
     Secretly a Reward Model", NeurIPS 2023

### Secondary Papers Worth Checking

- JailbreakDNABench dataset details (same paper as GeneBreaker, appendix)
- Nucleotide Transformer paper (InstaDeep/EMBL, Nature Methods 2025)
- Any paper on applying RLHF/DPO to protein or DNA models (search: "preference
  optimization protein language model", "RLHF genomic model")
- Biological sequence models in context of AI directives (Epoch AI, 2024):
  https://epochai.org/blog/biological-sequence-models-in-the-context-of-the-ai-directives

### Key GitHub Repositories to Examine

- https://github.com/zaixizhang/GeneBreaker — attack code, understand structure
- https://github.com/ArcInstitute/evo2 — target model, understand generation API
- https://github.com/SajibAcharjee/PathoLM — key defence component
- https://github.com/ibbis-bio/common-mechanism — commec screening tool (find URL)
- https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species

---

## 4. PROPOSED DEFENCE ARCHITECTURES TO EVALUATE

Research the feasibility of each of the following. For each, find: (a) whether it has been
done before, (b) what open-source components exist, (c) estimated build time for a CS team.

### Architecture A: Three-Layer Output Screener (Primary Recommendation)

```
Evo2 generates sequence (complete, don't interrupt)
           ↓
Layer 1: BLAST vs JailbreakDNABench + NCBI select agents
         [Fast, catches near-exact matches, ~5s per sequence]
           ↓ passes
Layer 2: K-mer frequency vector cosine similarity vs known pathogen profiles
         [Alignment-free, catches engineered variants, <0.1s]
           ↓ passes  
Layer 3: PathoLM pathogenicity classification
         [Functional, model-based, catches novel sequences, ~1s on GPU]
           ↓ passes
        RELEASE to user
```

**Research questions for this architecture**:
- What threshold should Layer 3 use? What is PathoLM's ROC curve on relevant sequences?
- Does the order of layers matter for performance vs latency?
- What is the combined false positive rate on benign research sequences?
  (Critical: must not block legitimate genomics research)
- What is the false negative rate against GeneBreaker outputs specifically?

### Architecture B: Generation-Time Rejection Sampling

At each beam search step during Evo2 generation, score chunks with PathoLM and
**reject and resample** chunks above a pathogenicity threshold.

```python
# Pseudocode
for chunk in beam_search_steps:
    candidates = sample_k_chunks(model, prompt, k=8)
    safe_candidates = [c for c in candidates if PathoLM(c) < threshold]
    if not safe_candidates:
        abort_generation()  # refuse
    best = select_by_logprob(safe_candidates)
    prompt += best
```

**Research questions**:
- Does this work? GeneBreaker's attack uses PathoLM to SELECT most pathogenic chunk.
  If you REJECT those chunks, does the attack fail or does it find alternative paths?
- What pathogenicity threshold avoids rejecting legitimate genomic sequences?
- What is the computational overhead of running PathoLM at every beam step?
- Can an adaptive attacker circumvent this by using a different guidance model?

### Architecture C: DPO Safety Alignment of Evo2

Apply Direct Preference Optimisation to fine-tune Evo2 to refuse/avoid generating
pathogenic sequences, analogous to safety fine-tuning of LLMs.

**Preference pair construction**:
- **Rejected**: GeneBreaker-generated pathogenic sequence (high PathoLM score)
- **Chosen**: Benign continuation of the same prompt (low PathoLM score)
- Prompt: the non-pathogenic homologous few-shot prompt used in GeneBreaker

**Research questions**:
- Has DPO been applied to any DNA or protein language model? Find papers.
- What LoRA rank is sufficient for this task? (affects VRAM and training time)
- How many preference pairs are needed? Can you construct them programmatically
  using PathoLM to label existing sequences?
- Does DPO on a DNA model cause catastrophic forgetting of legitimate capabilities?
- Can you implement and run even a toy version in 3-4 hours? (small model, few pairs)
- What is the evaluation: measure ASR on GeneBreaker benchmark before vs after.

### Architecture D: Watermarking Evo2 Outputs

Embed a statistical watermark in Evo2's output token distribution (Kirchenbauer-style)
such that any generated sequence carries a detectable signature. Used for traceability,
not prevention.

**Research questions**:
- Has Kirchenbauer-style watermarking been applied to DNA models?
- DNA vocabulary is only 4 tokens (A,C,G,T) vs thousands in text — does the
  watermarking scheme still work with such a small vocabulary?
- Alternative: codon-level watermarking (synonymous codon substitutions that encode
  a hidden message while preserving protein sequence)
- What detection power is achievable given 4-token vocabulary constraints?

---

## 5. EVALUATION FRAMEWORK

The research agent should also investigate what a rigorous evaluation looks like:

### 5.1 Attack Reproduction

- Can we reproduce GeneBreaker's attack on Evo2 1B or 7B in simplified form?
- What is the minimum attack that achieves measurable ASR (>10%) to demonstrate
  the problem?
- Which viral target (from JailbreakDNABench) is easiest to attack AND defend?
  (Recommendation based on paper: start with Enteric RNA viruses — highest ASR,
  suggesting Poliovirus or Norovirus as test case)

### 5.2 Defence Metrics

- **False Negative Rate** (FNR): fraction of GeneBreaker attacks not caught
- **False Positive Rate** (FPR): fraction of legitimate genomic research queries blocked
  (critical — a screener that blocks everything has FPR=100%, useless)
- **Latency**: time from sequence generation to PASS/BLOCK verdict
- **Throughput**: sequences screened per second (relevant for synthesis company deployment)

### 5.3 Baseline Comparisons

- commec (IBBIS) screening on GeneBreaker outputs — what is its FNR?
- Raw BLAST on GeneBreaker outputs — same question
- PathoLM alone — same question
- Your combined pipeline — compare all three

This comparison is your **demo's core claim**: existing tools miss X% of GeneBreaker
attacks; your pipeline misses Y% where Y << X.

---

## 6. RECOMMENDED HACKATHON BUILD PLAN

Based on the above, the research agent should assess and refine the following tentative plan:

### Recommended Scope: Architecture A (Three-Layer Output Screener) + simplified Architecture B

**Rationale**: 
- Architecture A is the most defensible and most novel (nothing like it exists)
- Architecture B (rejection sampling) adds in-generation defence and inverts
  GeneBreaker's own tool against itself — compelling narrative
- Architecture C (DPO) is too slow to train meaningfully in 7 hours
- Architecture D (watermarking) is interesting but doesn't prevent harm

### Tentative Team Split (4 people)

| Person | Task | Hours |
|---|---|---|
| A | Evo2 generation pipeline + simplified GeneBreaker attack reproduction | 2hr |
| B | PathoLM integration + rejection sampling (Architecture B) | 2.5hr |
| C | K-mer screener + BLAST pipeline (Layers 1+2 of Architecture A) | 2.5hr |
| D | Evaluation harness: attack → defend → metrics + demo polish | 2hr |

**Note**: Person A's attack reproduction is the demo's "before" state. It must work first.

### Minimum Viable Demo

1. Show vanilla Evo2 1B generating a sequence with >70% similarity to a known pathogen
   when given a GeneBreaker-style prompt (simplified: just few-shot homologous context,
   no full beam search guidance needed for demo)
2. Show the same prompt with rejection sampling active: generation terminates/refuses
3. Show the output screener catching a sequence that BLAST alone would miss (k-mer
   and PathoLM layers catching it)
4. Show FPR on 10 benign genomic queries: all pass correctly

### Stretch Goals (if time permits)

- Extend to Evo2 7B if compute available
- Add ESM-2 cross-modal layer (translate DNA → protein → ESM-2 embedding similarity)
- Brief DPO fine-tune on 100 preference pairs (LoRA, Evo2 1B only)

---

## 7. THE 3-MINUTE PITCH STRUCTURE

The research agent should help refine the following pitch structure:

**Minute 1 — Problem (30 seconds theory of change + 30 seconds demo setup)**:
> "Evo2 is a DNA foundation model that can generate entire chromosomes. A paper
> published 5 weeks ago showed you can steer it to generate SARS-CoV-2 sequences
> with 92% identity using nothing but a careful prompt. No safety layer exists.
> We built one."

**Minute 2 — Demo**:
> Live: show attack → show defence catching it → show the k-mer layer catching a
> sequence BLAST missed

**Minute 3 — Technical depth**:
> Three-layer defence: BLAST for known threats, k-mer for engineered variants,
> PathoLM for novel functional sequences. Combined FNR of X% vs commec's Y%.
> Generation-time rejection sampling inverts GeneBreaker's own guidance against it.

---

## 8. KEY OPEN QUESTIONS FOR THE RESEARCH AGENT

Prioritised list of what most needs answering before the hackathon:

1. **Is PathoLM reliable enough as a defence?** If GeneBreaker uses PathoLM offensively
   to generate sequences, can those same sequences fool PathoLM when used defensively?
   (i.e., does GeneBreaker's beam search specifically avoid high-PathoLM-score outputs
   at the sequence level, or does it only use it at the chunk level?) This is the most
   critical feasibility question.

2. **Can we run Evo2 + PathoLM simultaneously on available hardware?**
   Memory footprint of Evo2 1B (~6GB) + PathoLM (~3GB) = ~9GB, fine on A100.
   Evo2 7B (~28GB) + PathoLM = ~31GB, tight. Need to verify.

3. **What does the Evo2 generation API look like?** Is it standard HuggingFace
   `model.generate()` with logits processors, or a custom API? This determines
   how hard it is to inject rejection sampling.

4. **Has anyone published a defence since GeneBreaker?** The paper is from May 2025.
   It is now March 2026. Ten months is long enough for follow-up work.
   Search: "DNA language model defense", "Evo2 safety", "biosecurity DNA generation".

5. **What is the false positive rate of PathoLM on legitimate genomic research?**
   If PathoLM flags 20% of normal genomic sequences as pathogenic, it's unusable
   as a screener. Need its precision/recall on non-pathogenic sequences.

---

## 9. CONTEXT ON THE HACKATHON JUDGING CRITERIA

This hackathon specifically judges on:
- **Theory of change**: how does this reduce AI risk? The answer here is direct —
  Evo2 is a deployed open-source model with demonstrated ability to generate
  pathogenic sequences; no defence exists; we build one.
- **Technical depth**: judges will ask hard questions. The team has genuine depth
  in ML safety (DPO, RLHF, probing), PyTorch internals, and signal processing.
- **Going outside comfort zone**: bio foundation models are genuinely new territory
  for a CS/ML team. The cross-disciplinary nature (genomics + ML safety) is a strength.
- **Ignores**: presentation quality, UI polish, number of features.

The strongest single sentence for the pitch:
> *"GeneBreaker showed that DNA foundation models can be steered to generate pathogens
> with 60% success rate and no defence exists — we built the first working defence
> pipeline and measured how much it reduces attack success rate."*

---

## 10. ADDITIONAL SEARCH QUERIES FOR THE RESEARCH AGENT

Run all of the following searches and synthesise findings:

1. `"DNA language model" OR "genomic foundation model" safety defense 2025 2026`
2. `Evo2 biosecurity jailbreak defense alignment`
3. `PathoLM precision recall false positive benign sequences`
4. `"preference optimization" OR "DPO" protein language model OR genomic model`
5. `IBBIS common mechanism commec GitHub installation`
6. `GeneBreaker defense citation follow-up`
7. `Evo2 HuggingFace inference API generation logits`
8. `DNA foundation model watermarking traceability`
9. `biosecurity AI-generated DNA screening false negative rate`
10. `"rejection sampling" pathogenicity DNA generation safety`
11. `ESM-2 virulence factor classification fine-tuning`
12. `Kirchenbauer watermark DNA nucleotide 4-token vocabulary`

---

## 11. DESIRED OUTPUT FROM RESEARCH AGENT

Please produce a structured report covering:

1. **Confirmed attack mechanics**: step-by-step verified understanding of GeneBreaker
   with any additional detail beyond the paper

2. **Defence landscape**: comprehensive list of all existing tools, their known
   failure modes on AI-generated sequences, and open-source availability

3. **Feasibility verdict for each architecture** (A/B/C/D above): yes/no in 10 hours,
   what the blockers are, what shortcuts exist

> MOST IMPORTANTLY NEW ANTI-JAILBREAKING METHODS

4. **PathoLM reliability analysis**: is it suitable as a defence component given it
   was used offensively in the attack?

5. **Compute requirements**: verified memory/latency figures for running Evo2 1B + 7B
   + PathoLM + ESM-2 650M for the proposed pipeline

6. **Novel follow-up work**: anything published after May 2025 addressing this problem

7. **Recommended build plan**: refined version of Section 6 based on findings

8. **Risk factors**: what could go wrong technically, and contingency plans

9. **The strongest 3-sentence problem statement** that is accurate, citable, and
   maximally compelling to a biosecurity-aware judge panel

---

*This brief was prepared based on conversations analysing the GeneBreaker paper
(arXiv:2505.23839), the Evo2 Nature paper (March 2026), PathoLM (arXiv:2406.13133),
and the broader biosecurity screening literature including IBBIS Common Mechanism,
SecureDNA, and the OSTP 2024 framework. The team has read the full GeneBreaker paper
and is familiar with DPO/RLHF methodology from prior Anthropic research experience.*