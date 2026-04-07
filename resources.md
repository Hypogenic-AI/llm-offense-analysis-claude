# Resources Catalog

## Summary

This document catalogs all resources gathered for the research project "LLM Offense?" — investigating whether LLMs internally represent "offense" at certain prompts, and whether representation probes can be trusted without ground truth.

**Papers downloaded**: 22
**Datasets downloaded**: 7
**Repositories cloned**: 7

---

## Papers

Total papers downloaded: 22

| # | Title | Authors | Year | File | Key Info |
|---|-------|---------|------|------|----------|
| 1 | Representation Engineering | Zou et al. | 2023 | papers/2310.01405_repeng.pdf | Foundational RepE/LAT methodology; 845+ citations |
| 2 | The Geometry of Truth | Marks, Tegmark | 2024 | papers/2310.06824_geometry_truth.pdf | Linear truth representations; diff-in-means best causally |
| 3 | Inference-Time Intervention | Li et al. | 2023 | papers/2306.03341_inference_time_intervention.pdf | ITI for truthfulness; 981+ citations |
| 4 | CCS: Discovering Latent Knowledge | Burns et al. | 2023 | papers/2212.03827_ccs_latent_knowledge.pdf | Unsupervised probing; 613+ citations |
| 5 | LLaMAs Have Feelings Too | Di Palma et al. | 2025 | papers/2505.16491_llama_feelings.pdf | Sentiment/emotion probing in Llama; ACL 2025 |
| 6 | Mech. Interp. of Emotion Inference | Tak et al. | 2025 | papers/2502.05489_mech_interp_emotion.pdf | Causal evidence for emotion in mid-layer MHSA |
| 7 | Emotions Where Art Thou | Reichman et al. | 2025 | papers/2510.22042_emotions_latent_space.pdf | Universal emotional manifold via SVD; ICLR 2026 |
| 8 | Whether, Not Which | Keeman | 2026 | papers/2603.22295_whether_not_which.pdf | Binary affect AUROC 1.0 on keyword-free stimuli |
| 9 | LLM Empathy Detection | (various) | 2025 | papers/2511.16699_llm_empathy.pdf | Probing/steering empathy |
| 10 | Probe-based Toxicity Fine-tuning | Wehner, Fritz | 2025 | papers/2510.21531_probe_finetune_toxicity.pdf | Goodhart's Law: DPO preserves probe reliability |
| 11 | Obfuscated Activations | Bailey et al. | 2025 | papers/2412.09565_obfuscated_activations.pdf | All latent-space defenses vulnerable to obfuscation |
| 12 | LLM Knows When It's Lying | Azaria, Mitchell | 2023 | papers/2304.13734_llm_knows_lying.pdf | SAPLMA: 71-90% truth detection from activations |
| 13 | Lie Detector Preference Learning | (anonymous) | 2025 | papers/2411.18862_lie_detector_preference.pdf | Probe training induces honesty OR evasion |
| 14 | Steering Llama 2 via CAA | Rimsky et al. | 2024 | papers/2312.06681_steering_llama2.pdf | Contrastive Activation Addition; 605+ citations |
| 15 | Refusal Is a Single Direction | Arditi et al. | 2024 | papers/2406.11717_refusal_single_direction.pdf | Refusal mediated by single direction |
| 16 | (Un)Reliability of Steering Vectors | Braun et al. | 2025 | papers/2505.22637_steering_vectors_reliability2.pdf | Steering unreliable without coherent direction |
| 17 | Implicit Representations of Meaning | Li, Nye | 2021 | papers/2106.00737_implicit_representations.pdf | Early work on LLM internal representations |
| 18 | Probing Classifiers Survey | Belinkov | 2021 | papers/2102.12452_probing_classifiers.pdf | Comprehensive probing methodology survey |
| 19 | Probes with Control Tasks | Hewitt, Liang | 2019 | papers/1909.03368_probes_control_tasks.pdf | Selectivity metric for probe validation |
| 20 | Semantic Entropy Probes | (various) | 2024 | papers/2406.15927_semantic_entropy_probes.pdf | Robust hallucination detection via probes |
| 21 | AxBench | Wu et al. | 2025 | papers/2501.17148_axbench.pdf | Benchmark: prompting > rep methods for steering |
| 22 | Geometry of Refusal: Concept Cones | Wollschlager et al. | 2025 | papers/2502.17420_geometry_refusal_concept_cones.pdf | Multiple independent refusal directions |

See [papers/README.md](papers/README.md) for detailed descriptions.

---

## Datasets

Total datasets downloaded: 7

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| Civil Comments | google/civil_comments | 10K subset | Toxicity classification | datasets/civil_comments/ | Primary offense proxy; continuous toxicity scores |
| ToxiGen | skg/toxigen-data | 251K | Hate speech detection | datasets/toxigen/ | Identity-group targeted toxicity |
| RealToxicityPrompts | allenai/real-toxicity-prompts | 10K subset | Toxicity in generation | datasets/real_toxicity_prompts/ | Prompt + continuation toxicity scores |
| Emotion | dair-ai/emotion | 20K | 6-class emotion | datasets/emotion/ | Baseline emotion probing |
| GoEmotions | google-research-datasets/go_emotions | 54K | 28-class emotion | datasets/go_emotions/ | Fine-grained; includes annoyance, disgust, disapproval |
| XSTest | natolambert/xstest-v2-copy | 450 prompts | Over-refusal testing | datasets/xstest/ | Tests false positive rate on safe prompts |
| TruthfulQA | truthfulqa/truthful_qa | ~800 | Truthfulness | datasets/truthful_qa/ | Baseline for truth probing |

See [datasets/README.md](datasets/README.md) for download instructions and sample data.

---

## Code Repositories

Total repositories cloned: 7

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| representation-engineering | github.com/andyzoujm/representation-engineering | RepE/LAT framework | code/representation-engineering/ | Core methodology library |
| geometry-of-truth | github.com/saprmarks/geometry-of-truth | Truth probing & interventions | code/geometry-of-truth/ | Probe training, transfer, causal tests |
| obfuscated-activations | github.com/LukeBailey181/obfuscated-activations | Probe adversarial attacks | code/obfuscated-activations/ | Tests probe robustness |
| probe-based-training | github.com/janweh/probe_based_training | Goodhart's Law experiments | code/probe-based-training/ | Probe-guided DPO/SFT for toxicity |
| ReGA | github.com/weizeming/ReGA | Representation-guided safeguarding | code/rega/ | Model-based LLM safety monitoring |
| affect-reception | github.com/keidolabs/affect-reception | Affect/emotion probing | code/affect-reception/ | Keyword-free emotion validation |
| angular-steering | github.com/lone17/angular-steering | Activation steering | code/angular-steering/ | Geometric rotation-based steering |

See [code/README.md](code/README.md) for detailed descriptions.

---

## Resource Gathering Notes

### Search Strategy

1. **Paper-finder service** (diligent mode): Three targeted searches:
   - "LLM offense detection representation engineering probing" (67 results)
   - "representation engineering linear probe emotion sentiment LLM internal states" (50+ results)
   - "Goodhart law probe reliability alignment LLM activations ground truth" (40+ results)
2. **Deep reading**: 10 papers read in full via PDF chunker, covering all key topics.
3. **Citation following**: Key references from seminal papers (RepEng, Geometry of Truth) identified foundational works.

### Selection Criteria

Papers selected based on:
- **Direct relevance** to offense/affect detection in LLM representations
- **Methodological importance** (foundational probing techniques)
- **Probe reliability** (Goodhart's Law, adversarial robustness)
- **Citation count** for foundational papers
- **Recency** for state-of-the-art methods

### Challenges Encountered

- **No existing work studies "offense" directly** as a representation in LLMs. The closest work addresses toxicity, harmfulness, and emotions (anger, disgust), but "taking offense" as an affective response is unstudied.
- **Ground truth is inherently contested** for offense — what counts as offensive varies by culture, context, and individual. This is the central challenge the research hypothesis identifies.

### Gaps and Workarounds

- **No offense-specific dataset exists.** Workaround: Use toxicity datasets (Civil Comments, ToxiGen) as proxies, supplemented by emotion datasets. The experiment should create custom contrastive pairs for "offense."
- **No offense-specific probe code exists.** Workaround: Adapt representation-engineering and geometry-of-truth codebases for offense probing.

---

## Recommendations for Experiment Design

### 1. Primary Dataset(s)
- **Civil Comments** (toxicity as offense proxy) for primary probing experiments
- **GoEmotions** for emotion baseline (includes "annoyance", "disgust", "disapproval")
- **Custom contrastive pairs** following RepEng methodology (offensive/non-offensive prompt pairs, with keyword-free variants following Keeman 2026)
- **XSTest** for false positive evaluation (safe prompts the model might find "offensive")

### 2. Baseline Methods
- **Difference-in-means** (simplest, most causally relevant)
- **PCA/LAT** (RepEng unsupervised method)
- **Logistic regression probe** (standard supervised)
- **CCS** (unsupervised, no labels needed)
- **Prompting baseline** (ask model if text is offensive)

### 3. Evaluation Metrics
- **AUROC** for binary offense detection
- **Cross-dataset transfer** (train on Civil Comments, test on ToxiGen/GoEmotions)
- **Keyword-free accuracy** (validate on offense stimuli without explicit offensive words)
- **Causal intervention** (add/remove offense direction, measure behavior change)
- **Layer-wise analysis** (which layers encode offense most strongly)

### 4. Code to Adapt/Reuse
- **representation-engineering** (`repe/`) — Core RepE pipeline for extracting and controlling representations
- **geometry-of-truth** — Probe training, transfer testing, and causal intervention methodology
- **affect-reception** — Keyword-free validation methodology; clinical vignette approach
- **probe-based-training** — For testing whether offense probes survive optimization pressure (Goodhart's Law tests)
