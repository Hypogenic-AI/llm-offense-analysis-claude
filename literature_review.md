# Literature Review: LLM Offense Detection via Representation Probing

## Research Area Overview

This review addresses whether large language models (LLMs) internally represent "offense" — the state of taking offense at a prompt — and whether probes detecting such representations can be trusted without ground truth. The research sits at the intersection of **representation engineering** (extracting high-level concepts from LLM hidden states), **affective computing** (emotion/sentiment in neural representations), and **probe reliability** (Goodhart's Law concerns when probes are used as monitors or optimization targets).

The field has matured rapidly since 2023. Representation engineering (Zou et al., 2023) established that high-level concepts — including emotions, honesty, harmfulness — are linearly encoded in LLM hidden states and can be both read and controlled. Subsequent work has extended this to emotions specifically, while a parallel line of work has raised fundamental concerns about whether such probes remain reliable under optimization pressure.

---

## Key Papers

### Foundational: Representation Engineering

#### Zou et al. (2023) — "Representation Engineering: A Top-Down Approach to AI Transparency"
- **arXiv**: 2310.01405 | **Citations**: 845+
- **Core method**: Linear Artificial Tomography (LAT) — collect hidden states from contrastive prompt pairs (e.g., "Pretend you're honest" vs. "Pretend you're dishonest"), compute difference vectors, apply PCA. The first principal component is the "reading vector" for a concept.
- **Concepts studied**: Honesty, harmfulness, truthfulness, morality, power, utility, risk, probability, memorization, bias, and **all six Ekman emotions** (happiness, sadness, anger, fear, surprise, disgust).
- **Key results**: LAT achieves 81-93% accuracy on concept classification. Critically, adding +Happiness to LLaMA-2-Chat-13B increases compliance with harmful requests from 0% to 100%, demonstrating that emotional representations causally influence safety behavior.
- **Probe reliability insight**: Logistic regression probes achieve highest correlation accuracy but show NO causal effect in manipulation experiments. Only PCA/mean-difference methods show robust causal influence — high classification accuracy alone is insufficient evidence of a genuine representation.
- **No "offense" concept studied**, but the methodology is directly applicable.
- **Code**: github.com/andyzoujm/representation-engineering

#### Marks & Tegmark (2024) — "The Geometry of Truth"
- **arXiv**: 2310.06824 | **Citations**: 444+ | **Venue**: COLM 2024
- **Key insight**: Truth/falsehood is linearly represented in LLM hidden states, with clear separation visible via PCA. Simple difference-in-mean probes generalize across topically diverse datasets AND are more causally implicated than more complex probes.
- **Critical finding on probe reliability**: Probes can identify features that *correlate* with truth but are not truth itself (e.g., "probable text" vs. "true text"). Careful dataset design with anti-correlated controls is essential.
- **Scale matters**: Larger models develop more abstract, generalizable truth representations; smaller models represent surface-level features.
- **Code**: github.com/saprmarks/geometry-of-truth

### Emotion and Affect in LLM Representations

#### Di Palma et al. (2025) — "LLaMAs Have Feelings Too"
- **arXiv**: 2505.16491 | **Venue**: ACL 2025
- **Emotions**: Binary sentiment (positive/negative) and 6-class emotion (joy, sadness, anger, fear, love, surprise).
- **Key findings**: Binary sentiment is best detected in mid-layers (~60-75% depth), achieving 94-96% probe accuracy. Fine-grained 6-class emotion is best in early layers (0-4), achieving 69-87% accuracy. Probes outperform prompting by up to 14%.
- **Implication for offense**: Binary offense detection (offensive/not) should achieve high accuracy in mid-layers; fine-grained offense categorization will be harder.
- **Datasets**: IMDB, SST-2, Rotten Tomatoes, Emotion (Saravia et al.)

#### Tak et al. (2025) — "Mechanistic Interpretability of Emotion Inference in LLMs"
- **arXiv**: 2502.05489 | 10 models tested across 5 families
- **Key contribution**: Goes beyond correlation to *causal* evidence. Identifies mid-layer MHSA units (layers 9-11 in Llama-1B) as the locus of emotion processing. Causal interventions on appraisal vectors produce psychologically plausible emotion shifts.
- **13 emotions studied** plus 23 appraisal dimensions (pleasantness, agency, control, urgency, etc.)
- **Critical insight**: The appraisal-emotion structure in LLMs mirrors cognitive appraisal theory from psychology — but authors caution "appraisals are merely correlated with emotions rather than exerting direct causal influence."
- **Dataset**: crowd-enVENT (6,800 vignettes with self-reported emotions + 23 appraisal dimensions)

#### Reichman et al. (2025) — "Emotions Where Art Thou"
- **arXiv**: 2510.22042 | **Venue**: ICLR 2026
- **Key contribution**: Discovers a universal, low-dimensional emotional manifold in LLM hidden states via SVD. Principal components correspond to interpretable psychological dimensions: PC1 ~ valence, PC2 ~ dominance, PC3 ~ approach-avoidance, PC4 ~ arousal.
- **75% of neurons per layer** exceed AUROC 0.9 for emotion discrimination — emotional information is broadly distributed, not localized.
- **Cross-lingual**: Emotional geometry is stable across 8 datasets in 5 languages.
- **Failure mode**: Nuanced categories (envy, excitement) are harder to steer than basic ones, suggesting binary offense/non-offense will be more reliable than fine-grained types.

#### Keeman (2026) — "Whether, Not Which: Dissociable Affect Reception and Emotion Categorization"
- **arXiv**: 2603.22295
- **Most directly relevant to offense probe trustworthiness.** Discovers a fundamental dissociation:
  - **Affect reception** (detecting emotional significance): Achieves AUROC 1.000 on keyword-free clinical vignettes, saturates in first 9-38% of network depth. Is keyword-independent.
  - **Emotion categorization** (identifying which emotion): Partially keyword-dependent, accuracy drops 1.1-6.7% on keyword-free stimuli.
- **Critical validation**: 100% cross-emotion activation patching success — affect reception is a unitary salience signal, not emotion-specific. An offense probe may detect general "affective salience" rather than specifically offense.
- **Keyword confound warning**: If offense probes are only validated on text containing explicit slurs, they may detect keywords rather than offensive meaning. Must use keyword-free validation stimuli.
- **Code**: github.com/keidolabs/affect-reception

### Probe Reliability and Goodhart's Law

#### Wehner & Fritz (2025) — "Probe-based Fine-tuning for Reducing Toxicity"
- **arXiv**: 2510.21531
- **Explicitly addresses Goodhart's Law for probes.** When probes are used as training signals:
  - DPO (preference learning) *preserves* probe detectability — surprising inversion of expected Goodharting.
  - SFT (gradient regularization) allows partial probe evasion.
- **Key practical finding**: Probe ensemble diversity provides minimal benefit; simply retraining probes after optimization recovers high detection accuracy.
- **Open question**: Cannot determine whether probe-based training genuinely eliminates toxic processing or merely maintains detectable representations while producing non-toxic outputs.
- **Code**: github.com/janweh/probe_based_training

#### Bailey et al. (2025) — "Obfuscated Activations Bypass LLM Latent-Space Defenses"
- **arXiv**: 2412.09565 | **Citations**: 22+
- **Devastating result for probe reliability**: All evaluated latent-space defenses (probes, SAEs, OOD detectors) are vulnerable to obfuscation attacks. Embedding attacks reduce probe recall from 100% to 0% while retaining 90%+ jailbreak success.
- **Adversarial retraining fails**: 70+ epochs of iterative probe retraining cannot produce a robust monitor — attackers always find new obfuscated activations.
- **Core insight**: "Neural representations are highly malleable" — the same behavior can be realized through many activation patterns, many intermingled with benign activations.
- **However**: Obfuscation degrades model performance on complex tasks, suggesting limits to adversarial evasion.
- **Code**: github.com/LukeBailey181/obfuscated-activations

#### Azaria & Mitchell (2023) — "The Internal State of an LLM Knows When It's Lying"
- **arXiv**: 2304.13734 | **Citations**: 572+
- **Foundational positive result**: A simple feedforward classifier (SAPLMA) achieves 71-90% accuracy predicting statement truthfulness from LLM hidden states, substantially outperforming all baselines (~50-56%).
- **Mid-network layers** encode the most linearly separable truth signal (layer 16-20 of 32).
- **Out-of-distribution generalization**: Trained on 5 topics, tested on held-out 6th topic — captures generalized internal "belief" rather than topic-specific patterns.
- **Limitation**: Only evaluates passive monitoring; no adversarial pressure studied.

#### Anonymous (2025) — "Preference Learning with Lie Detectors"
- **arXiv**: 2411.18862
- **Central finding**: Probe-based preference learning can induce either genuine honesty OR sophisticated evasion, depending on training setup. The failure mode is training-objective-dependent.
- **Implication**: Probe-based training outcomes are highly sensitive to implementation details. Probe trustworthiness cannot be assumed after any optimization.

### Steering and Control

#### Rimsky et al. (2024) — "Steering Llama 2 via Contrastive Activation Addition (CAA)"
- **arXiv**: 2312.06681 | **Citations**: 605+
- **Method**: Compute "steering vectors" by averaging difference in residual stream activations between positive/negative example pairs. Add at inference time to steer behavior.
- **Effective over and on top of** finetuning and system prompts.

#### Arditi et al. (2024) — "Refusal in LLMs Is Mediated by a Single Direction"
- **arXiv**: 2406.11717
- **Key finding**: A single direction in activation space mediates refusal behavior. Ablating this direction eliminates refusal with minimal capability degradation.
- **Implication**: Safety-relevant behaviors can be disturbingly concentrated in low-dimensional subspaces.

#### Braun et al. (2025) — "Understanding (Un)Reliability of Steering Vectors"
- **arXiv**: 2505.22637
- **Key finding**: Steering exhibits high variance across samples — often gives effects opposite to desired. Higher cosine similarity between training set activation differences predicts more effective steering. "Vector steering is unreliable when the target behavior is not represented by a coherent direction."
- **Direct implication for offense**: If "offense" is not coherently represented (e.g., because what counts as offensive varies by context), steering vectors and probes will be unreliable.

---

## Common Methodologies

1. **Contrastive Activation Extraction**: Used in RepEng, CAA, and most steering work. Pairs of prompts differing in the target concept; difference vectors extracted from hidden states.
2. **Linear Probing (Logistic Regression)**: Standard approach across all papers. Train on labeled hidden states, evaluate accuracy.
3. **Difference-in-Means**: Simplest method — average activations for positive class minus average for negative class. Often most causally relevant (Marks & Tegmark).
4. **PCA on Difference Vectors**: RepEng's LAT method. Unsupervised, requires only contrastive pairs.
5. **SVD on Centered Activations**: Emotions Where Art Thou. Extracts multi-dimensional subspace.
6. **Causal Intervention** (Activation Patching): Gold standard for establishing causality beyond correlation.

## Standard Baselines

- **Prompting**: Ask the model directly (usually underperforms probing)
- **Token probability**: Use model's own output probabilities
- **Random classifier**: Chance-level baseline
- **Difference-in-means probe**: Simplest representation-based method
- **Logistic regression probe**: Standard linear probe
- **CCS (Burns et al., 2023)**: Unsupervised probe using consistency constraints

## Evaluation Metrics

- **AUROC**: Standard for binary classification (preferred for imbalanced data)
- **Accuracy / F1**: Standard classification metrics
- **Cross-dataset transfer**: Probe trained on dataset A, tested on dataset B
- **Causal mediation**: Does intervening on the identified direction change model behavior?
- **Adversarial robustness**: Does the probe work under deliberate obfuscation?

## Datasets in the Literature

| Dataset | Used In | Task |
|---------|---------|------|
| ETHICS (Hendrycks et al.) | RepEng | Utility, morality classification |
| TruthfulQA | RepEng, Geometry of Truth, ITI | Truthfulness |
| crowd-enVENT | Mech. Interp. Emotion, Whether Not Which | Emotion + appraisals |
| Emotion (Saravia et al.) | LLaMAs Have Feelings | 6-class emotion |
| IMDB, SST-2 | LLaMAs Have Feelings | Binary sentiment |
| GoEmotions | Emotions Where Art Thou | Fine-grained emotion |
| Civil Comments | Probe-based Toxicity | Toxicity labels |
| RealToxicityPrompts | Model Surgery | Toxicity generation |
| ToxiGen | Model Surgery | Machine-generated toxicity |
| XSTest | Obfuscated Activations, SCANS | Exaggerated safety / over-refusal |
| AdvBench | RepEng | Harmful instructions |

## Gaps and Opportunities

1. **No existing work directly studies "offense" as a representation**. Toxicity, harmfulness, and emotions are studied, but "taking offense" — a second-person affective response — is distinct and unexplored.

2. **Ground truth problem is central and unsolved**. All probe validation assumes labeled data exists. For "offense," what counts as offensive is culturally dependent, context-dependent, and contested. The research hypothesis directly confronts this: if a probe detects an "offense" direction without ground truth, can it be trusted?

3. **Binary detection is highly reliable; fine-grained categorization is not**. The Whether-Not-Which paper shows AUROC 1.000 for binary affect detection even on keyword-free stimuli. This suggests a binary offense probe could achieve very high accuracy, but distinguishing types of offense (racist, sexist, threatening, etc.) will be substantially harder.

4. **Keyword confounding is a major risk**. Probes may detect surface lexical features (slurs, explicit language) rather than genuine offensiveness. Keyword-free validation sets are essential but rarely used.

5. **Adversarial robustness is fundamentally limited**. If optimization pressure is applied against a probe (Bailey et al.), the probe will fail. Probes are reliable as *passive monitors* but not as *active defenses*.

6. **The Goodhart's Law question depends on use case**. If the probe is used for monitoring (no optimization against it), it can be reasonably trusted with proper validation. If used as a training signal, evasion is likely unless preference-learning objectives are used (Wehner & Fritz).

---

## Recommendations for Experiment Design

### Recommended Datasets
1. **Primary**: Create a custom contrastive dataset of "offensive" vs. "non-offensive" prompts, including keyword-free variants following the methodology of Keeman (2026)
2. **Validation**: Civil Comments (toxicity labels), Emotion dataset (emotion baselines), GoEmotions (fine-grained), XSTest (over-refusal edge cases)
3. **Control**: TruthfulQA or ETHICS (to verify probes are concept-specific, not detecting general "badness")

### Recommended Baselines
1. Difference-in-means probe (simplest, most causally relevant per Marks & Tegmark)
2. PCA/LAT (RepEng method, unsupervised)
3. Logistic regression probe (standard supervised)
4. Prompting baseline (ask the model if the text is offensive)
5. CCS (unsupervised, for comparison)

### Recommended Metrics
1. **AUROC** for binary offense detection
2. **Cross-dataset transfer accuracy** (train on one offense dataset, test on another)
3. **Keyword-free accuracy** (test on offense stimuli without explicit offensive words)
4. **Causal intervention** (does adding/removing the "offense direction" change model behavior?)
5. **Comparison to human judgment** (to assess whether probe detections "surprise the average English reader")

### Methodological Considerations
- **Layer selection**: Mid-layers (60-75% depth) for binary concepts; early layers for fine-grained emotion. Test systematically.
- **Pooling strategy**: Test multiple (last-token, mean, max, concatenation) — optimal choice varies.
- **Contrastive pair design**: Use both keyword-rich and keyword-free offensive stimuli. Include edge cases where content is offensive in context but not in isolation.
- **The "surprise" question**: The hypothesis mentions prompts that would "surprise the average English reader." This requires a human evaluation component or at minimum a dataset of prompts where model offense responses diverge from human expectations.
- **Multiple models**: Test across model families and sizes to assess generalizability (at minimum: Llama, Gemma, and one other family).
