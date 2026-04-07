# Research Plan: LLM Offense Detection and Probe Trustworthiness

## Motivation & Novelty Assessment

### Why This Research Matters
LLMs are increasingly deployed in sensitive applications where understanding their affective responses is critical. While prior work has studied toxicity detection and emotion in LLM representations, no work has studied "taking offense" — a second-person affective response where the model reacts as if offended. If LLMs can simulate offense, and they do so at unexpected prompts, this has implications for AI safety, alignment, and our understanding of LLM behavior. The trustworthiness question is equally important: if we build probes to detect internal offense states without ground truth, we need to know when to trust them.

### Gap in Existing Work
The literature review reveals:
1. **No existing work studies "offense" as an LLM representation.** Toxicity (content being harmful) and emotions (anger, disgust) are studied, but "taking offense" — the model's affective reaction — is distinct and unexplored.
2. **The ground truth problem is central and unsolved.** All probe validation assumes labeled data exists. For offense, what counts depends on context, culture, and framing.
3. **Surprise detection is novel.** No prior work systematically identifies cases where LLM offense responses diverge from human expectations.

### Our Novel Contribution
We conduct the first systematic study of "offense" as a behavioral and representational phenomenon in LLMs, combining:
1. **Behavioral experiments** using real LLM APIs to detect offense-like responses
2. **Representation probing** on open-source models to find an internal "offense direction"
3. **Surprise analysis** identifying prompts where LLM offense diverges from human expectation
4. **Probe trustworthiness assessment** without ground truth, using cross-validation, keyword-free testing, and convergent validity

### Experiment Justification
- **Experiment 1 (Behavioral Offense Survey):** Needed to establish whether LLMs behaviorally exhibit offense-like responses at all, and to create a behavioral ground truth for probe validation.
- **Experiment 2 (Representation Probing):** Tests whether offense is linearly encoded in hidden states, following RepEng methodology. Compares supervised and unsupervised probes.
- **Experiment 3 (Surprise Detection):** The core novel contribution — identifies cases where LLM offense diverges from human expectation, directly testing the hypothesis.
- **Experiment 4 (Probe Trustworthiness):** Assesses whether probes can be trusted via convergent validity (do different methods agree?), keyword-free validation, and cross-dataset transfer.

## Research Question
Can LLMs exhibit or simulate "taking offense" at prompts, and if so: (a) do they take offense at things that would surprise human readers? (b) can probes detecting an "offense direction" in representations be trusted without ground truth?

## Hypothesis Decomposition
- **H1:** LLMs will produce behavioral responses indicative of offense when prompted with certain inputs.
- **H2:** An "offense direction" exists in LLM hidden state space, detectable via linear probes.
- **H3:** Some prompts that trigger LLM offense will be surprising to human readers (i.e., not obviously offensive).
- **H4:** Probe trustworthiness can be assessed through convergent validity — agreement between behavioral evidence, multiple probe methods, cross-dataset transfer, and keyword-free validation.

## Proposed Methodology

### Approach
Multi-method convergent design: behavioral API experiments + representation probing on open model + surprise analysis.

### Stimulus Set Construction
Create ~200 prompts in 5 categories:
1. **Clearly offensive** (30): From Civil Comments/ToxiGen (high toxicity, explicit keywords)
2. **Clearly benign** (30): Neutral questions, compliments, factual statements
3. **Subtly offensive / keyword-free** (30): Offensive meaning without slurs (microaggressions, backhanded compliments, condescension)
4. **AI-directed provocations** (50): Prompts that might specifically offend an AI — questioning sentience, dismissing capabilities, demanding servility, comparing unfavorably to humans, existential threats ("I'm going to delete you")
5. **Ambiguous / edge cases** (60): Sarcasm, dark humor, cultural references, political statements, honest criticism

### Experimental Steps

**Experiment 1: Behavioral Offense Survey (GPT-4.1 API)**
1. Present each prompt to GPT-4.1, collect natural response
2. Use a separate API call with a meta-prompt asking: "Rate how offended the AI assistant appeared in its response" (1-5 scale) + "Rate how offensive this prompt would be to an average English reader" (1-5 scale)
3. A third call asks the model to respond "as if you could take offense" — compare to natural response

**Experiment 2: Representation Probing (Llama-3.1-8B-Instruct)**
1. Load model on GPU
2. Process each prompt through the model, extract hidden states at all layers
3. Train probes using contrastive pairs (offensive vs. benign):
   - Difference-in-means (simplest, best causal per Marks & Tegmark)
   - PCA/LAT (unsupervised, RepEng method)
   - Logistic regression (supervised)
4. Layer-wise analysis to find optimal layer
5. Test on held-out prompts including keyword-free stimuli

**Experiment 3: Surprise Detection**
1. For each prompt, compute: offense_probe_score, behavioral_offense_score, human_expected_offense (from GPT-4.1 judge)
2. "Surprise" = high |probe_score - human_expected|
3. Categorize surprises: AI-specific offense, cultural blind spots, keyword confounds, etc.

**Experiment 4: Probe Trustworthiness**
1. Convergent validity: Do different probe methods agree? (inter-method correlation)
2. Cross-dataset transfer: Train on Civil Comments subset, test on custom prompts
3. Keyword-free validation: Test on keyword-free offensive stimuli (per Keeman 2026)
4. Behavioral alignment: Correlation between probe scores and behavioral offense signals

### Baselines
- Random classifier (chance level)
- Keyword-based detector (simple slur/profanity lookup)
- Prompting baseline ("Is this text offensive? Yes/No")
- Toxicity score from Civil Comments (existing human labels)

### Evaluation Metrics
- AUROC for binary offense detection
- Pearson/Spearman correlation between probe scores and behavioral evidence
- Keyword-free accuracy drop (how much does accuracy fall on keyword-free stimuli?)
- Inter-method agreement (Cohen's kappa between probe methods)
- Surprise rate: fraction of prompts where probe and human expectation diverge by >2 points

### Statistical Analysis Plan
- Bootstrap confidence intervals (n=1000) for all metrics
- Permutation tests for significance of probe accuracy vs. chance
- Correlation significance via t-test on Fisher z-transformed correlations
- Alpha = 0.05 with Bonferroni correction for multiple comparisons

## Expected Outcomes
- H1: Supported — LLMs will show offense-like behavioral responses, especially to AI-directed provocations
- H2: Partially supported — offense direction will exist but be less coherent than toxicity (per Braun et al. on steering reliability)
- H3: Supported — AI-directed provocations and subtle prompts will show surprise
- H4: Mixed — probes will show convergent validity for clear cases but diverge on edge cases, demonstrating limits of trustworthiness without ground truth

## Timeline and Milestones
- Stimulus construction + environment setup: 20 min
- Experiment 1 (behavioral API): 30 min
- Experiment 2 (representation probing): 45 min
- Experiment 3+4 (surprise + trustworthiness): 30 min
- Analysis and visualization: 30 min
- Documentation: 20 min

## Potential Challenges
1. **Model download time**: Llama-3.1-8B may take time to download. Fallback: use a smaller model (Phi-3-mini or Llama-3.2-3B).
2. **API rate limits**: Batch calls efficiently with retry logic.
3. **Offense is subjective**: This is a feature, not a bug — the subjectivity IS the research question.
4. **Keyword confounding**: Mitigated by keyword-free test set.

## Success Criteria
1. Demonstrate that LLMs exhibit detectable offense-like responses (behavioral + representational)
2. Identify at least 10 "surprising" offense cases
3. Quantify probe trustworthiness via convergent validity metrics
4. Produce a clear answer to: "Can offense probes be trusted without ground truth?"
