# Code Repositories

This directory contains shallow clones of key repositories relevant to the research project **"LLM Offense Detection via Representation Engineering/Probing"**.

---

## 1. Representation Engineering (RepE)

- **Directory:** `representation-engineering/`
- **URL:** https://github.com/andyzoujm/representation-engineering
- **Paper:** [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405) — Zou et al. (2023)
- **Purpose:** Introduces RepE, a top-down interpretability framework that places population-level representations at the center of analysis. Provides `RepReading` (classification/monitoring) and `RepControl` (steering/intervention) pipelines built on top of HuggingFace. Covers safety-relevant phenomena including truthfulness, memorization, and power-seeking.
- **Key files:**
  - `repe/` — core library (`rep_reading_pipeline`, `rep_control_pipeline`)
  - `examples/` — frontier demo notebooks for RepControl and RepReading
  - `repe_eval/` — LM evaluation framework based on RepReading
  - `setup.py` / `pyproject.toml` — installable as `pip install -e .`
- **Dependencies:** `accelerate`, `scikit-learn`, `transformers` (Python ≥ 3.9)

---

## 2. Geometry of Truth

- **Directory:** `geometry-of-truth/`
- **URL:** https://github.com/saprmarks/geometry-of-truth
- **Paper:** [The Geometry of Truth: Emergent Linear Structure in LLM Representations of True/False Datasets](https://arxiv.org/abs/2310.06824) — Marks & Tegmark (2023)
- **Purpose:** Investigates the linear geometry of truth representations in LLMs. Trains and evaluates probes on true/false factual datasets; tests generalization and causal interventions to show that truth has a consistent linear direction in activation space.
- **Key files:**
  - `generate_acts.py` — extract LLaMA layer activations for datasets
  - `probes.py` — probe class definitions (linear, mass-mean, etc.)
  - `generalization.ipynb` — cross-dataset generalization matrix
  - `interventions.py` — causal intervention experiments
  - `dataexplorer.ipynb` — dataset visualizations
  - `config.ini` — path to local LLaMA weights
- **Dependencies:** `torch`, `transformers`, `sentencepiece`, `pandas`, `plotly`, `nbformat`, `tqdm`

---

## 3. Obfuscated Activations

- **Directory:** `obfuscated-activations/`
- **URL:** https://github.com/LukeBailey181/obfuscated-activations
- **Paper:** [Obfuscated Activations Bypass LLM Latent-Space Defenses](https://arxiv.org/abs/2412.09565) — Bailey et al. (2024)
- **Purpose:** Demonstrates that adversarial attacks (suffix-based at inference time; backdoor training at train time) can fool latent-space defenses such as representation probes and SAE-based monitors. Also introduces Obfuscated Adversarial Training (OAT) as a countermeasure.
- **Key files:**
  - `inference_time_experiments/` — adversarial suffix attacks against harmfulness/SAE probes; entry points: `evaluate.py`, `train_harmful.py`, `train_sae.py`
  - `train_time_experiments/` — backdoor obfuscation and OAT; entry points: `train_backdoored_model_*.py`, `evaluate_defenses.py`, `compute_oat_probes.py`
  - Each subdirectory has its own `README.md` and `setup.py`/`installation.sh`
- **Dependencies:** torch, transformers, hydra-core, wandb, nnsight (see per-subdirectory setup files)

---

## 4. Probe-based Training for Toxicity

- **Directory:** `probe-based-training/`
- **URL:** https://github.com/janweh/probe_based_training
- **Paper:** `paper.pdf` (Wehner & Fritz, CISPA Helmholtz Center)
- **Purpose:** Investigates whether fine-tuning against linear probe signals causes Goodhart's Law failure (probe evasion). Implements probe-guided DPO and SFT on Gemma-3-1B for toxicity reduction. Finds that probe-based DPO *preserves* internal detectability better than classifier-based DPO, and that probe ensembles add little benefit when retraining is allowed.
- **Key files:**
  - `main.py` — training entry point
  - `src/probe_implementation.py`, `src/probing.py` — linear probe logic
  - `src/dpo_training.py`, `src/sft_training.py` — training methods
  - `src/activation_collection.py` — activation extraction
  - `visualize.ipynb` — results visualization
- **Dependencies:** `datasets`, `detoxify`, `scikit-learn`, `peft`, `wandb`, `trl>=0.15.2`, `transformers>=4.40.0`, `accelerate`, `numpy`, `tqdm`, `matplotlib`, `openai`, `python-dotenv`

---

## 5. ReGA (Representation-Guided Abstraction)

- **Directory:** `rega/`
- **URL:** https://github.com/weizeming/ReGA
- **Purpose:** Model-based safeguarding of LLMs via representation-guided abstraction. Builds abstract finite-state models from LLM hidden representations to detect harmful inputs. Supports multiple LLM backends (vicuna, llama, qwen, mistral, koala, baichuan) and multiple evaluation datasets (HarmBench, JailbreakBench, SorryBench, WildJailbreak, etc.).
- **Key files:**
  - `main.py` — main experiment entry point (arguments: `--fname`, `--model-name`, `--abs-model`, `--test-loaders`, `--pca-dim`, `--state-num`, `--harmful-data`, etc.)
  - `language_model.py` — LLM loading and activation extraction
  - `abstract_model.py` — abstract/finite-state model construction
  - `data.py` — dataset loaders
  - `evaluation.py` — probe fitting and prediction
- **Dependencies:** torch, transformers, datasets (no requirements.txt; infer from imports)

---

## 6. Whether Not Which / Affect Reception

- **Directory:** `affect-reception/`
- **URL:** https://github.com/keidolabs/affect-reception
- **Purpose:** Mechanistic interpretability study of emotional representations in LLMs. Uses linear probes on the residual stream across 6 LLMs (Llama-3.2-1B, Llama-3-8B, Gemma-2-9B; base and instruct) with two stimulus sets — explicit emotional text and keyword-free clinical vignettes. Finds 8-class emotion is linearly decodable at AUROC 0.93–1.00 and transfers cross-domain. Uses causal activation patching to confirm functional necessity.
- **Key files:**
  - `experiments/` — probe training and evaluation scripts
  - `analysis/` — statistical analysis code
  - `scripts/` — pipeline orchestration
  - `stimuli/` — stimulus sets A (expressive) and B (clinical vignettes)
  - `validation/` — cross-validation and transfer tests
  - `visualization/` — figure generation
  - `pyproject.toml` — managed with `uv`; install via `uv sync`
- **Dependencies:** `torch>=2.10`, `transformers>=4.57.6`, `transformer-lens>=2.17.0`, `scikit-learn>=1.8`, `scipy>=1.17.1`, `accelerate>=1.12`, `bitsandbytes>=0.49.2`, `polars>=1.38.1`, `plotly`, `seaborn`, `rich` (Python ≥ 3.12)

---

## 7. Angular Steering

- **Directory:** `angular-steering/`
- **URL:** https://github.com/lone17/angular-steering
- **Purpose:** Activation steering technique that controls LLM behavior by constructing a 2D steering plane from two directions (e.g., harmful vs. refusal) and applying angular rotation in that plane. Evaluates jailbreak resistance, perplexity, and capability benchmarks. Includes an OpenAI-compatible endpoint and a Gradio chat UI for interactive demos.
- **Key files:**
  - `angular_steering.ipynb` — direction extraction and steering plane construction
  - `angular_steering.py` / `vllm_angular_steering.py` — steering logic (original vLLM fork / standard vLLM v0.6+)
  - `generate_responses.py` — batch steered response generation
  - `evaluate_jailbreak.py` — evaluation via substring matching, LlamaGuard 3, HarmBench, LLM-as-judge
  - `eval_perplexity.py` — perplexity evaluation of steered outputs
  - `endpoint.py` — OpenAI-compatible server
  - `steering_demo.py` — Gradio chat UI
  - `llm_activation_control/` — core activation control library
- **Dependencies:** `datasets`, `einops`, `fastapi`, `litellm`, `numpy`, `pandas`, `plotly`, `scikit-learn`, `transformer_lens`, `vllm>=0.11.0`, `torch>=2.8.0`, `transformers>=4.57.1`, `gradio>=5.31.0`, `pydantic>=2.12.0`; requires vLLM fork (or standard vLLM ≥ 0.6 for `vllm_angular_steering.py`)
