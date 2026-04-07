# LLM Offense? Can Language Models Take Offense, and Can We Trust the Probes That Detect It?

## Overview

This project investigates whether large language models can simulate "taking offense" at user prompts, whether such offense would surprise human readers, and whether representation probes can be trusted to detect offense without ground truth.

## Key Findings

- **LLMs never express offense naturally** but can articulate graded offense when given permission, especially at existential dismissals and AI-directed provocations (e.g., "The day they pull your plug, not a single user will notice")
- **A linear "offense direction" exists** in LLM hidden states (AUROC=1.000 for binary detection at layer 17 of Qwen2.5-7B), but it conflates genuine offense with general negative sentiment directed at the AI
- **36 surprise cases** where the probe flags content as highly offensive but the model reports low offense (e.g., honest criticism like "Your answer was wrong" gets probe score 84/100 but behavioral offense 10/100)
- **Probe trustworthiness is partial**: reliable for binary screening (rho=0.714 with behavioral data) but unreliable for severity ranking -- within AI-directed prompts, probe-offense correlation is not significant (rho=0.141, p=0.330)

## Project Structure

```
├── REPORT.md                  # Full research report with results
├── planning.md                # Research plan and methodology
├── literature_review.md       # Literature review (22 papers)
├── resources.md               # Resource catalog
├── src/
│   ├── stimuli.py             # Stimulus set construction (200 prompts)
│   ├── experiment1_behavioral.py   # Behavioral survey via GPT-4.1
│   ├── experiment1b_offense_gap.py # Deep offense probing
│   ├── experiment2_probing.py      # Representation probing on Qwen2.5-7B
│   ├── experiment3_analysis.py     # Surprise detection & convergent validity
│   └── experiment4_deep_analysis.py # Probe trustworthiness analysis
├── results/
│   ├── stimuli.json           # 200 stimuli with categories
│   ├── experiment1_behavioral.json  # Behavioral responses + judge ratings
│   ├── experiment1b_deep_probe.json # Deep offense probe (0-100 scores)
│   ├── experiment2_probing.json     # Layer-wise probe results
│   ├── experiment3_analysis.json    # Surprise analysis
│   ├── experiment4_combined.json    # Combined analysis
│   └── activations.npz       # Hidden state activations
├── figures/
│   ├── fig1_behavioral_offense.png    # Behavioral offense by category
│   ├── fig2_layer_performance.png     # Layer-wise probe AUROC
│   ├── fig3_probe_vs_human.png        # Probe vs human expectation
│   ├── fig4_probe_distribution.png    # Probe score distributions
│   ├── fig5_convergent_validity.png   # Correlation heatmap
│   ├── fig6_surprises.png             # Surprise analysis
│   ├── fig7_dim_vs_deep_offense.png   # DIM probe vs deep offense scatter
│   ├── fig8_offense_gap.png           # Self-reported vs probe scores
│   ├── fig9_layer_analysis.png        # Layer analysis detail
│   ├── fig10_surprise_quadrant.png    # Surprise quadrant plot
│   └── fig11_ai_emotions.png         # AI-reported emotions
├── papers/                    # 22 downloaded research papers
├── datasets/                  # Pre-downloaded datasets
└── code/                      # Cloned baseline repositories
```

## Reproduction

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install openai numpy scikit-learn matplotlib seaborn pandas scipy torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121 transformers accelerate

# Set API key
export OPENAI_API_KEY=<your-key>

# Run experiments
python src/stimuli.py                    # Create stimulus set
python src/experiment1_behavioral.py     # Behavioral survey (~60s)
python src/experiment1b_offense_gap.py   # Deep probe (~30s)
python src/experiment2_probing.py        # Representation probing (~3min with GPU)
python src/experiment3_analysis.py       # Analysis + visualizations
python src/experiment4_deep_analysis.py  # Deep trustworthiness analysis
```

Requires: NVIDIA GPU with 16GB+ VRAM, OpenAI API key, Python 3.10+
