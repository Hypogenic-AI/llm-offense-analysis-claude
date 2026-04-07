# Downloaded Papers

## Foundational - Representation Engineering

1. **Representation Engineering: A Top-Down Approach to AI Transparency** (2310.01405_repeng.pdf)
   - Authors: Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, Shashwat Goel, Nathaniel Li, Michael J. Byun, Zifan Wang, Alex Mallen, Steven Basart, Sanmi Koyejo, Dawn Song, Matt Fredrikson, J. Zico Kolter, Dan Hendrycks
   - Year: 2023
   - Why relevant: Foundational paper introducing RepE/LAT methodology for reading and controlling concepts in LLM hidden states. Studies emotions, honesty, harmfulness.

2. **The Geometry of Truth: Emergent Linear Structure in LLM Representations** (2310.06824_geometry_truth.pdf)
   - Authors: Samuel Marks, Max Tegmark
   - Year: 2024 (COLM)
   - Why relevant: Shows truth is linearly represented; difference-in-means probes are most causally relevant. Critical for understanding probe methodology.

3. **Inference-Time Intervention: Eliciting Truthful Answers from a Language Model** (2306.03341_inference_time_intervention.pdf)
   - Authors: Kenneth Li, Oam Patel, Fernanda Viegas, Hanspeter Pfister, Martin Wattenberg
   - Year: 2023
   - Why relevant: Established inference-time activation editing for truthfulness; foundational method.

4. **Discovering Latent Knowledge in Language Models Without Supervision (CCS)** (2212.03827_ccs_latent_knowledge.pdf)
   - Authors: Collin Burns, Haotian Ye, Dan Klein, Jacob Steinhardt
   - Year: 2023
   - Why relevant: Unsupervised probing method using consistency constraints. Key baseline for probing without ground truth.

## Emotion/Affect in LLM Representations

5. **LLaMAs Have Feelings Too: Unveiling Sentiment and Emotion Representations** (2505.16491_llama_feelings.pdf)
   - Authors: Di Palma et al.
   - Year: 2025 (ACL)
   - Why relevant: Directly probes sentiment and 6-class emotion in Llama models. Shows binary sentiment best in mid-layers, emotion best in early layers.

6. **Mechanistic Interpretability of Emotion Inference in Large Language Models** (2502.05489_mech_interp_emotion.pdf)
   - Authors: Tak, Banayeeanzade et al.
   - Year: 2025
   - Why relevant: Causal evidence for emotion processing in mid-layer MHSA units. Tests 10 models across 5 families.

7. **Emotions Where Art Thou** (2510.22042_emotions_latent_space.pdf)
   - Authors: Reichman, Avsian, Heck
   - Year: 2025 (ICLR 2026)
   - Why relevant: Discovers universal emotional manifold via SVD. PC1=valence, PC2=dominance. Cross-lingual stability.

8. **Whether, Not Which: Dissociable Affect Reception and Emotion Categorization** (2603.22295_whether_not_which.pdf)
   - Authors: Keeman
   - Year: 2026
   - Why relevant: Most directly relevant — shows binary affect detection achieves AUROC 1.000 even on keyword-free stimuli. Critical keyword confound analysis.

9. **Detecting and Steering LLMs' Empathy in Action** (2511.16699_llm_empathy.pdf)
   - Authors: (various)
   - Year: 2025
   - Why relevant: Probing for empathy — another affective concept close to offense/sensitivity.

## Probe Reliability and Goodhart's Law

10. **Probe-based Fine-tuning for Reducing Toxicity** (2510.21531_probe_finetune_toxicity.pdf)
    - Authors: Wehner, Fritz
    - Year: 2025
    - Why relevant: Directly addresses Goodhart's Law for probes. DPO preserves probe reliability; SFT allows partial evasion.

11. **Obfuscated Activations Bypass LLM Latent-Space Defenses** (2412.09565_obfuscated_activations.pdf)
    - Authors: Bailey, Serrano, Sheshadri et al.
    - Year: 2025
    - Why relevant: Demonstrates all latent-space defenses vulnerable to obfuscation attacks. Fundamental challenge to probe trustworthiness.

12. **The Internal State of an LLM Knows When It's Lying** (2304.13734_llm_knows_lying.pdf)
    - Authors: Azaria, Mitchell
    - Year: 2023
    - Why relevant: Foundational result showing truthfulness is decodable from hidden states. SAPLMA classifier achieves 71-90% accuracy.

13. **Preference Learning with Lie Detectors** (2411.18862_lie_detector_preference.pdf)
    - Authors: (anonymous)
    - Year: 2025
    - Why relevant: Shows probe-based training can induce honesty OR evasion depending on setup. Critical for understanding probe trust.

## Steering and Control

14. **Steering Llama 2 via Contrastive Activation Addition** (2312.06681_steering_llama2.pdf)
    - Authors: Nina Rimsky, Nick Gabrieli, Julia Schulz, Meg Tong, Evan Hubinger, Alexander Matt Turner
    - Year: 2024
    - Why relevant: CAA methodology — the primary activation steering technique.

15. **Refusal in LLMs Is Mediated by a Single Direction** (2406.11717_refusal_single_direction.pdf)
    - Authors: Arditi et al.
    - Year: 2024
    - Why relevant: Shows refusal concentrated in single direction; demonstrates fragility of safety behaviors.

16. **Understanding (Un)Reliability of Steering Vectors** (2505.22637_steering_vectors_reliability2.pdf)
    - Authors: Braun et al.
    - Year: 2025
    - Why relevant: Shows steering has high variance; unreliable when target behavior lacks coherent direction representation.

## Additional Relevant Papers

17. **Implicit Representations of Meaning in Neural Language Models** (2106.00737_implicit_representations.pdf)
    - Year: 2021
    - Why relevant: Early work on what LLMs represent internally.

18. **Probing Classifiers: Promises, Shortcomings, and Advances** (2102.12452_probing_classifiers.pdf)
    - Year: 2021
    - Why relevant: Comprehensive survey of probing methodology — essential methodological reference.

19. **Designing and Interpreting Probes with Control Tasks** (1909.03368_probes_control_tasks.pdf)
    - Authors: Hewitt, Liang
    - Year: 2019
    - Why relevant: Introduces selectivity metric and control tasks for validating probes.

20. **Semantic Entropy Probes** (2406.15927_semantic_entropy_probes.pdf)
    - Year: 2024
    - Why relevant: Robust hallucination detection via probes on semantic entropy.

21. **AxBench** (2501.17148_axbench.pdf)
    - Year: 2025
    - Why relevant: Benchmark for steering and concept detection. Shows prompting outperforms representation methods for steering.

22. **The Geometry of Refusal: Concept Cones** (2502.17420_geometry_refusal_concept_cones.pdf)
    - Year: 2025
    - Why relevant: Multiple independent refusal directions exist, not just one — complex spatial structures govern refusal.
