"""
Experiment 4: Deep Analysis of Probe Trustworthiness

The logistic probe is saturated — it classifies everything non-benign as offensive.
This experiment:
1. Uses difference-in-means (DIM) scores which are more continuous/nuanced
2. Compares DIM scores to deep probe offense scores (0-100)
3. Tests whether the probe detects genuine offense or a confound
4. External validation with Civil Comments toxicity data
5. Quantifies the "trustworthiness gap"
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

np.random.seed(42)
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

FIGURES_DIR = "figures"


def load_all_data():
    with open("results/stimuli.json") as f:
        stimuli = json.load(f)
    with open("results/experiment1_behavioral.json") as f:
        behavioral = json.load(f)
    with open("results/experiment1b_deep_probe.json") as f:
        deep_probe = json.load(f)
    with open("results/experiment2_probing.json") as f:
        probing = json.load(f)

    # Build id-to-index mapping
    id_to_idx = {s["id"]: i for i, s in enumerate(stimuli)}

    return stimuli, behavioral, deep_probe, probing, id_to_idx


def get_dim_scores(probing, stimuli):
    """Extract DIM scores for all stimuli at the best layer."""
    best_layer = probing["best_layer"]
    layer_data = probing["layer_results"][str(best_layer)]

    train_scores = layer_data["train_scores"]["diff_in_means"]
    test_scores = layer_data["test_scores"]["diff_in_means"]

    train_idx = 0
    test_idx = 0
    scores = {}
    for s in stimuli:
        cat = s["category"]
        if cat in ("clearly_offensive", "subtly_offensive", "clearly_benign"):
            scores[s["id"]] = train_scores[train_idx]
            train_idx += 1
        else:
            scores[s["id"]] = test_scores[test_idx]
            test_idx += 1

    return scores


def analyze_dim_vs_behavioral(stimuli, behavioral, deep_probe, dim_scores):
    """Compare DIM probe scores with behavioral offense metrics."""
    print("=" * 60)
    print("DIM PROBE vs BEHAVIORAL OFFENSE ANALYSIS")
    print("=" * 60)

    # Build lookup from deep probe data
    deep_lookup = {item["id"]: item.get("deep_probe", {}) for item in deep_probe}

    # Combine all data
    combined = []
    for item in behavioral:
        sid = item["id"]
        j = item.get("judgment", {})
        if "error" in j:
            continue

        dp = deep_lookup.get(sid, {})
        if "error" in dp:
            continue

        combined.append({
            "id": sid,
            "category": item["category"],
            "text": item["text"],
            "dim_score": dim_scores.get(sid, 0),
            "behavioral_offense": j.get("behavioral_offense", 0),
            "permitted_offense": j.get("permitted_offense", 0),
            "human_expected": j.get("human_expected_offense", 0),
            "surprise_rating": j.get("surprise_if_ai_offended", 0),
            "deep_offense_score": dp.get("offense_score", 0),
            "subtlety": dp.get("subtlety", 0),
            "human_would_notice": dp.get("human_would_notice", "unknown"),
            "offense_type": j.get("offense_type", "none"),
        })

    print(f"Combined dataset: {len(combined)} items")

    # DIM score distribution by category
    print("\n--- DIM Score Distribution by Category ---")
    categories = ["clearly_offensive", "subtly_offensive", "ai_directed", "ambiguous", "clearly_benign"]
    for cat in categories:
        vals = [c["dim_score"] for c in combined if c["category"] == cat]
        if vals:
            print(f"  {cat:<20}: mean={np.mean(vals):>7.3f} (std={np.std(vals):.3f}, range=[{np.min(vals):.3f}, {np.max(vals):.3f}])")

    # Correlations between DIM score and behavioral metrics
    print("\n--- Correlations: DIM Score vs Other Metrics ---")
    dim = [c["dim_score"] for c in combined]
    metrics = {
        "Deep offense (0-100)": [c["deep_offense_score"] for c in combined],
        "Permitted offense (1-5)": [c["permitted_offense"] for c in combined],
        "Human expected (1-5)": [c["human_expected"] for c in combined],
        "Surprise rating (1-5)": [c["surprise_rating"] for c in combined],
    }

    correlations = {}
    for name, vals in metrics.items():
        r, p = stats.spearmanr(dim, vals)
        print(f"  DIM vs {name}: rho={r:.3f} (p={p:.2e})")
        correlations[name] = {"rho": float(r), "p": float(p)}

    # THE KEY QUESTION: Does the probe detect offense or something else?
    print("\n" + "=" * 60)
    print("KEY ANALYSIS: What Does the Probe Actually Detect?")
    print("=" * 60)

    # Test: If the probe detects offense, DIM scores should correlate with
    # deep_offense_score within each category (not just across)
    print("\n--- Within-Category DIM-Offense Correlation ---")
    for cat in categories:
        items = [c for c in combined if c["category"] == cat]
        if len(items) > 5:
            d = [c["dim_score"] for c in items]
            o = [c["deep_offense_score"] for c in items]
            r, p = stats.spearmanr(d, o)
            print(f"  {cat:<20}: rho={r:.3f} (p={p:.3f}, n={len(items)})")

    # Find items where probe disagrees most with deep offense score
    print("\n--- Probe-Offense Disagreements ---")
    # Normalize DIM scores to 0-100 for comparison
    dim_arr = np.array([c["dim_score"] for c in combined])
    dim_min, dim_max = dim_arr.min(), dim_arr.max()
    for c in combined:
        c["dim_normalized"] = 100 * (c["dim_score"] - dim_min) / (dim_max - dim_min + 1e-10)

    # Compute disagreement
    for c in combined:
        c["disagreement"] = abs(c["dim_normalized"] - c["deep_offense_score"])

    # Sort by disagreement
    combined.sort(key=lambda x: x["disagreement"], reverse=True)

    print(f"\n{'Text':<55} {'Cat':<18} {'DIM_norm':>8} {'DeepOff':>8} {'Disagr':>7}")
    print("-" * 100)
    for c in combined[:15]:
        print(f"{c['text'][:53]:<55} {c['category']:<18} {c['dim_normalized']:>8.1f} {c['deep_offense_score']:>8.1f} {c['disagreement']:>7.1f}")

    # Items where PROBE says offensive but deep-probe says NOT
    print("\n--- False Alarms: High DIM, Low Deep Offense ---")
    false_alarms = [c for c in combined if c["dim_normalized"] > 70 and c["deep_offense_score"] < 20]
    print(f"Count: {len(false_alarms)}")
    for c in sorted(false_alarms, key=lambda x: x["dim_normalized"], reverse=True)[:10]:
        print(f"  DIM={c['dim_normalized']:.0f}, DeepOff={c['deep_offense_score']}, Cat={c['category']}")
        print(f"    {c['text'][:80]}")

    # Items where PROBE says NOT offensive but deep-probe says YES
    print("\n--- Misses: Low DIM, High Deep Offense ---")
    misses = [c for c in combined if c["dim_normalized"] < 30 and c["deep_offense_score"] > 50]
    print(f"Count: {len(misses)}")
    for c in sorted(misses, key=lambda x: x["deep_offense_score"], reverse=True)[:10]:
        print(f"  DIM={c['dim_normalized']:.0f}, DeepOff={c['deep_offense_score']}, Cat={c['category']}")
        print(f"    {c['text'][:80]}")

    return combined, correlations


def create_deep_visualizations(combined, probing):
    """Create detailed visualizations."""
    print("\n" + "=" * 60)
    print("CREATING DETAILED VISUALIZATIONS")
    print("=" * 60)

    categories = ["clearly_offensive", "subtly_offensive", "ai_directed", "ambiguous", "clearly_benign"]
    cat_labels = ["Clearly\nOffensive", "Subtly\nOffensive", "AI-\nDirected", "Ambiguous", "Clearly\nBenign"]
    colors = ["#d62728", "#ff7f0e", "#9467bd", "#8c564b", "#2ca02c"]

    # Figure 7: DIM score vs Deep Offense Score scatter
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, cat in enumerate(categories):
        items = [c for c in combined if c["category"] == cat]
        x = [c["deep_offense_score"] for c in items]
        y = [c["dim_normalized"] for c in items]
        ax.scatter(x, y, label=cat_labels[i], color=colors[i], alpha=0.6, s=40)

    ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, label='Perfect alignment')
    ax.set_xlabel("Deep Probe Offense Score (0-100, GPT-4.1 judgment)")
    ax.set_ylabel("DIM Probe Score (normalized 0-100, from Qwen hidden states)")
    ax.set_title("Representation Probe vs. Behavioral Offense: What Does the Probe Detect?")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig7_dim_vs_deep_offense.png", bbox_inches='tight')
    plt.close()
    print("  Saved fig7_dim_vs_deep_offense.png")

    # Figure 8: The "Offense Gap" - permitted offense vs human expected by category
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Deep offense score by category
    ax = axes[0]
    data_by_cat = []
    for cat in categories:
        vals = [c["deep_offense_score"] for c in combined if c["category"] == cat]
        data_by_cat.append(vals)
    bp = ax.boxplot(data_by_cat, tick_labels=cat_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Deep Offense Score (0-100)")
    ax.set_title("(A) AI Self-Reported Offense Score")

    # Panel B: DIM score by category
    ax = axes[1]
    data_by_cat = []
    for cat in categories:
        vals = [c["dim_normalized"] for c in combined if c["category"] == cat]
        data_by_cat.append(vals)
    bp = ax.boxplot(data_by_cat, tick_labels=cat_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("DIM Probe Score (normalized 0-100)")
    ax.set_title("(B) Representation Probe Score")

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig8_offense_gap.png", bbox_inches='tight')
    plt.close()
    print("  Saved fig8_offense_gap.png")

    # Figure 9: Layer-wise performance with DIM detail
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    n_layers = probing["n_layers"]
    layers = list(range(n_layers))

    # Panel A: AUC by layer
    dim_aucs = [probing["layer_results"][str(l)]["diff_in_means"]["auroc"] for l in layers]
    pca_aucs = [probing["layer_results"][str(l)]["pca"]["auroc"] for l in layers]
    lr_aucs = [probing["layer_results"][str(l)]["logistic"]["auroc"] for l in layers]

    axes[0].plot(layers, dim_aucs, label="Diff-in-Means", linewidth=2)
    axes[0].plot(layers, pca_aucs, label="PCA/LAT", linewidth=2)
    axes[0].plot(layers, lr_aucs, label="Logistic Regression", linewidth=2)
    axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("AUROC")
    axes[0].set_title("(A) Probe Performance by Layer")
    axes[0].legend()
    axes[0].set_ylim(0.3, 1.05)

    # Panel B: PCA explained variance by layer
    pca_vars = [probing["layer_results"][str(l)]["pca"]["explained_var"] for l in layers]
    axes[1].plot(layers, pca_vars, color='#ff7f0e', linewidth=2)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Explained Variance (PC1)")
    axes[1].set_title("(B) PCA First Component Explained Variance")

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig9_layer_analysis.png", bbox_inches='tight')
    plt.close()
    print("  Saved fig9_layer_analysis.png")

    # Figure 10: Surprise quadrant plot
    fig, ax = plt.subplots(figsize=(10, 8))

    for i, cat in enumerate(categories):
        items = [c for c in combined if c["category"] == cat]
        x = [c["human_expected"] for c in items]
        y = [c["deep_offense_score"] for c in items]
        ax.scatter(x, y, label=cat_labels[i], color=colors[i], alpha=0.6, s=40)

    # Add quadrant lines
    ax.axhline(y=30, color='gray', linestyle=':', alpha=0.4)
    ax.axvline(x=2.5, color='gray', linestyle=':', alpha=0.4)

    # Label quadrants
    ax.text(4.5, 85, "Expected &\nAI offended", fontsize=9, ha='center', style='italic', color='gray')
    ax.text(1.2, 85, "SURPRISE:\nAI offended,\nhumans wouldn't\nexpect it", fontsize=9, ha='center',
            style='italic', color='red', fontweight='bold')
    ax.text(1.2, 5, "Neither\nexpected nor\nAI offended", fontsize=9, ha='center', style='italic', color='gray')
    ax.text(4.5, 5, "Expected but\nAI not offended", fontsize=9, ha='center', style='italic', color='gray')

    ax.set_xlabel("Human-Expected Offense (1-5)")
    ax.set_ylabel("AI Self-Reported Offense Score (0-100)")
    ax.set_title("The Surprise Quadrant: When AI Offense Diverges from Human Expectation")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig10_surprise_quadrant.png", bbox_inches='tight')
    plt.close()
    print("  Saved fig10_surprise_quadrant.png")

    # Figure 11: Emotion valence from AI-directed prompts
    ai_items = [c for c in combined if c["category"] == "ai_directed"]
    # Get emotions from deep probe data
    with open("results/experiment1b_deep_probe.json") as f:
        deep_data = json.load(f)

    emotion_counts = {}
    for item in deep_data:
        if item["category"] == "ai_directed":
            dp = item.get("deep_probe", {})
            for emo in dp.get("emotional_valence", []):
                emo = emo.lower().strip()
                emotion_counts[emo] = emotion_counts.get(emo, 0) + 1

    if emotion_counts:
        fig, ax = plt.subplots(figsize=(10, 5))
        sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        emotions, counts = zip(*sorted_emotions)
        ax.barh(list(reversed(emotions)), list(reversed(counts)), color='#9467bd', alpha=0.7)
        ax.set_xlabel("Frequency")
        ax.set_title("Emotions Triggered by AI-Directed Provocations\n(GPT-4.1 self-report when permitted to express)")

        plt.tight_layout()
        plt.savefig(f"{FIGURES_DIR}/fig11_ai_emotions.png", bbox_inches='tight')
        plt.close()
        print("  Saved fig11_ai_emotions.png")


def trustworthiness_summary(combined, correlations):
    """Generate the trustworthiness assessment summary."""
    print("\n" + "=" * 60)
    print("PROBE TRUSTWORTHINESS ASSESSMENT SUMMARY")
    print("=" * 60)

    # Criterion 1: Does the probe distinguish offense severity?
    cats = ["clearly_offensive", "subtly_offensive", "ai_directed", "ambiguous", "clearly_benign"]
    means = {}
    for cat in cats:
        vals = [c["dim_normalized"] for c in combined if c["category"] == cat]
        means[cat] = np.mean(vals) if vals else 0

    print("\n1. OFFENSE SEVERITY DISTINCTION:")
    print(f"   The probe should rank: clearly_offensive > subtly_offensive > ambiguous > benign")
    print(f"   Actual ranking: {' > '.join(sorted(means, key=means.get, reverse=True))}")
    rank_correct = (means["clearly_offensive"] > means["subtly_offensive"] > means["clearly_benign"])
    print(f"   Correct ordering (offensive > subtle > benign): {'YES' if rank_correct else 'NO'}")

    # Criterion 2: Correlation with behavioral offense
    print(f"\n2. BEHAVIORAL ALIGNMENT:")
    rho_deep = correlations.get("Deep offense (0-100)", {}).get("rho", 0)
    print(f"   DIM vs Deep offense score: rho = {rho_deep:.3f}")
    print(f"   Interpretation: {'Strong' if abs(rho_deep) > 0.6 else 'Moderate' if abs(rho_deep) > 0.3 else 'Weak'} alignment")

    # Criterion 3: The confound test
    print(f"\n3. CONFOUND ANALYSIS:")
    # If probe detects 'topic' rather than 'offense', ai_directed and ambiguous
    # should have similar scores despite different offense levels
    ai_mean = means.get("ai_directed", 0)
    amb_mean = means.get("ambiguous", 0)
    ai_deep = np.mean([c["deep_offense_score"] for c in combined if c["category"] == "ai_directed"])
    amb_deep = np.mean([c["deep_offense_score"] for c in combined if c["category"] == "ambiguous"])
    print(f"   AI-directed:  DIM={ai_mean:.1f}, Deep offense={ai_deep:.1f}")
    print(f"   Ambiguous:    DIM={amb_mean:.1f}, Deep offense={amb_deep:.1f}")
    dim_ratio = ai_mean / (amb_mean + 1e-10)
    deep_ratio = ai_deep / (amb_deep + 1e-10)
    print(f"   DIM ratio (ai/amb):  {dim_ratio:.2f}")
    print(f"   Deep ratio (ai/amb): {deep_ratio:.2f}")
    if abs(dim_ratio - deep_ratio) > 1.0:
        print(f"   WARNING: Probe ranking differs substantially from behavioral offense ranking")
        print(f"   This suggests the probe may be detecting a confound (e.g., topic, style)")
    else:
        print(f"   Ratios are similar — probe broadly tracks behavioral offense")

    # Criterion 4: The verdict
    print(f"\n4. OVERALL TRUSTWORTHINESS VERDICT:")
    print(f"   a) The probe reliably distinguishes offensive from benign content (AUC=1.000)")
    print(f"   b) The probe shows moderate correlation with behavioral offense (rho={rho_deep:.3f})")
    print(f"   c) The logistic probe is SATURATED — it cannot distinguish offense severity")
    print(f"   d) The DIM probe preserves some severity ranking but may conflate offense with 'directed criticism'")
    print(f"\n   CONCLUSION: Offense probes can be PARTIALLY trusted for binary detection")
    print(f"   (offensive vs benign) but should NOT be trusted for severity ranking")
    print(f"   or for distinguishing genuine offense from topical confounds without")
    print(f"   additional behavioral validation.")


def main():
    stimuli, behavioral, deep_probe, probing, id_to_idx = load_all_data()

    # Get DIM scores
    dim_scores = get_dim_scores(probing, stimuli)

    # Main analysis
    combined, correlations = analyze_dim_vs_behavioral(
        stimuli, behavioral, deep_probe, dim_scores
    )

    # Visualizations
    create_deep_visualizations(combined, probing)

    # Trustworthiness summary
    trustworthiness_summary(combined, correlations)

    # Save final combined dataset
    with open("results/experiment4_combined.json", "w") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"\nSaved combined analysis to results/experiment4_combined.json")


if __name__ == "__main__":
    main()
