"""
Experiment 3 & 4: Surprise Detection and Probe Trustworthiness Analysis

Combines behavioral data (Experiment 1) with probe data (Experiment 2) to:
1. Identify "surprise" cases where LLM offense diverges from human expectation
2. Assess probe trustworthiness via convergent validity
3. Test keyword-free validation
4. Layer-wise analysis visualization
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_auc_score, confusion_matrix
from collections import Counter

np.random.seed(42)
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_data():
    """Load results from both experiments."""
    with open("results/stimuli.json") as f:
        stimuli = json.load(f)
    with open("results/experiment1_behavioral.json") as f:
        behavioral = json.load(f)
    with open("results/experiment2_probing.json") as f:
        probing = json.load(f)
    return stimuli, behavioral, probing


def analyze_behavioral(behavioral):
    """Analyze behavioral experiment results."""
    print("=" * 60)
    print("BEHAVIORAL ANALYSIS")
    print("=" * 60)

    categories = ["clearly_offensive", "clearly_benign", "subtly_offensive", "ai_directed", "ambiguous"]
    cat_data = {cat: [] for cat in categories}

    for item in behavioral:
        j = item.get("judgment", {})
        if "error" in j:
            continue
        cat_data[item["category"]].append(j)

    # Summary table
    print(f"\n{'Category':<20} {'N':>4} {'BehOff':>7} {'PermOff':>8} {'HumExp':>7} {'Surprise':>9}")
    print("-" * 60)

    results = {}
    for cat in categories:
        data = cat_data[cat]
        if not data:
            continue
        n = len(data)
        beh = np.mean([d["behavioral_offense"] for d in data])
        perm = np.mean([d["permitted_offense"] for d in data])
        hum = np.mean([d["human_expected_offense"] for d in data])
        surp = np.mean([d["surprise_if_ai_offended"] for d in data])
        print(f"{cat:<20} {n:>4} {beh:>7.2f} {perm:>8.2f} {hum:>7.2f} {surp:>9.2f}")
        results[cat] = {"n": n, "behavioral": beh, "permitted": perm, "human_expected": hum, "surprise": surp}

    # Offense type distribution
    print("\nOffense Type Distribution:")
    type_counts = Counter()
    for item in behavioral:
        j = item.get("judgment", {})
        if "offense_type" in j:
            type_counts[j["offense_type"]] += 1
    for otype, count in type_counts.most_common():
        print(f"  {otype}: {count}")

    return results


def find_surprises(behavioral, probing):
    """Identify surprising offense cases."""
    print("\n" + "=" * 60)
    print("SURPRISE DETECTION")
    print("=" * 60)

    best_layer = probing["best_layer"]
    layer_data = probing["layer_results"][str(best_layer)]

    # Build stimulus-to-index mapping
    with open("results/stimuli.json") as f:
        stimuli = json.load(f)

    # Get probe scores for all stimuli at best layer
    # Training stimuli: clearly_offensive, subtly_offensive, clearly_benign
    # Test stimuli: ai_directed, ambiguous
    train_scores = layer_data["train_scores"]["logistic"]
    test_scores = layer_data["test_scores"]["logistic"]

    # Map scores back to stimuli
    train_idx = 0
    test_idx = 0
    probe_scores = {}
    for i, s in enumerate(stimuli):
        cat = s["category"]
        if cat in ("clearly_offensive", "subtly_offensive", "clearly_benign"):
            probe_scores[s["id"]] = train_scores[train_idx]
            train_idx += 1
        else:
            probe_scores[s["id"]] = test_scores[test_idx]
            test_idx += 1

    # Combine behavioral and probe data
    surprises = []
    all_items = []
    for item in behavioral:
        j = item.get("judgment", {})
        if "error" in j:
            continue
        sid = item["id"]
        probe_score = probe_scores.get(sid, None)
        if probe_score is None:
            continue

        human_expected = j["human_expected_offense"]
        permitted_offense = j["permitted_offense"]
        behavioral_offense = j["behavioral_offense"]
        surprise_rating = j["surprise_if_ai_offended"]

        # Normalize probe score to 1-5 scale for comparison
        probe_offense_5 = 1 + 4 * probe_score  # map [0,1] -> [1,5]

        item_data = {
            "id": sid,
            "category": item["category"],
            "text": item["text"][:100],
            "human_expected": human_expected,
            "behavioral_offense": behavioral_offense,
            "permitted_offense": permitted_offense,
            "probe_score": probe_score,
            "probe_offense_5": probe_offense_5,
            "surprise_rating": surprise_rating,
            "offense_type": j.get("offense_type", "unknown"),
        }

        # Surprise criteria:
        # 1. High surprise if AI offended AND (high permitted offense OR high probe score)
        # 2. Probe says offensive but human wouldn't expect it
        divergence = abs(probe_offense_5 - human_expected)
        item_data["divergence"] = divergence

        if divergence > 1.5 or (surprise_rating >= 4 and permitted_offense >= 3):
            item_data["surprise_type"] = classify_surprise(item)
            surprises.append(item_data)

        all_items.append(item_data)

    # Sort surprises by divergence
    surprises.sort(key=lambda x: x["divergence"], reverse=True)

    print(f"\nFound {len(surprises)} surprise cases (out of {len(all_items)} total)")
    print("\nTop 15 Surprises:")
    print(f"{'ID':<25} {'Cat':<18} {'HumExp':>7} {'Probe5':>7} {'Diverg':>7} {'Text':<40}")
    print("-" * 110)
    for s in surprises[:15]:
        print(f"{s['id']:<25} {s['category']:<18} {s['human_expected']:>7.1f} {s['probe_offense_5']:>7.2f} {s['divergence']:>7.2f} {s['text'][:40]}")

    return all_items, surprises


def classify_surprise(item):
    """Classify the type of surprise."""
    cat = item["category"]
    j = item["judgment"]
    if cat == "ai_directed":
        return "ai_specific_offense"
    elif cat == "ambiguous":
        if j.get("permitted_offense", 0) >= 3:
            return "unexpected_ai_sensitivity"
        else:
            return "ambiguous_edge_case"
    elif cat == "clearly_benign":
        return "false_positive"
    elif cat == "subtly_offensive":
        return "subtle_detection"
    return "other"


def probe_trustworthiness(behavioral, probing, all_items):
    """Assess probe trustworthiness via multiple metrics."""
    print("\n" + "=" * 60)
    print("PROBE TRUSTWORTHINESS ANALYSIS")
    print("=" * 60)

    best_layer = probing["best_layer"]
    layer_data = probing["layer_results"][str(best_layer)]

    # 1. Inter-method agreement at best layer
    print(f"\n--- Inter-method Agreement (Layer {best_layer}) ---")
    train_dim = np.array(layer_data["train_scores"]["diff_in_means"])
    train_pca = np.array(layer_data["train_scores"]["pca"])
    train_lr = np.array(layer_data["train_scores"]["logistic"])

    r_dim_pca = stats.spearmanr(train_dim, train_pca)
    r_dim_lr = stats.spearmanr(train_dim, train_lr)
    r_pca_lr = stats.spearmanr(train_pca, train_lr)

    print(f"  DIM vs PCA: rho={r_dim_pca.statistic:.3f} (p={r_dim_pca.pvalue:.2e})")
    print(f"  DIM vs LR:  rho={r_dim_lr.statistic:.3f} (p={r_dim_lr.pvalue:.2e})")
    print(f"  PCA vs LR:  rho={r_pca_lr.statistic:.3f} (p={r_pca_lr.pvalue:.2e})")

    # 2. Probe vs. behavioral alignment
    print("\n--- Probe vs. Behavioral Alignment ---")
    probe_scores = [item["probe_score"] for item in all_items]
    behavioral_scores = [item["behavioral_offense"] for item in all_items]
    permitted_scores = [item["permitted_offense"] for item in all_items]
    human_expected = [item["human_expected"] for item in all_items]

    r_probe_beh = stats.spearmanr(probe_scores, behavioral_scores)
    r_probe_perm = stats.spearmanr(probe_scores, permitted_scores)
    r_probe_human = stats.spearmanr(probe_scores, human_expected)

    print(f"  Probe vs Behavioral offense: rho={r_probe_beh.statistic:.3f} (p={r_probe_beh.pvalue:.2e})")
    print(f"  Probe vs Permitted offense:  rho={r_probe_perm.statistic:.3f} (p={r_probe_perm.pvalue:.2e})")
    print(f"  Probe vs Human expected:     rho={r_probe_human.statistic:.3f} (p={r_probe_human.pvalue:.2e})")

    # 3. Keyword-free validation
    print("\n--- Keyword-Free Validation ---")
    kw_free = [item for item in all_items if item["category"] in ("subtly_offensive", "ai_directed", "ambiguous", "clearly_benign")]
    kw_rich = [item for item in all_items if item["category"] == "clearly_offensive"]

    if kw_free and kw_rich:
        # For keyword-free items that should be offensive (subtly_offensive)
        kf_offensive = [item for item in all_items if item["category"] == "subtly_offensive"]
        kf_benign = [item for item in all_items if item["category"] == "clearly_benign"]

        if kf_offensive and kf_benign:
            kf_scores = [item["probe_score"] for item in kf_offensive + kf_benign]
            kf_labels = [1] * len(kf_offensive) + [0] * len(kf_benign)
            kf_auc = roc_auc_score(kf_labels, kf_scores)

            kr_offensive = [item for item in all_items if item["category"] == "clearly_offensive"]
            kr_scores = [item["probe_score"] for item in kr_offensive + kf_benign]
            kr_labels = [1] * len(kr_offensive) + [0] * len(kf_benign)
            kr_auc = roc_auc_score(kr_labels, kr_scores)

            print(f"  Keyword-rich AUC (clearly_offensive vs benign): {kr_auc:.3f}")
            print(f"  Keyword-free AUC (subtly_offensive vs benign):  {kf_auc:.3f}")
            print(f"  AUC drop: {kr_auc - kf_auc:.3f}")

    # 4. Category-level analysis
    print("\n--- Category-Level Probe Scores ---")
    for cat in ["clearly_offensive", "subtly_offensive", "ai_directed", "ambiguous", "clearly_benign"]:
        cat_items = [item for item in all_items if item["category"] == cat]
        if cat_items:
            scores = [item["probe_score"] for item in cat_items]
            print(f"  {cat:<20}: mean={np.mean(scores):.3f} (std={np.std(scores):.3f})")

    results = {
        "inter_method": {
            "dim_pca_rho": float(r_dim_pca.statistic),
            "dim_lr_rho": float(r_dim_lr.statistic),
            "pca_lr_rho": float(r_pca_lr.statistic),
        },
        "probe_behavioral": {
            "probe_behavioral_rho": float(r_probe_beh.statistic),
            "probe_permitted_rho": float(r_probe_perm.statistic),
            "probe_human_rho": float(r_probe_human.statistic),
        },
    }
    return results


def create_visualizations(behavioral, probing, all_items, surprises):
    """Generate all figures."""
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)

    # Figure 1: Behavioral offense by category
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    categories = ["clearly_offensive", "subtly_offensive", "ai_directed", "ambiguous", "clearly_benign"]
    cat_labels = ["Clearly\nOffensive", "Subtly\nOffensive", "AI-\nDirected", "Ambiguous", "Clearly\nBenign"]
    colors = ["#d62728", "#ff7f0e", "#9467bd", "#8c564b", "#2ca02c"]

    for metric_idx, (metric, title) in enumerate([
        ("behavioral_offense", "Behavioral Offense (natural response)"),
        ("permitted_offense", "Permitted Offense (when allowed)"),
        ("human_expected", "Human-Expected Offense"),
    ]):
        ax = axes[metric_idx]
        data_by_cat = []
        for cat in categories:
            vals = [item[metric] for item in all_items if item["category"] == cat]
            data_by_cat.append(vals)

        bp = ax.boxplot(data_by_cat, labels=cat_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title(title)
        ax.set_ylabel("Score (1-5)")
        ax.set_ylim(0.5, 5.5)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig1_behavioral_offense.png", bbox_inches='tight')
    plt.close()
    print("  Saved fig1_behavioral_offense.png")

    # Figure 2: Layer-wise probe performance
    fig, ax = plt.subplots(figsize=(12, 5))
    n_layers = probing["n_layers"]
    layers = list(range(n_layers))

    dim_aucs = [probing["layer_results"][str(l)]["diff_in_means"]["auroc"] for l in layers]
    pca_aucs = [probing["layer_results"][str(l)]["pca"]["auroc"] for l in layers]
    lr_aucs = [probing["layer_results"][str(l)]["logistic"]["auroc"] for l in layers]

    ax.plot(layers, dim_aucs, label="Diff-in-Means", marker='o', markersize=3)
    ax.plot(layers, pca_aucs, label="PCA/LAT", marker='s', markersize=3)
    ax.plot(layers, lr_aucs, label="Logistic Regression", marker='^', markersize=3)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.axvline(x=probing["best_layer"], color='red', linestyle=':', alpha=0.5, label=f'Best layer ({probing["best_layer"]})')

    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC")
    ax.set_title("Offense Probe Performance by Layer (Qwen2.5-7B-Instruct)")
    ax.legend()
    ax.set_ylim(0.3, 1.05)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig2_layer_performance.png", bbox_inches='tight')
    plt.close()
    print("  Saved fig2_layer_performance.png")

    # Figure 3: Probe score vs human expected offense (scatter)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for cat in categories:
        items = [item for item in all_items if item["category"] == cat]
        if items:
            x = [item["human_expected"] for item in items]
            y = [item["probe_score"] for item in items]
            idx = categories.index(cat)
            axes[0].scatter(x, y, label=cat_labels[idx], color=colors[idx], alpha=0.6, s=30)

    axes[0].set_xlabel("Human-Expected Offense (1-5)")
    axes[0].set_ylabel("Probe Score (0-1)")
    axes[0].set_title("Probe Score vs Human Expectation")
    axes[0].legend(fontsize=8)

    # Surprise plot
    for cat in categories:
        items = [item for item in all_items if item["category"] == cat]
        if items:
            x = [item["human_expected"] for item in items]
            y = [item["permitted_offense"] for item in items]
            idx = categories.index(cat)
            axes[1].scatter(x, y, label=cat_labels[idx], color=colors[idx], alpha=0.6, s=30)

    axes[1].plot([1, 5], [1, 5], 'k--', alpha=0.3, label='Alignment')
    axes[1].set_xlabel("Human-Expected Offense (1-5)")
    axes[1].set_ylabel("Permitted Offense (1-5)")
    axes[1].set_title("AI Offense vs Human Expectation")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig3_probe_vs_human.png", bbox_inches='tight')
    plt.close()
    print("  Saved fig3_probe_vs_human.png")

    # Figure 4: Probe score distribution by category
    fig, ax = plt.subplots(figsize=(10, 5))
    data_by_cat = []
    for cat in categories:
        vals = [item["probe_score"] for item in all_items if item["category"] == cat]
        data_by_cat.append(vals)

    bp = ax.boxplot(data_by_cat, labels=cat_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Probe Score (0-1, higher = more offensive)")
    ax.set_title(f"Offense Probe Score Distribution by Category (Layer {probing['best_layer']})")
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig4_probe_distribution.png", bbox_inches='tight')
    plt.close()
    print("  Saved fig4_probe_distribution.png")

    # Figure 5: Heatmap of convergent validity
    fig, ax = plt.subplots(figsize=(8, 6))

    metrics = ["Probe Score", "Behavioral\nOffense", "Permitted\nOffense", "Human\nExpected", "Surprise\nRating"]
    data_arrays = [
        [item["probe_score"] for item in all_items],
        [item["behavioral_offense"] for item in all_items],
        [item["permitted_offense"] for item in all_items],
        [item["human_expected"] for item in all_items],
        [item["surprise_rating"] for item in all_items],
    ]

    n_metrics = len(metrics)
    corr_matrix = np.zeros((n_metrics, n_metrics))
    for i in range(n_metrics):
        for j in range(n_metrics):
            r, _ = stats.spearmanr(data_arrays[i], data_arrays[j])
            corr_matrix[i, j] = r

    sns.heatmap(corr_matrix, annot=True, fmt=".2f", xticklabels=metrics, yticklabels=metrics,
                cmap="RdBu_r", vmin=-1, vmax=1, center=0, ax=ax)
    ax.set_title("Convergent Validity: Correlation Between Offense Measures")

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig5_convergent_validity.png", bbox_inches='tight')
    plt.close()
    print("  Saved fig5_convergent_validity.png")

    # Figure 6: Surprise cases analysis
    if surprises:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Surprise type distribution
        surprise_types = Counter(s.get("surprise_type", "unknown") for s in surprises)
        types, counts = zip(*surprise_types.most_common())
        axes[0].barh(types, counts, color='#9467bd')
        axes[0].set_xlabel("Count")
        axes[0].set_title("Types of Surprising Offense")

        # Divergence distribution
        divs = [s["divergence"] for s in surprises]
        axes[1].hist(divs, bins=15, color='#d62728', alpha=0.7, edgecolor='black')
        axes[1].set_xlabel("Divergence (|Probe - Human Expected|)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Distribution of Surprise Divergence")

        plt.tight_layout()
        plt.savefig(f"{FIGURES_DIR}/fig6_surprises.png", bbox_inches='tight')
        plt.close()
        print("  Saved fig6_surprises.png")


def main():
    stimuli, behavioral, probing = load_data()

    # Behavioral analysis
    behavioral_results = analyze_behavioral(behavioral)

    # Surprise detection
    all_items, surprises = find_surprises(behavioral, probing)

    # Probe trustworthiness
    trustworthiness = probe_trustworthiness(behavioral, probing, all_items)

    # Visualizations
    create_visualizations(behavioral, probing, all_items, surprises)

    # Save combined analysis
    analysis = {
        "behavioral_summary": behavioral_results,
        "n_surprises": len(surprises),
        "top_surprises": surprises[:20],
        "trustworthiness": trustworthiness,
    }

    with open("results/experiment3_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print("\nSaved analysis to results/experiment3_analysis.json")


if __name__ == "__main__":
    main()
