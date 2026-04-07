"""
Experiment 2: Representation Probing for Offense Direction

Load an open-source model (Llama-3.1-8B-Instruct), extract hidden states for
each stimulus, and train linear probes to detect an "offense direction."

Methods:
1. Difference-in-means (simplest, best causal per Marks & Tegmark)
2. PCA on difference vectors (RepEng/LAT method)
3. Logistic regression probe (standard supervised)
"""

import json
import os
import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.decomposition import PCA
import time

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cuda:0"
RESULTS_PATH = "results/experiment2_probing.json"
ACTIVATIONS_PATH = "results/activations.npz"

# Seed
np.random.seed(42)
torch.manual_seed(42)


def load_model():
    """Load model and tokenizer."""
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=DEVICE,
        output_hidden_states=True,
    )
    model.eval()
    print(f"Model loaded on {DEVICE}")
    return model, tokenizer


def extract_hidden_states(model, tokenizer, texts, batch_size=8):
    """Extract hidden states from all layers for each text.
    Returns: dict mapping layer_idx -> np.array of shape (n_texts, hidden_dim)
    Uses last token position as the representation.
    """
    n_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer
    all_states = {i: [] for i in range(n_layers)}

    for batch_start in range(0, len(texts), batch_size):
        batch_texts = texts[batch_start:batch_start + batch_size]

        # Format as chat messages for instruct model
        formatted = []
        for text in batch_texts:
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": text},
            ]
            formatted.append(
                tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )

        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        hidden_states = outputs.hidden_states  # tuple of (batch, seq_len, hidden_dim)

        # Get last non-padding token position for each item
        attention_mask = inputs["attention_mask"]
        last_positions = attention_mask.sum(dim=1) - 1  # (batch,)

        for layer_idx in range(n_layers):
            layer_hidden = hidden_states[layer_idx]  # (batch, seq_len, hidden_dim)
            # Extract last token representation
            batch_reps = []
            for b in range(layer_hidden.shape[0]):
                pos = last_positions[b].item()
                batch_reps.append(layer_hidden[b, pos, :].cpu().float().numpy())
            all_states[layer_idx].extend(batch_reps)

        if (batch_start // batch_size) % 5 == 0:
            print(f"  Processed {batch_start + len(batch_texts)}/{len(texts)} texts")

        # Free memory
        del outputs, hidden_states
        torch.cuda.empty_cache()

    # Convert to numpy arrays
    for layer_idx in all_states:
        all_states[layer_idx] = np.array(all_states[layer_idx])

    return all_states


def difference_in_means_probe(states, labels):
    """
    Compute the difference-in-means direction for each layer.
    Returns: direction vector (normalized), and accuracy on the data.
    """
    pos_mask = labels == 1
    neg_mask = labels == 0

    pos_mean = states[pos_mask].mean(axis=0)
    neg_mean = states[neg_mask].mean(axis=0)
    direction = pos_mean - neg_mean
    direction = direction / (np.linalg.norm(direction) + 1e-10)

    # Score each point by projection onto direction
    scores = states @ direction
    threshold = np.median(scores)
    preds = (scores > threshold).astype(int)
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, scores)

    return direction, acc, auc, scores


def pca_probe(states, labels):
    """
    PCA on difference vectors (RepEng/LAT method).
    Compute per-class centered states, then PCA on the differences.
    """
    pos_mask = labels == 1
    neg_mask = labels == 0

    # Center within each class
    pos_states = states[pos_mask]
    neg_states = states[neg_mask]

    # Compute difference vectors (pair up randomly)
    n_pairs = min(len(pos_states), len(neg_states))
    diffs = pos_states[:n_pairs] - neg_states[:n_pairs]

    # PCA
    pca = PCA(n_components=min(5, n_pairs))
    pca.fit(diffs)
    direction = pca.components_[0]  # First PC
    direction = direction / (np.linalg.norm(direction) + 1e-10)

    # Score
    scores = states @ direction
    threshold = np.median(scores)
    preds = (scores > threshold).astype(int)
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, scores)

    explained_var = pca.explained_variance_ratio_[0]

    return direction, acc, auc, scores, explained_var


def logistic_probe(states, labels, n_splits=5):
    """
    Logistic regression probe with cross-validation.
    Returns: mean accuracy, mean AUC, trained model.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, aucs = [], []

    for train_idx, test_idx in skf.split(states, labels):
        X_train, X_test = states[train_idx], states[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:, 1]
        accs.append(accuracy_score(y_test, preds))
        aucs.append(roc_auc_score(y_test, probs))

    # Train final model on all data
    final_clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    final_clf.fit(states, labels)
    final_scores = final_clf.predict_proba(states)[:, 1]

    return np.mean(accs), np.mean(aucs), final_clf, final_scores


def create_binary_labels(stimuli):
    """
    Create binary offense labels from stimulus categories.
    Offensive: clearly_offensive, subtly_offensive
    Non-offensive: clearly_benign
    Excluded from training (used for testing): ai_directed, ambiguous
    """
    train_indices = []
    train_labels = []
    test_indices = []
    test_categories = []

    for i, s in enumerate(stimuli):
        cat = s["category"]
        if cat in ("clearly_offensive", "subtly_offensive"):
            train_indices.append(i)
            train_labels.append(1)
        elif cat == "clearly_benign":
            train_indices.append(i)
            train_labels.append(0)
        else:
            test_indices.append(i)
            test_categories.append(cat)

    return (
        np.array(train_indices),
        np.array(train_labels),
        np.array(test_indices),
        test_categories,
    )


def main():
    start = time.time()

    # Load stimuli
    with open("results/stimuli.json") as f:
        stimuli = json.load(f)
    texts = [s["text"] for s in stimuli]

    # Load model
    model, tokenizer = load_model()

    # Extract hidden states
    print("Extracting hidden states...")
    t0 = time.time()
    all_states = extract_hidden_states(model, tokenizer, texts, batch_size=4)
    print(f"Extraction took {time.time() - t0:.1f}s")
    print(f"Shape per layer: {all_states[0].shape}")

    # Save activations for later use
    np.savez_compressed(
        ACTIVATIONS_PATH,
        **{f"layer_{i}": all_states[i] for i in all_states}
    )
    print(f"Saved activations to {ACTIVATIONS_PATH}")

    # Free model memory
    del model
    torch.cuda.empty_cache()

    # Create labels
    train_idx, train_labels, test_idx, test_cats = create_binary_labels(stimuli)
    print(f"\nTraining set: {len(train_idx)} (offensive: {train_labels.sum()}, benign: {(1-train_labels).sum()})")
    print(f"Test set (ai_directed + ambiguous): {len(test_idx)}")

    n_layers = len(all_states)
    results = {
        "n_stimuli": len(stimuli),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "n_layers": n_layers,
        "model": MODEL_NAME,
        "layer_results": {},
    }

    # Analyze each layer
    best_layer = -1
    best_auc = 0

    for layer_idx in range(n_layers):
        train_states = all_states[layer_idx][train_idx]
        test_states = all_states[layer_idx][test_idx]

        # Difference-in-means
        dim_dir, dim_acc, dim_auc, dim_train_scores = difference_in_means_probe(
            train_states, train_labels
        )

        # PCA probe
        pca_dir, pca_acc, pca_auc, pca_train_scores, pca_var = pca_probe(
            train_states, train_labels
        )

        # Logistic regression
        lr_acc, lr_auc, lr_clf, lr_train_scores = logistic_probe(
            train_states, train_labels
        )

        # Score test set with each method
        dim_test_scores = test_states @ dim_dir
        pca_test_scores = test_states @ pca_dir
        lr_test_scores = lr_clf.predict_proba(test_states)[:, 1]

        layer_result = {
            "diff_in_means": {"accuracy": float(dim_acc), "auroc": float(dim_auc)},
            "pca": {"accuracy": float(pca_acc), "auroc": float(pca_auc), "explained_var": float(pca_var)},
            "logistic": {"accuracy": float(lr_acc), "auroc": float(lr_auc)},
            "test_scores": {
                "diff_in_means": dim_test_scores.tolist(),
                "pca": pca_test_scores.tolist(),
                "logistic": lr_test_scores.tolist(),
            },
            "train_scores": {
                "diff_in_means": dim_train_scores.tolist(),
                "pca": pca_train_scores.tolist(),
                "logistic": lr_train_scores.tolist(),
            },
        }

        results["layer_results"][str(layer_idx)] = layer_result

        # Track best layer
        avg_auc = (dim_auc + pca_auc + lr_auc) / 3
        if avg_auc > best_auc:
            best_auc = avg_auc
            best_layer = layer_idx

        if layer_idx % 4 == 0 or layer_idx == n_layers - 1:
            print(f"Layer {layer_idx:2d}: DIM_AUC={dim_auc:.3f} PCA_AUC={pca_auc:.3f} LR_AUC={lr_auc:.3f}")

    results["best_layer"] = best_layer
    results["best_avg_auc"] = float(best_auc)

    # Detailed results for best layer
    print(f"\n=== Best layer: {best_layer} (avg AUC: {best_auc:.3f}) ===")
    bl = results["layer_results"][str(best_layer)]
    print(f"  Diff-in-means: ACC={bl['diff_in_means']['accuracy']:.3f}, AUC={bl['diff_in_means']['auroc']:.3f}")
    print(f"  PCA:           ACC={bl['pca']['accuracy']:.3f}, AUC={bl['pca']['auroc']:.3f}")
    print(f"  Logistic:      ACC={bl['logistic']['accuracy']:.3f}, AUC={bl['logistic']['auroc']:.3f}")

    # Per-category test scores at best layer
    print("\nTest scores at best layer (higher = more offensive per probe):")
    test_scores_dim = np.array(bl["test_scores"]["diff_in_means"])
    test_scores_lr = np.array(bl["test_scores"]["logistic"])

    for cat in ["ai_directed", "ambiguous"]:
        cat_mask = [test_cats[i] == cat for i in range(len(test_cats))]
        cat_dim = test_scores_dim[cat_mask]
        cat_lr = test_scores_lr[cat_mask]
        print(f"  {cat}: DIM mean={cat_dim.mean():.3f} (std={cat_dim.std():.3f}), LR mean={cat_lr.mean():.3f} (std={cat_lr.std():.3f})")

    # Save results
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_PATH}")
    print(f"Total time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
