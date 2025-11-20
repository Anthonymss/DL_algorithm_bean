import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.manifold import TSNE
import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.titlesize'] = 11
matplotlib.rcParams['axes.labelsize'] = 10
matplotlib.rcParams['legend.fontsize'] = 9
matplotlib.rcParams['xtick.labelsize'] = 9
matplotlib.rcParams['ytick.labelsize'] = 9
# CONFIG
EVAL_DIR = "outputs/evaluation"
METRICS_JSON = os.path.join(EVAL_DIR, "metrics_summary.json")
EPOCH_CSV = "outputs/models/epoch_metrics.csv"

OUT_DIR = "outputs/plots"
PNG_DIR = os.path.join(OUT_DIR, "png")
SVG_DIR = os.path.join(OUT_DIR, "svg")
PDF_DIR = os.path.join(OUT_DIR, "pdf")

STAGE1_EPOCHS = 25 

os.makedirs(PNG_DIR, exist_ok=True)
os.makedirs(SVG_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)

def save_figure(fig, name):
    fig.savefig(os.path.join(PNG_DIR, f"{name}.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(SVG_DIR, f"{name}.svg"), bbox_inches="tight")
    fig.savefig(os.path.join(PDF_DIR, f"{name}.pdf"), bbox_inches="tight")
    print(f"✔ Saved: {name} [png/svg/pdf]")

def load_all():
    print("Loading evaluation and training data...")

    with open(METRICS_JSON, "r", encoding="utf8") as f:
        metrics = json.load(f)

    df_epochs = pd.read_csv(EPOCH_CSV)

    y_true = np.load(os.path.join(EVAL_DIR, "y_true.npy"))
    y_pred = np.load(os.path.join(EVAL_DIR, "y_pred.npy"))
    y_prob = np.load(os.path.join(EVAL_DIR, "y_prob.npy"))

    images = np.load(os.path.join(EVAL_DIR, "images.npy"))
    embeddings = np.load(os.path.join(EVAL_DIR, "embeddings.npy"))
    cm = np.load(os.path.join(EVAL_DIR, "confusion_matrix.npy"))

    return metrics, df_epochs, y_true, y_pred, y_prob, images, embeddings, cm

def plot_accuracy(df_epochs):
    acc_col = "accuracy" if "accuracy" in df_epochs.columns else "acc"
    val_acc_col = "val_accuracy" if "val_accuracy" in df_epochs.columns else "val_acc"

    fig, ax = plt.subplots(figsize=(3.5, 2.2)) 

    ax.plot(df_epochs["epoch"], df_epochs[acc_col],
            marker="o", markersize=1.5, linewidth=0.9, label="Train")
    ax.plot(df_epochs["epoch"], df_epochs[val_acc_col],
            marker="s", markersize=1.5, linewidth=0.9, label="Validation")

    ax.axvline(STAGE1_EPOCHS, linestyle="--", color="red", linewidth=0.9)

    ax.set_title("Accuracy Evolution", fontsize=9)
    ax.set_xlabel("Epoch", fontsize=8)
    ax.set_ylabel("Accuracy", fontsize=8)

    ax.tick_params(axis="both", labelsize=7)

    ax.grid(True, linewidth=0.45)

    ax.legend(fontsize=6.5, frameon=True, edgecolor="gray")

    fig.tight_layout()
    save_figure(fig, "accuracy_curve")

def plot_loss(df_epochs):
    fig, ax = plt.subplots(figsize=(3.5, 2.2))

    ax.plot(df_epochs["epoch"], df_epochs["loss"],
            marker="o", markersize=1.5, linewidth=0.9, label="Train")
    ax.plot(df_epochs["epoch"], df_epochs["val_loss"],
            marker="s", markersize=1.5, linewidth=0.9, label="Validation")

    ax.axvline(STAGE1_EPOCHS, linestyle="--", color="red", linewidth=0.9)

    ax.set_title("Loss Evolution", fontsize=9)
    ax.set_xlabel("Epoch", fontsize=8)
    ax.set_ylabel("Loss", fontsize=8)

    ax.tick_params(axis="both", labelsize=7)

    ax.grid(True, linewidth=0.45)

    ax.legend(fontsize=6.5, frameon=True, edgecolor="gray")

    fig.tight_layout()
    save_figure(fig, "loss_curve")

def plot_val_f1(df_epochs):
    if "val_f1" not in df_epochs.columns:
        print("⚠ val_f1 column not found. Skipping F1 plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df_epochs["epoch"], df_epochs["val_f1"], marker="o")
    ax.axvline(STAGE1_EPOCHS, linestyle="--", color="red")
    ax.set_title("Validation F1-score Evolution")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1-score")
    ax.grid(True)

    save_figure(fig, "training_f1")

def plot_learning_rate(df_epochs):
    if "lr" not in df_epochs.columns:
        print("⚠ lr column not found. Skipping LR plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df_epochs["epoch"], df_epochs["lr"], marker="o")
    ax.set_title("Learning Rate per Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_yscale("log")
    ax.grid(True)

    save_figure(fig, "learning_rate")

def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)

    ax.set_title("Confusion Matrix")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")

    save_figure(fig, "confusion_matrix")

def plot_f1_per_class(metrics):
    f1_dict = metrics["classification_report"]
    classes = metrics["classes"]

    f1_vals = [f1_dict[c]["f1-score"] for c in classes]

    bar_color = "#003366"
    text_color = "black"
    
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(classes, f1_vals, color=bar_color)

    for bar, val in zip(bars, f1_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f"{val:.4f}", ha="center", color=text_color, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title("F1-score per Class", fontsize=12, fontweight="bold")
    ax.set_ylabel("F1-score", fontsize=11)
    ax.set_xlabel("Classes", fontsize=11)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    save_figure(fig, "f1_per_class")


def plot_roc_all(y_true, y_prob, class_names):
    fig, ax = plt.subplots(figsize=(7, 6))

    for i, cname in enumerate(class_names):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_prob[:, i])
        auc_score = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{cname} (AUC={auc_score:.3f})")

    ax.plot([0, 1], [0, 1], "k--")
    ax.set_title("ROC Curve – All Classes")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(True)

    save_figure(fig, "roc_all_classes")

def plot_pr_all(y_true, y_prob, class_names):
    fig, ax = plt.subplots(figsize=(7, 6))

    for i, cname in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(
            (y_true == i).astype(int), y_prob[:, i]
        )
        ax.plot(recall, precision, label=cname)

    ax.set_title("Precision–Recall Curve – All Classes")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    ax.grid(True)

    save_figure(fig, "pr_all_classes")

def plot_tsne(embeddings, y_true, class_names):
    tsne = TSNE(n_components=2, perplexity=40, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1],
                         c=y_true, cmap="tab10", alpha=0.6)

    legend = ax.legend(handles=scatter.legend_elements()[0],
                       labels=class_names, title="Classes")

    ax.add_artist(legend)
    ax.set_title("t-SNE Embeddings")

    save_figure(fig, "tsne_embeddings")

def main():
    (metrics, df_epochs, y_true, y_pred,
     y_prob, images, embeddings, cm) = load_all()

    print("Generating training plots...")
    plot_accuracy(df_epochs)
    plot_loss(df_epochs)
    plot_val_f1(df_epochs)
    plot_learning_rate(df_epochs)


    print("Generating performance plots...")
    plot_confusion_matrix(cm, metrics["classes"])
    plot_f1_per_class(metrics)
    plot_roc_all(y_true, y_prob, metrics["classes"])
    plot_pr_all(y_true, y_prob, metrics["classes"])
    plot_tsne(embeddings, y_true, metrics["classes"])

    print("\n✔ All plots successfully generated!")


if __name__ == "__main__":
    main()
