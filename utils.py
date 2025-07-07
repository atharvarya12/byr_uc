import pickle
import tarfile
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc

def save_pickle(obj, filename):
    """Save an object to a .pkl file."""
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
    print(f"✅ Saved pickle: {filename}")

def load_pickle(filename):
    """Load an object from a .pkl file."""
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    print(f"📦 Loaded pickle: {filename}")
    return obj

def compress_model(input_path, output_path):
    """Compress a file using tar.gz format."""
    with tarfile.open(output_path, "w:gz") as tar:
        arcname = os.path.basename(input_path)
        tar.add(input_path, arcname=arcname)
    print(f"📦 Compressed model to: {output_path}")

def save_classification_report(y_true, y_pred, model_name, output_dir="reports"):
    """Save classification report to a text file."""
    report = classification_report(y_true, y_pred)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{model_name}_classification_report.txt")
    with open(path, 'w') as f:
        f.write(report)
    print(f"📊 Classification report saved: {path}")
    return report

def save_roc_curve(y_true, y_probs, model_name, output_dir="reports"):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{model_name}_roc_curve.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"📈 ROC curve saved: {plot_path}")
    return roc_auc

def save_metadata(metadata_dict, filename="models/metadata.json"):
    """Save model metadata (e.g. params, scores) to JSON."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(metadata_dict, f, indent=4)
    print(f"📘 Metadata saved: {filename}")
