import pickle
import tarfile
import os

def save_pickle(model, path):
    """Save a model to a pickle file."""
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Pickle saved: {path}")

def compress_model(pickle_path, tar_path):
    """Compress the pickle model file to a .tar.gz archive."""
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(pickle_path, arcname=os.path.basename(pickle_path))
    print(f"✅ Model compressed to: {tar_path}")
