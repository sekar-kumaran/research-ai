import os
import hashlib
import json

def get_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

artifacts_dir = "hf_full/artifacts"
manifest = []

expected_artifacts = {
    "classifier.joblib": {"type": "model", "role": "Classification inference"},
    "tfidf_vectorizer.joblib": {"type": "vectorizer", "role": "Text preprocessing"},
    "labels.joblib": {"type": "mapping", "role": "Label decoding"},
    "paper_index.faiss": {"type": "index", "role": "Similarity search"},
    "paper_metadata.parquet": {"type": "metadata", "role": "Document lookup"},
    "embedding_model_name.joblib": {"type": "config", "role": "Embedding model name"}
}

for filename, info in expected_artifacts.items():
    filepath = os.path.join(artifacts_dir, filename)
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        sha256 = get_sha256(filepath)
        manifest.append({
            "filename": filename,
            "size": size,
            "SHA256": sha256,
            "type": info["type"],
            "expected_runtime_role": info["role"]
        })

with open(os.path.join(artifacts_dir, "MANIFEST.json"), "w") as f:
    json.dump(manifest, f, indent=4)

print("Manifest created.")
