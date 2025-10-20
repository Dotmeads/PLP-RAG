import os
import json
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer


class IntentClassifier:
    def __init__(self, model_path=None):
        """Load the intent classifier and supporting metadata."""

        if model_path is None:
            # Go up two levels from chatbot_system/ to project root, then to models/
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_dir = os.path.join(base_dir, "models", "intent")
            model_path = os.path.join(model_dir, "minilm_logreg.joblib")
            meta_path = os.path.join(model_dir, "minilm_meta.json")
            labels_path = os.path.join(model_dir, "minilm_labels.json")

        # Validate all required files
        for p in [model_path, meta_path, labels_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing file: {p}")

        # Load model, metadata, and labels
        self.clf = joblib.load(model_path)

        with open(meta_path, "r") as f:
            self.meta = json.load(f)
        with open(labels_path, "r") as f:
            self.labels = json.load(f)

        # Initialize encoder
        embedding_model = self.meta.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        self.encoder = SentenceTransformer(embedding_model)

        # Confidence threshold (default 0.7)
        self.threshold = float(self.meta.get("threshold", 0.7))

    def predict(self, text):
        """Return (intent, confidence) for input text."""
        if not text or not text.strip():
            return "unknown", 0.0

        emb = self.encoder.encode([text])
        probs = self.clf.predict_proba(emb)[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        intent = self.labels[idx] if conf >= self.threshold else "unknown"
        return intent, conf


if __name__ == "__main__":
    clf = IntentClassifier()
    test_msgs = [
        "I want to adopt a puppy near KL",
        "How to train my dog?",
        "bye bye see you later",
        "hi i wan a doog"
    ]
    for msg in test_msgs:
        intent, conf = clf.predict(msg)
        print(f"{msg} → {intent} ({conf:.2f})")