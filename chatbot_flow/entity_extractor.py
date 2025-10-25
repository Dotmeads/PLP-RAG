"""
Entity Extraction for Pet Chatbot
Extracts pet-related entities using fine-tuned NER model from Hugging Face Hub.
"""

import os
from transformers import pipeline
from .synonyms import SYNONYMS, canonicalize, postprocess_entities

class EntityExtractor:
    def __init__(self, model_repo="kerrringuo/pet-adoption-ner"):
        """Load fine-tuned NER transformer directly from Hugging Face Hub"""
        
        # Load the NER model
        self.ner_pipe = pipeline(
            "ner",
            model=model_repo,
            tokenizer=model_repo,
            aggregation_strategy="simple"
        )

    def extract(self, text):
        """Return dict of canonicalized entity_name: value."""
        if not text or not text.strip():
            return {}

        results = self.ner_pipe(text)
        entities = {}

        # Collect recognized entities
        for r in results:
            ent_type = r["entity_group"]
            val = r["word"].strip()
            entities[ent_type] = val

        # Canonicalize + postprocess
        entities = postprocess_entities(entities)

        # Fallback keyword check (handles cases like "jb")
        lowered_text = text.lower()
        for canon, variants in SYNONYMS.items():
            for v in variants:
                if f" {v.lower()} " in f" {lowered_text} ":
                    # Assign based on what kind of canonical term this is
                    if canon in [
                        "Kuala Lumpur","Selangor","Penang","Johor","Sabah","Sarawak","Perak",
                        "Negeri Sembilan","Melaka","Pahang","Kedah","Kelantan","Terengganu",
                        "Putrajaya","Labuan"
                    ] and "STATE" not in entities:
                        entities["STATE"] = canon
                    break

        return entities


if __name__ == "__main__":
    ner = EntityExtractor()
    tests = [
        "any cute husky puppies around KL?",
        "adopt fluffy white kitten near jb",
        "looking for brown poodle dog in selangor",
        "kl"
    ]
    for t in tests:
        print(f"\nQuery: {t}")
        print("Extracted:", ner.extract(t))