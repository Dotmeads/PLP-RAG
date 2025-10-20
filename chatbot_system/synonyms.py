# src/synonyms.py
import re
from rapidfuzz import process, fuzz

SYNONYMS = {
    # === Breeds ===
    "Siberian Husky": ["husky", "sib husky"],
    "Golden Retriever": ["retriever"],
    "Labrador Retriever": ["lab", "labrador"],
    "Poodle": ["toy poodle", "mini poodle", "poodlee"],
    "Shiba Inu": ["shiba"],
    "Beagle": ["begle"],
    "Chihuahua": ["chi", "chihua hua"],
    "German Shepherd": ["shepherd", "german sheperd"],
    "Bulldog": ["english bulldog", "bulldogge"],
    "Pug": ["puggie"],
    "Rottweiler": ["rottie", "rotweiller"],
    "Dalmatian": ["dalmation", "spotted"],
    "Jack Russell Terrier": ["jack russell"],
    "Cocker Spaniel": ["cocker", "spaniel"],
    "Doberman": ["dobie", "dobermann"],
    "Mixed Breed": ["mix", "mixed", "mongrel", "stray", "rescue dog"],
    "British Shorthair": ["british short hair", "british blue"],
    "Bengal": ["leopard", "spotted"],
    "Scottish Fold": ["scottish fold"],
    "Abyssinian": ["abyssinian"],
    "Sphynx": ["hairless"],

    # === Colors ===
    "black": ["dark", "jet black", "charcoal", "hitam"],
    "white": ["off white", "ivory", "snowy", "putih"],
    "brown": ["brown", "chocolate", "coffee", "mocha", "coklat", "espresso"],
    "golden": ["light brown", "tan", "yellowish", "gold"],
    "gray": ["grey", "silver", "ash", "smokey"],
    "orange": ["ginger", "orangey"],
    "cream": ["beige", "light yellow"],
    "yellow": ["mustard", "pale gold"],
    "blue": ["bluish grey"],
    "red": ["reddish", "rust"],
    "tabby": ["striped", "tiger pattern"],
    "calico": ["tri color", "patchy"],
    "tortoiseshell": ["tortie"],

    # === Gender ===
    "male": ["boy", "boi", "m"],
    "female": ["girl", "gurl", "f"],

    # === Pet type ===
    "dog": ["dog", "dogs", "pup", "puppy", "puppies", "doggo"],
    "cat": ["cat", "cats", "kitty", "kitten", "kittens", "feline"],

    # === Fur length ===
    "short fur": ["short hair"],
    "medium fur": ["medium hair"],
    "long fur": ["long hair", "fluffy"],

    # === Age ===
    "baby": ["infant", "newborn", "kitten", "puppy", "puppies"],
    "young": ["juvenile", "teen"],
    "adult": ["grown"],
    "senior": ["old", "elderly"],

    # === States / locations ===
    "Kuala Lumpur": ["kl", "city centre", "k.l."],
    "Selangor": ["selangor"],
    "Penang": ["pulau pinang", "pg"],
    "Johor": ["jb", "johore"],
    "Sabah": ["kk", "kota kinabalu"],
    "Sarawak": ["kuching", "swk"],
    "Perak": ["ipoh", "perk"],
    "Negeri Sembilan": ["n9", "seremban"],
    "Melaka": ["malacca", "mlk"],
    "Pahang": ["kuantan", "phg"],
    "Kedah": ["alor setar"],
    "Kelantan": ["kota bharu", "ktn"],
    "Terengganu": ["kuala terengganu", "tgg"],
    "Putrajaya": ["putra jaya"],
    "Labuan": ["labuan"]
}


def canonicalize(text, threshold=85):
    """Return canonical form if synonym or fuzzy match found."""
    if not text:
        return text

    # Normalize: lowercase, strip punctuation and extra spaces
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)   # remove ?,.,!, etc.
    text = text.replace("_", " ")

    # Exact and synonym match
    for canon, variants in SYNONYMS.items():
        all_variants = [canon.lower()] + [v.lower() for v in variants]
        if text in all_variants:
            return canon

    # Fuzzy fallback across all variants
    all_terms = {canon: [canon] + vals for canon, vals in SYNONYMS.items()}
    flat = [(canon, variant.lower()) for canon, variants in all_terms.items() for variant in variants]
    best_match = process.extractOne(text, [v for _, v in flat], scorer=fuzz.token_sort_ratio)
    if best_match and best_match[1] >= threshold:
        for canon, variant in flat:
            if variant == best_match[0]:
                return canon

    return text


def postprocess_entities(entities):
    """Normalize synonyms and handle derived mappings like kitten → cat + baby."""
    ents = dict(entities)

    pet_type = ents.get("PET_TYPE", "").lower()
    if pet_type in ["kitten", "kittens"]:
        ents["PET_TYPE"] = "cat"
        ents["AGE"] = "baby"
    elif pet_type in ["puppy", "puppies", "pup"]:
        ents["PET_TYPE"] = "dog"
        ents["AGE"] = "baby"

    # Canonicalize all values
    ents = {k: canonicalize(v) for k, v in ents.items()}
    return ents