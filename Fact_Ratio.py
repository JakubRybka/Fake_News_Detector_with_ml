"""
Knowledge Base Fact Matching Script

This script computes the semantic similarity between each sentence in an article and a predefined
set of factual knowledge base entries. It adds a new feature to the existing article dataset
that represents the ratio of article sentences that match any fact from the knowledge base
with similarity above a given threshold.

Dependencies:
- sentence-transformers
- pandas
"""

# === Imports ===
from sentence_transformers import SentenceTransformer, util
import pandas as pd

# === Load Sentence Embedding Model ===
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight yet effective embedding model

# === Load Knowledge Base ===
kb = pd.read_csv("know_base_new.csv", sep=";")  # Expected to contain a 'fact' column
kb_facts = list(kb['fact'])  # Extract factual statements
kb_embeddings = model.encode(kb_facts, convert_to_tensor=True)  # Encode all facts once for efficiency


# === Function: Check for Fact Matches ===
def check_kb(text, threshold=0.6):
    """
    Checks which sentences in a given text semantically match any knowledge base fact.

    Args:
        text (str): Input text to analyze.
        threshold (float): Cosine similarity threshold for determining a match.

    Returns:
        list of tuples: Matched sentences and their similarity scores.
    """
    sentences = text.split('.')
    results = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or '"' in sentence or "'" in sentence:
            continue  # Skip empty or quoted sentences (e.g., dialogue, quotes)

        sent_embedding = model.encode(sentence, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(sent_embedding, kb_embeddings)[0]  # Similarity to all KB facts

        for idx, score in enumerate(cosine_scores):
            if score.item() >= threshold:
                results.append((sentence, round(score.item(), 3)))  # Store matched sentence and its score

    return results


# === Load Existing Feature Set and Article Texts ===
df = pd.read_csv("feature_a.csv")  # Contains previously extracted features
articles = pd.read_json("without_assessment.jsonl", lines=True)  # Original articles

# === Compute Fact Match Ratio ===
fact_ratios = []  # Stores the final fact ratio feature

for i, (_, row) in enumerate(articles.iterrows()):
    print(i)
    text = row.get("text", "")

    if not isinstance(text, str) or not text.strip():
        fact_ratios.append(0)
        continue

    sentences = [s for s in text.split('.') if s.strip()]
    matches = check_kb(text)

    # Calculate ratio of matched sentences to total sentences
    ratio = len(matches) / len(sentences) if sentences else 0
    fact_ratios.append(ratio)

# === Save New Feature ===
df['Fact_ratio'] = fact_ratios
df.to_csv("feature_a2.csv", index=False)
