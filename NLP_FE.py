"""
NLP Feature Extraction Pipeline

This script performs preprocessing, linguistic feature extraction, and TF-IDF feature computation
on a dataset of articles (assumed to have 'Title' and 'Text' fields). The results are stored
in a CSV file for further use in tasks like classification or analysis.
"""

# === Imports ===
import re
import spacy
import textstat
from spacytextblob.spacytextblob import SpacyTextBlob
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker
from collections import Counter
from itertools import islice
from sklearn.feature_extraction.text import TfidfVectorizer

# === Load spaCy model and sentiment analyzer ===
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe('spacytextblob')


# === Preprocessing Functions ===

def preprocess(text):
    """
    Lowercases, removes non-letter characters, tokenizes, removes stopwords and stems words.

    Args:
        text (str): Raw input text.

    Returns:
        str: Preprocessed text.
    """
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)


# === NLP Feature Extraction Functions ===

def pos_tagging(doc):
    """Returns a list of (token, POS-tag) pairs from the document."""
    return [(token.text, token.pos_) for token in doc]

def name_entity(doc):
    """Returns named entities with their labels."""
    return [(ent.text, ent.label_) for ent in doc.ents]

def freq_pos_tagging(doc):
    """
    Returns frequency (normalized by word count) of NOUN, VERB, ADJ tags.

    Args:
        doc (spacy.Doc): Parsed text.

    Returns:
        dict: Normalized counts of selected POS tags.
    """
    words = [token for token in doc if not token.is_punct and not token.is_space]
    num_words = len(words)
    pos_counts = {"NOUN": 0, "VERB": 0, "ADJ": 0}
    for token in words:
        if token.pos_ in pos_counts:
            pos_counts[token.pos_] += 1
    return {pos: count / num_words if num_words else 0 for pos, count in pos_counts.items()}

def freq_name(p):
    """
    Counts occurrences of specific named entity types.

    Args:
        p (list): List of (entity_text, label) tuples.

    Returns:
        dict: Counts of specified entity types.
    """
    counts = {k: 0 for k in ['ORG', 'GPE', 'DATE', 'PERSON', 'CARDINAL', 'LOC', 'QUANTITY', 'FAC']}
    for _, name in p:
        if name in counts:
            counts[name] += 1
    return {k if k != "CARDINAL" else "CARD": v for k, v in counts.items()}

def emotion(doc):
    """Returns sentiment polarity using TextBlob extension."""
    return [doc._.blob.polarity]

# === Text Complexity and Lexical Features ===

def complexity(doc):
    """
    Calculates various complexity metrics for the text.

    Returns:
        Tuple: (num_sentences, num_words, unique_ratio, total_chars, avg_word_len, avg_sent_len)
    """
    sentences = [sent.text for sent in doc.sents if sent.text.strip()]
    num_sentences = len(sentences)
    words = [token.text for token in doc if not token.is_punct and not token.is_space]
    num_words = len(words)
    unique_words = set(word.lower() for word in words)
    ratio_unique_words = len(unique_words) / num_words
    total_chars = sum(len(word) for word in words)
    avg_word_length_chars = total_chars / num_words if num_words > 0 else 0
    avg_sentence_length = num_words / num_sentences if num_sentences > 0 else 0
    return num_sentences, num_words, ratio_unique_words, total_chars, avg_word_length_chars, avg_sentence_length

def num_sen(doc): return len([sent.text for sent in doc.sents if sent.text.strip()])

def num_words(doc): return len([token.text for token in doc if not token.is_punct and not token.is_space])

def unique_words_ratio(doc):
    words = [token.text for token in doc if not token.is_punct and not token.is_space]
    return len(set(w.lower() for w in words)) / len(words)

def avg_word_length(doc):
    words = [token.text for token in doc if not token.is_punct and not token.is_space]
    return sum(len(word) for word in words) / len(words)

def avg_sentence_len(doc):
    words = [token.text for token in doc if not token.is_punct and not token.is_space]
    sentences = [sent.text for sent in doc.sents if sent.text.strip()]
    return len(words) / len(sentences) if sentences else 0

def total_charss(doc):
    return sum(len(token.text) for token in doc if not token.is_punct and not token.is_space)

def pos_distribution(doc):
    """
    Prints the normalized distribution of all POS tags in the document.
    """
    words = [token.text for token in doc if not token.is_punct and not token.is_space]
    pos_counts = {}
    for token in doc:
        if not token.is_punct and not token.is_space:
            pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
    num_words = len(words)
    pos_distribution = {pos: count / num_words for pos, count in pos_counts.items()} if num_words else {}
    print(pos_distribution)

# === Readability and Spelling Features ===

def syllables_in_word(word):
    """Returns the number of syllables in a word."""
    return textstat.syllable_count(word) if isinstance(word, str) and word.strip() else 0

def complex_word(doc, min_syllables=3):
    """Returns ratio of complex words (with >= min_syllables) in the document."""
    num_words = len(doc)
    return sum(1 for token in doc if token.is_alpha and syllables_in_word(token.text) >= min_syllables) / num_words

def avg_word_length_syllables(doc):
    """Average number of syllables per word."""
    total_syllables = 0
    word_count = 0
    for token in doc:
        if token.is_alpha:
            total_syllables += syllables_in_word(token.text)
            word_count += 1
    return total_syllables / word_count if word_count else 0

def error_analysis(text, lang='en'):
    """
    Calculates spelling error ratio in the text.

    Args:
        text (str): Raw text input.
        lang (str): Language for spellchecking.

    Returns:
        float: Misspelled word ratio.
    """
    if not text.strip():
        return 0.0
    spell = SpellChecker(language=lang)
    words = text.lower().split()
    total_words = 0
    misspelled_count = 0
    for word in words:
        clean_word = "".join(char for char in word if char.isalpha())
        if clean_word:
            total_words += 1
            if clean_word != spell.correction(clean_word):
                misspelled_count += 1
    return misspelled_count / total_words if total_words else 0.0

# === POS N-gram Features ===

def pos_sequence_patterns(doc, n=3, top_k=10):
    """
    Returns most common POS n-grams in the document.

    Args:
        n (int): Length of POS n-grams.
        top_k (int): Number of top n-grams to return.
    """
    pos_tags = [token.pos_ for token in doc if not token.is_punct and not token.is_space]
    ngrams = zip(*(islice(pos_tags, i, None) for i in range(n)))
    ngram_list = [" ".join(ngram) for ngram in ngrams]
    return Counter(ngram_list).most_common(top_k)

def pos_sequence_features(doc, n=3, patterns=None):
    """
    Returns frequency of specified POS n-gram patterns in the document.

    Args:
        patterns (list): List of POS patterns (e.g. ["NOUN VERB NOUN"]).

    Returns:
        dict: POS n-gram counts.
    """
    pos_tags = [token.pos_ for token in doc if not token.is_punct and not token.is_space]
    ngrams = zip(*(islice(pos_tags, i, None) for i in range(n)))
    ngram_list = [" ".join(ngram) for ngram in ngrams]
    counter = Counter(ngram_list)
    if patterns is None:
        patterns = ["NOUN VERB NOUN", "ADJ NOUN VERB", "VERB ADJ NOUN"]
    return {f"ngram_{p.replace(' ', '_')}": counter.get(p, 0) for p in patterns}


# === Main Feature Extraction Driver ===

def extract_features(article):
    """
    Main function to extract features from a single article row.

    Args:
        article (dict): Row from input dataframe with 'Title' and 'Text'.

    Returns:
        dict: All extracted features.
    """
    title = article.get("Title", "")
    text_body = article.get("Text", "")
    if pd.isna(title): title = ""
    if pd.isna(text_body): text_body = ""
    text = f"{title}\n{text_body}"
    print(article['Index'])

    doc = nlp(text)
    entities = name_entity(doc)
    pos_ngrams = pos_sequence_features(doc)
    entity_counts = freq_name(entities)
    pos_ratios = freq_pos_tagging(doc)

    return {
        **pos_ratios,
        **entity_counts,
        **pos_ngrams,
        "unique_words_ratio": unique_words_ratio(doc),
        "avg_word_length": avg_word_length(doc),
        "avg_sentence_length": avg_sentence_len(doc),
        "total_chars": total_charss(doc),
        "emotion": emotion(doc)[0],
        "spelling_error_ratio": error_analysis(text),
        "avg_word_len_syllables": avg_word_length_syllables(doc),
        "complex_word_ratio": complex_word(doc)
    }


# === Run the Feature Extraction Pipeline ===

# Load articles
articles = pd.read_csv("climate_articles.csv")
# Extract NLP Features for each article
df = pd.DataFrame([
    {
        "Index": i,
        "Label": row['label'],
        **extract_features(row)
    }
    for i, (_, row) in enumerate(articles.iterrows())
])

# Preprocess for TF-IDF
articles['cleaned_text'] = articles['Text'].fillna('').apply(preprocess)

# Load target vocabulary for TF-IDF
for_vocab = pd.read_csv("features.csv")
kes = [x.split("_")[1] for x in for_vocab.keys() if "tfidf" in x]
vocab = {word: i for i, word in enumerate(kes)}

# Fit and transform TF-IDF
vectorizer = TfidfVectorizer(vocabulary=vocab)
vectorizer.fit(articles['cleaned_text'])
TF_features = vectorizer.transform(articles['cleaned_text'])
TF_df = pd.DataFrame(TF_features.toarray(), columns=[f"tfidf_{word}" for word in vectorizer.get_feature_names_out()])

# Combine all features
combined_df = pd.concat([df.reset_index(drop=True), TF_df.reset_index(drop=True)], axis=1)

# Save to CSV
combined_df.to_csv("feature_a.csv", index=False)
