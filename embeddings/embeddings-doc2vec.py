import re
from collections import Counter

import numpy as np
import pandas as pd
from gensim.models import Doc2Vec, KeyedVectors
from gensim.models.doc2vec import TaggedDocument
from nltk.corpus import stopwords as nltk_stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from scipy.linalg import svd


def prepare_tokens(corpus) -> tuple[list[str], list[list[str]], list[list[str]]]:
    """Returns (doc_ids, doc_tokens, sentences)"""
    doc_ids, doc_tokens, sentences = [], [], []
    for doc_id, text in corpus:
        doc_ids.append(doc_id)
        sentence_tokens = tokenize_nltk(text)
        sentences.extend(sentence_tokens)
        doc_tokens.append([tok for sent in sentence_tokens for tok in sent])
    return doc_ids, doc_tokens, sentences


_WORD_RE = re.compile(r"^[a-z]+$")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def tokenize_nltk(text, remove_stopwords=False, min_len=2):
    """
    Sentence split: NLTK's punkt model (statistical, handles abbreviations
    like "Mr." better than a punctuation regex).
    Word tokenize: NLTK's word_tokenize, lowercased, with punctuation-only
    tokens stripped to match tokenize_gensim's behaviour as closely as
    possible.

    Requires: nltk.download("punkt_tab"); nltk.download("stopwords")
    """
    text = text.strip()
    if not text:
        return []

    stop = (
        frozenset(nltk_stopwords.words("english")) if remove_stopwords else frozenset()
    )

    sentences = []
    for raw_sentence in sent_tokenize(text):
        raw_tokens = word_tokenize(raw_sentence.lower())
        tokens = [t for t in raw_tokens if _WORD_RE.match(t) and len(t) >= min_len]
        if remove_stopwords:
            tokens = [t for t in tokens if t not in stop]
        if tokens:
            sentences.append(tokens)
    return sentences


def sif_project(doc_tokens, doc_ids, wv, a=1e-3, remove_top_component=True):
    """
    SIF (smooth inverse frequency) weighted average of word vectors.

    weight(w) = a / (a + p(w)), where p(w) is w's relative frequency
    across doc_tokens. Rare words get weighted up, common words down.

    If remove_top_component, the top principal component across all
    resulting document vectors is subtracted out (via scipy.linalg.svd)
    -- this removes a shared "dominant direction" that
    would otherwise dominate every document vector and wash out the
    signal you actually want to project.

    doc_tokens: list of token lists, one per document.
    doc_ids:    parallel list of document identifiers.
    wv:         a gensim KeyedVectors, e.g. w2v.wv.

    Returns: {doc_id: np.ndarray}
    """
    word_freq = Counter()
    for tokens in doc_tokens:
        word_freq.update(tokens)
    total_tokens = sum(word_freq.values())

    def weighted_average(tokens):
        vecs, weights = [], []
        for tok in tokens:
            if tok in wv:
                vecs.append(wv[tok])
                p_w = word_freq[tok] / total_tokens
                weights.append(a / (a + p_w))
        if not vecs:
            return np.zeros(wv.vector_size)
        vecs = np.stack(vecs)
        weights = np.array(weights).reshape(-1, 1)
        return (vecs * weights).sum(axis=0) / weights.sum()

    raw_vectors = {
        doc_id: weighted_average(tokens) for doc_id, tokens in zip(doc_ids, doc_tokens)
    }

    if not remove_top_component or len(raw_vectors) <= 1:
        return raw_vectors

    matrix = np.stack(list(raw_vectors.values()))
    centered = matrix - matrix.mean(axis=0)
    _, _, vt = svd(centered, full_matrices=False)
    top_component = vt[0]
    projections = matrix @ top_component
    adjusted = matrix - np.outer(projections, top_component)
    return {doc_id: adjusted[i] for i, doc_id in enumerate(raw_vectors.keys())}


def pca_difference_axis(
    wv: KeyedVectors,
    pairs: list[tuple[str, str]],
) -> np.ndarray:
    """
    Bolukbasi-style axis: stack (positive - negative) difference vectors for
    several antonym pairs, then take the first right-singular vector of the
    (mean-centered) stack via SVD as the axis direction. Sign is oriented
    so that it points toward the "positive" (rationality) pole on average.
    """

    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    diffs = []
    for pos_w, neg_w in pairs:
        if pos_w not in wv or neg_w not in wv:
            print(f"[axis] skipping pair ({pos_w}, {neg_w}) -- not both in vocab")
            continue
        diffs.append(_normalize(wv[pos_w]) - _normalize(wv[neg_w]))
    if len(diffs) < 2:
        raise ValueError("Need at least 2 valid antonym pairs for the PCA-based axis.")

    diff_matrix = np.stack(diffs)  # shape (n_pairs, dim)
    centered = diff_matrix - diff_matrix.mean(axis=0)
    # SVD of the centered matrix: rows of Vt are the principal directions
    _, _, vt = svd(centered, full_matrices=False)
    axis = vt[0]

    # orient the sign: axis should point toward the positive pole on average
    mean_diff = diff_matrix.mean(axis=0)
    if np.dot(axis, mean_diff) < 0:
        axis = -axis
    return _normalize(axis)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


df = pd.read_parquet("/mnt/data/cevaw/corpus_1998_to_2025_v8.parquet")
df = df.query("`interject` == '0' and `body`.str.len() > 500")
documents = list(enumerate([cell for cell in df["body"]]))


print("Preparing tokens...")
doc_ids, doc_tokens, sentences = prepare_tokens(documents)
print("Done.")


# Setting the dm=1 (PV-DM) is actually essential to make the later pca_difference_axs() function
# work. This is necessary to make the vector space into a word space (and not a full document space).
# Otherwise, we'd have to use full speeches as the extreme examples to discover the axis.

print("Creating Doc2Vec model...")
tagged = [
    TaggedDocument(words=tokens, tags=[doc_id])
    for tokens, doc_id in zip(doc_tokens, doc_ids)
]
model = Doc2Vec(
    documents=tagged,
    vector_size=300,
    window=10,
    min_count=5,
    dm=1,  # PV-DM; set dm=0 for PV-DBOW
    workers=8,
    epochs=20,
    seed=42,
)
print("Done.")

print("Extracting vectors...")
vectors = {
    doc_id: model.dv[doc_id] for doc_id in doc_ids
}
# vectors = {}
# for tokens, doc_id in zip(doc_tokens, doc_ids):
#     if doc_id in model.dv:
#         vectors[doc_id] = model.dv[doc_id]
#     else:
#         vectors[doc_id] = model.infer_vector(tokens)
print("Done.")

RATIONALITY_WORDS = [
    "reason",
    "logic",
    "logical",
    "evidence",
    "rational",
    "analysis",
    "therefore",
    "consider",
    "fact",
    "argument",
    "conclude",
    "objective",
    "assess",
    "data",
    "reasonable",
]

EMOTION_WORDS = [
    "feel",
    "fear",
    "hope",
    "worry",
    "angry",
    "passion",
    "heart",
    "emotional",
    "hurt",
    "love",
    "grief",
    "anxious",
    "compassion",
    "outrage",
    "heartbroken",
]

PAIRS = [(fst, snd) for fst in RATIONALITY_WORDS for snd in EMOTION_WORDS]

print("Computing PCA axis...")
axis = pca_difference_axis(model.wv, PAIRS)
print("Done.")

print("Computing final cosine similarity...")
projected = {doc_id: cosine_similarity(vec, axis) for doc_id, vec in vectors.items()}
print("Done.")

ordered = sorted(projected.items(), key=lambda i: i[1])


id, score = ordered[0]
print(f"ID={id} score={score}")
print()
print(documents[id][1])
print()

print("==============")

id, score = ordered[-1]
print(f"ID={id} score={score}")
print()
print(documents[id][1])
print()
