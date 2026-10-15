from itertools import repeat
import pickle
import uuid
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors, Word2Vec
from gensim.utils import simple_preprocess
from nltk.tokenize import sent_tokenize
from scipy.linalg import svd


def prepare_tokens(corpus) -> tuple[list[list[str]], list[list[str]]]:
    """Returns (doc_ids, doc_tokens, sentences)"""
    doc_tokens, sentences = [], []
    for text in corpus:
        sentence_tokens = tokenize(text)
        sentences.extend(sentence_tokens)
        doc_tokens.append([tok for sent in sentence_tokens for tok in sent])
    return doc_tokens, sentences


def tokenize(text, min_len=2, max_len=25):
    """
    Sentence split: NLTK's punkt (handles e.g. "Mr." correctly).
    Word tokenize: gensim's simple_preprocess (lowercases, deaccents,
    strips punctuation).
    """
    text = text.strip()
    if not text:
        return []
    sentences = []
    for raw_sentence in sent_tokenize(text):
        tokens = simple_preprocess(raw_sentence, deacc=True, min_len=min_len, max_len=max_len)
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


def mean_difference_axis(
    wv: KeyedVectors,
    left_axis: list[str],
    right_axis: list[str],
) -> np.ndarray:
    """
    ACO-style axis: mean of normalized positive-pole vectors minus mean of
    normalized negative-pole vectors, then normalized. Positive projection
    scores mean "more rational", negative means "more emotional".
    """
    def _filter_to_vocab(words: list[str], wv: KeyedVectors) -> list[str]:
        present = [w for w in words if w in wv]
        missing = sorted(set(words) - set(present))
        if missing:
            print(f"[axis] dropping {len(missing)} seed words not in vocab: {missing}")
        return present


    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


    pos = _filter_to_vocab(left_axis, wv)
    neg = _filter_to_vocab(right_axis, wv)
    if not pos or not neg:
        raise ValueError("Not enough seed words survived vocab filtering to build an axis.")

    pos_vecs = np.stack([_normalize(wv[w]) for w in pos])
    neg_vecs = np.stack([_normalize(wv[w]) for w in neg])
    axis = pos_vecs.mean(axis=0) - neg_vecs.mean(axis=0)
    return _normalize(axis)


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


def compute_scores(corpus, left_axis, right_axis):
    print("Preparing tokens...")
    doc_tokens, sentences = prepare_tokens(corpus.values())
    print("Done.")

    print("Creating Word2Vec model...")
    model = Word2Vec(
        sentences=sentences,
        vector_size=300,
        window=10,
        min_count=5,
        sg=1,
        workers=12,
        epochs=20,
        seed=42,
    )
    print("Done.")

    print("Performing SIF projecting...")
    vectors = sif_project(doc_tokens, corpus.keys(), model.wv)
    print("Done.")

    print("Computing PCA axis...")
    # axis = pca_difference_axis(model.wv, axis_pairs)
    axis = mean_difference_axis(model.wv, left_axis, right_axis)
    print("Done.")

    print("Computing final cosine similarity...")
    projected = {doc_id: [cosine_similarity(vec, axis)] for doc_id, vec in vectors.items()}
    print("Done.")

    return projected


def bootstrap(corpus, left_axis, right_axis, n_runs):
    corpus_sampleds = []
    for _ in range(n_runs):
        idxs = np.random.choice(list(corpus.keys()), size=len(corpus), replace=True)
        corpus_sampled = {idx: corpus[idx] for idx in idxs}
        corpus_sampleds.append(corpus_sampled)

    with ProcessPoolExecutor(max_workers=16) as executor:
        scoress = executor.map(
            compute_scores, corpus_sampleds, repeat(left_axis), repeat(right_axis)
        )

        compiled = {}
        for n, scores in enumerate(scoress):
            print(f"=== Processing results of bootstrap {n} ===")
            for (id, score) in scores.items():
                compiled.setdefault(id, []).append(score)

    return compiled


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

# ANTONYM_PAIRS = [
#     ("therefore", "heartbroken"),   # logical connector vs. direct emotional exclamation
#     ("evidence", "devastated"),     # "the evidence shows..." vs. "families are devastated by..."
#     ("consider", "afraid"),         # "I ask the House to consider..." vs. "people are afraid that..."
#     ("conclude", "hope"),           # "I conclude that..." vs. "I hope the Minister will..."
#     ("analysis", "outraged"),       # "on analysis..." vs. "my constituents are outraged..."
#     ("reason", "grief"),            # "for this reason..." vs. "the grief of these families..."
#     ("objectively", "desperately"), # "objectively speaking..." vs. "we desperately need..."
#     ("data", "suffering"),          # "the data shows..." vs. "people are suffering..."
#     ("assess", "worried"),          # "I have assessed..." vs. "I am worried that..."
#     ("logically", "tragically"),    # "logically, this follows..." vs. "tragically, this failed..."
#     ("fact", "anger"),              # "the fact is..." vs. "the anger in my constituency..."
#     ("balance", "compassion"),      # "on balance..." vs. "I ask this House for compassion"
# ]

df = pd.read_parquet("/mnt/data/cevaw/corpus_1998_to_2025_v8.parquet")
corpus = {row.Index: row.body for row in df.itertuples()}

# scores = bootstrap(corpus, RATIONALITY_WORDS, EMOTION_WORDS, n_runs=32)
scores = compute_scores(corpus, RATIONALITY_WORDS, EMOTION_WORDS)

with open("scores.pkl", "wb") as f:
    pickle.dump(scores, f)

df.loc[list(scores.keys()), "emotionality_EMBEDDING"] = [np.mean(xs) for xs in scores.values()]
df.loc[list(scores.keys()), "emotionality_EMBEDDING_error"] = [np.std(xs) for xs in scores.values()]

ordered = sorted(scores.items(), key=lambda x: np.mean(x[1]))

id, score = ordered[0]
print(f"ID={id} score={score}")
print()
print(df.loc[id]["body"])
print()

print("==============")

id, score = ordered[-1]
print(f"ID={id} score={score}")
print()
print(df.loc[id]["body"])
print()