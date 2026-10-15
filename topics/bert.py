import os
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import MaximalMarginalRelevance


# =====================================================================
# 0. YOUR DATA
# =====================================================================
# IMPORTANT: feed RAW documents here -- NOT your Gensim-preprocessed,
# stop-word-stripped, Porter-stemmed token lists. The embedding model
# needs real sentences with words intact. Any word cleanup happens later,
# in the vectorizer stage (section 3), never here.
#
#   documents = [open(p).read() for p in my_files]   # e.g.

df = pd.read_parquet("/mnt/data/cevaw/corpus_1998_to_2025_v8.parquet")
df = df.query("`body`.str.len() > 250")

documents = [cell for cell in df["body"]]

print(f"{len(documents)} documents in total")

# ---------------------------------------------------------------------
# 0a. Document identifiers -- the dataframe's index
# ---------------------------------------------------------------------
# The .query() filter above preserves the original index (it only drops
# rows, never renumbers them), so df.index.tolist() lines up positionally
# with `documents` and is a stable handle back to each row of the
# original parquet.
doc_ids = df.index.tolist()

assert len(doc_ids) == len(documents), (
    "doc_ids and documents have different lengths -- this should never "
    "happen since both come directly from the filtered df."
)

EMBEDDING_CACHE = "embeddings.npy"   # where cached embeddings live


# =====================================================================
# 1. EXPENSIVE STAGE -- embeddings (computed ONCE, then cached)
# =====================================================================
# This is the slow, GPU-friendly part. We compute it once and save to
# disk. Every subsequent run loads from EMBEDDING_CACHE and skips this
# entirely, so you can tune clustering below without paying this cost.
#
# CONFIG YOU MIGHT CHANGE:
#   - EMBEDDING_MODEL: the single biggest quality lever. Swapping this
#     moves your topics more than any UMAP/HDBSCAN tuning will, so if you
#     change it, DELETE the cache file to force a recompute.
#       "all-MiniLM-L6-v2"  -> fast, 384-dim, good default
#       "all-mpnet-base-v2" -> slower, higher quality, general purpose
#       (domain-specific models exist for legal/clinical/scientific text)
EMBEDDING_MODEL = "all-mpnet-base-v2"

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

if os.path.exists(EMBEDDING_CACHE):
    print(f"Loading cached embeddings from {EMBEDDING_CACHE}")
    embeddings = np.load(EMBEDDING_CACHE)
    # Safety check: cache must match the current document count.
    if embeddings.shape[0] != len(documents):
        raise ValueError(
            f"Cache has {embeddings.shape[0]} rows but you have "
            f"{len(documents)} documents. Delete {EMBEDDING_CACHE} "
            "and recompute."
        )
else:
    print("Computing embeddings (slow -- happens once)...")
    embeddings = embedding_model.encode(
        documents,
        show_progress_bar=True,
        # batch_size=64,   # raise on a GPU for throughput
    )
    np.save(EMBEDDING_CACHE, embeddings)
    print(f"Saved embeddings to {EMBEDDING_CACHE}")


# =====================================================================
# 2. CHEAP STAGE -- dimensionality reduction (UMAP)
# =====================================================================
# CONFIG YOU MIGHT CHANGE:
#   - n_neighbors: local vs global. ~5 -> more, smaller, fine-grained
#     topics; ~50 -> fewer, broader topics. An indirect topic-count knob.
#   - n_components: target dimensions. 5 is typical; going much higher
#     hurts HDBSCAN.
#   - min_dist: keep at 0.0 for topic modeling (packs points tightly so
#     HDBSCAN finds dense clusters). Higher values are for visualization.
#   - random_state: set for reproducibility so your comparison is stable.
#     NOTE: fixing it disables UMAP's parallelism -> slower runs. Fine for
#     a comparison, annoying in a tight tuning loop.
umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    random_state=42,
)


# =====================================================================
# 3. CHEAP STAGE -- clustering (HDBSCAN) == your real topic-count control
# =====================================================================
# CONFIG YOU MIGHT CHANGE:
#   - min_cluster_size: THE main topic-count lever. Larger -> fewer,
#     bigger topics. Raise this to steer toward your ~30 LDA topics.
#     (Note: this is what BERTopic's `min_topic_size` argument sets under
#     the hood. Because we pass our own hdbscan_model below, DON'T also
#     set min_topic_size on BERTopic -- it would be ignored.)
#   - min_samples: how aggressively docs get dumped into the -1 outlier
#     bucket. Lower it if too much of your corpus lands in -1.
#   - cluster_selection_method: 'eom' -> fewer/larger clusters (default);
#     'leaf' -> more, smaller, more uniform clusters.
hdbscan_model = HDBSCAN(
    min_cluster_size=10,
    min_samples=5,        # defaults to min_cluster_size if left None
    metric="euclidean",
    cluster_selection_method="eom",
    cluster_selection_epsilon=0.2,
    prediction_data=True,    # needed if you later call .transform()
)


# =====================================================================
# 4. CHEAP STAGE -- topic word extraction (this is what you'll COMPARE)
# =====================================================================
# The vectorizer + c-TF-IDF turn each cluster's pooled text into ranked
# descriptive words. This is where your Gensim filtering instincts apply
# -- but it only affects topic LABELS, never which docs cluster together.
#
# CONFIG YOU MIGHT CHANGE (CountVectorizer):
#   - stop_words: cleans junk from topic labels without any stemming, so
#     no more "studi"/"happi" truncation from the LDA side.
#   - ngram_range: (1,2) lets labels include two-word phrases, usually
#     more readable than single tokens.
#   - min_df / max_df: the analogues of your filter_extremes no_below /
#     no_above. Drop terms in too few / too many documents.
vectorizer_model = CountVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,          # like no_below
    max_df=0.5,        # like no_above (fraction of docs)
)

# CONFIG YOU MIGHT CHANGE (c-TF-IDF):
#   - reduce_frequent_words: softly dampens very common words (gentler
#     than hard stop-word removal).
#   - bm25_weighting: BM25-style weighting; some find it yields more
#     distinctive topic words. Since you're comparing word LISTS, toggling
#     this directly changes what you're looking at.
ctfidf_model = ClassTfidfTransformer(
    reduce_frequent_words=True,
    bm25_weighting=False,
)

# CONFIG YOU MIGHT CHANGE (representation fine-tuning, optional):
#   MMR diversifies each topic's word list so it isn't ten near-synonyms.
#   diversity: 0.0 = no diversification, 1.0 = maximal. ~0.3 is a sane
#   starting point. Set representation_model=None to skip this stage.
representation_model = MaximalMarginalRelevance(diversity=0.3)


# =====================================================================
# 5. ASSEMBLE + FIT  (cheap: embeddings are passed in, not recomputed)
# =====================================================================
# Passing embeddings=... is the key to the fast-iteration workflow: with
# precomputed embeddings supplied, BERTopic skips the expensive encode
# step. We still hand it embedding_model so it can embed *new* docs later
# via .transform() if you want, but it is NOT used to embed the training
# set here.
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    ctfidf_model=ctfidf_model,
    representation_model=representation_model,
    calculate_probabilities=False,   # True is slower; needed for soft probs
    verbose=True,
)

print("Forming topics...")

topics, probs = topic_model.fit_transform(documents, embeddings=embeddings)

# Optional: collapse toward a target count AFTER fitting, if you'd rather
# fix the number than tune min_cluster_size. Comment out to leave as-is.
# topic_model.reduce_topics(documents, nr_topics="auto")

# ---------------------------------------------------------------------
# IMPORTANT: reduce_topics() updates the model IN PLACE (topic_model.
# topics_ / topic_model.probabilities_) but it does NOT update the
# local `topics` / `probs` variables above -- those are now stale
# (they still reflect the pre-reduction cluster count). Anything you
# persist or inspect after this point should read from the model's
# attributes, not from the `topics`/`probs` variables.
# ---------------------------------------------------------------------
topics = topic_model.topics_
probs = topic_model.probabilities_


# =====================================================================
# 6. INSPECT
# =====================================================================
print("\n=== Topic overview (topic -1 is the outlier bucket) ===")
print(topic_model.get_topic_info())

# Per-document assignments with probabilities:
#   topic_model.get_document_info(documents)


# =====================================================================
# 7. PERSIST RESULTS
# =====================================================================
# Two flat files, joinable on topic_id:
#   (a) topic_representative_words.csv -- one row per topic, its
#       ranked representative words. This is what you'll eyeball later
#       to assign a single human-readable label per topic.
#   (b) document_topic_assignments.csv -- one row per document, its
#       assigned topic_id (+ probability, if you ever turn
#       calculate_probabilities=True). Join this to (a) on topic_id,
#       or back to your original parquet on document_id.
OUTPUT_DIR = "output/bertopic"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- (a) topic_id -> representative words -----------------------------
topic_info = topic_model.get_topic_info()   # reflects post-reduce_topics state

topic_word_records = []
for tid in topic_info["Topic"]:
    word_scores = topic_model.get_topic(tid)   # [(word, score), ...] or False
    words = [w for w, _ in word_scores] if word_scores else []
    count = int(topic_info.loc[topic_info["Topic"] == tid, "Count"].iloc[0])
    topic_word_records.append({
        "topic_id": tid,
        "document_count": count,
        "is_outlier_bucket": tid == -1,
        "representative_words": ", ".join(words),
        "label": None,   # placeholder for the single-word label you'll add later
    })

topic_words_df = pd.DataFrame(topic_word_records)
topic_words_path = os.path.join(OUTPUT_DIR, "topic_representative_words.csv")
topic_words_df.to_csv(topic_words_path, index=False)
print(f"Saved {len(topic_words_df)} topics -> {topic_words_path}")

# ---- (b) document_id -> assigned topic_id ------------------------------
doc_topic_df = pd.DataFrame({
    "document_id": doc_ids,
    "topic_id": topics,
})

# `probs` is None whenever calculate_probabilities=False (the current
# setting above); keep the column present but empty for a stable schema
# so downstream code doesn't need to branch on whether it exists.
doc_topic_df["probability"] = probs if probs is not None else np.nan

doc_topic_path = os.path.join(OUTPUT_DIR, "document_topic_assignments.csv")
doc_topic_df.to_csv(doc_topic_path, index=False)
print(f"Saved {len(doc_topic_df)} document assignments -> {doc_topic_path}")

# ---- (c) optional: the fitted model itself ------------------------------
# Lets you reload topic_model later (e.g. to assign labels, or to run
# .transform() on new speeches) without re-running the embedding/UMAP/
# HDBSCAN pipeline. "safetensors" is the recommended format, but note:
# it does NOT persist the UMAP/HDBSCAN sub-models, so a *reloaded* model
# can still call .transform() (topics come back via document-embedding
# similarity) but won't give you the exact HDBSCAN-based soft
# probabilities you'd get from the live model in this session. If you
# need that later, save with serialization="pickle" instead (bigger,
# slower, and version-fragile across BERTopic/Python upgrades, but
# preserves the UMAP/HDBSCAN objects exactly as fitted).
SAVE_MODEL = True
if SAVE_MODEL:
    model_path = os.path.join(OUTPUT_DIR, "model")
    topic_model.save(
        model_path,
        serialization="safetensors",
        save_ctfidf=True,
        save_embedding_model=f"sentence-transformers/{EMBEDDING_MODEL}",
    )
    print(f"Saved model -> {model_path}")


# =====================================================================
# 8. COMPARE BERTopic topics against your Gensim LDA topics
# =====================================================================
# Assumes your earlier `lda` (LdaModel) is in scope. If it's pickled:
#   from gensim.models import LdaModel
#   lda = LdaModel.load("your_lda.model")

TOP_N = 10   # words per topic to show / compare


def lda_topic_words(lda, top_n=TOP_N):
    """{topic_id: [word, ...]} from a Gensim LdaModel. These words are
    Porter-STEMMED (from your preprocess_string pipeline)."""
    out = {}
    for tid in range(lda.num_topics):
        out[tid] = [w for w, _ in lda.show_topic(tid, topn=top_n)]
    return out


def bertopic_topic_words(model, top_n=TOP_N):
    """{topic_id: [word, ...]} from BERTopic, excluding the -1 outliers.
    These words are NOT stemmed."""
    out = {}
    for tid in model.get_topics():        # dict keyed by topic id
        if tid == -1:
            continue
        out[tid] = [w for w, _ in model.get_topic(tid)][:top_n]
    return out


# --- 8a. Side-by-side dump for eyeballing --------------------------------
# lda_words = lda_topic_words(lda)              # requires `lda` in scope
bt_words = bertopic_topic_words(topic_model)

# print("\n=== LDA topics (stemmed) ===")
# for tid, words in lda_words.items():
#     print(f"LDA {tid:>2}: {', '.join(words)}")

print("\n=== BERTopic topics (unstemmed) ===")
for tid, words in bt_words.items():
    print(f"BT  {tid:>2}: {', '.join(words)}")


# --- 8b. Automated best-match by word overlap ----------------------------
# Careful: your LDA words are STEMMED ("studi") and BERTopic's are not
# ("study", "studies"). A raw word-set overlap would badly undercount the
# real similarity. So we stem BOTH sides with the same Porter stemmer
# before comparing -- this normalizes them onto common ground.
from gensim.parsing.preprocessing import stem_text


def stem_set(words):
    # stem_text works on a string; join, stem, re-split.
    return set(stem_text(" ".join(words)).split())


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a or b) else 0.0