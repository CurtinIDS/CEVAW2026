import os

# Keep every worker process single-threaded. Numeric libraries under numpy
# (OpenBLAS, MKL, etc.) default to spawning one thread per CPU core; if we
# then also run N processes at once, we get N processes x (cores) threads
# fighting over the same physical cores, which is slower than helpful.
# These must be set before numpy (and anything that imports numpy, like
# gensim) is imported anywhere in this process.
for _env_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_env_var] = "1"

import csv
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel, LdaMulticore
from gensim.parsing.preprocessing import preprocess_string


# ---------------------------------------------------------------------------
# Worker-process setup
# ---------------------------------------------------------------------------
# corpus_bow / dictionary / texts are handed to each worker exactly once,
# via the pool initializer, rather than being re-pickled for every task.

_worker_data = {}


def _init_worker(corpus_bow, dictionary, texts, workers_per_model):
    """Runs once per worker process when the pool starts."""
    _worker_data["corpus_bow"] = corpus_bow
    _worker_data["dictionary"] = dictionary
    _worker_data["texts"] = texts
    _worker_data["workers"] = workers_per_model


def _fit_and_score(k):
    """Train one LdaModel for topic count k and score its coherence.

    Runs inside a worker process. Each process trains a single model on a
    single thread, so there's no conflict with LdaMulticore's restrictions
    -- alpha="auto" works fine here. There's also no `workers=` argument
    to set: that parameter parallelizes *within* one model, but here the
    parallelism is across models (across k values) instead.
    """
    print(f"Fitting and scoring for k={k}...")
    d = _worker_data

    lda = LdaMulticore(
        corpus=d["corpus_bow"],
        id2word=d["dictionary"],
        num_topics=k,
        workers=d["workers"],
        passes=10,
        iterations=100,
        chunksize=500,
        random_state=42,
        # alpha="auto",
        alpha="asymmetric",
        eta="auto",
        eval_every=None,
    )
    cm = CoherenceModel(
        model=lda,
        texts=d["texts"],
        dictionary=d["dictionary"],
        coherence="c_v",
        processes=d["workers"],
    )
    score = cm.get_coherence()
    print(f"k={k:>3}   coherence={score:.4f}")
    return k, score, lda


def coherence_sweep(corpus_bow, dictionary, texts, ks=()):
    workers_per_model = max(1, (os.cpu_count() or 1) // len(ks))
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=len(ks),
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(corpus_bow, dictionary, texts, workers_per_model),
    ) as ex:
        results = list(ex.map(_fit_and_score, ks))

    coherence_scores = [score for _, score, _ in results]
    models = [lda for _, _, lda in results]
    return coherence_scores, models


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

def export_doc_topics(model, corpus_bow, doc_ids, path):
    """Document ID -> dominant topic (and its probability) for one model."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["doc_id", "topic_id", "topic_prob"])

        for doc_id, bow in zip(doc_ids, corpus_bow):
            topic_probs = model.get_document_topics(bow, minimum_probability=0.0)
            dominant_topic, dominant_prob = max(topic_probs, key=lambda tp: tp[1])
            writer.writerow([doc_id, dominant_topic, f"{dominant_prob:.4f}"])


def export_topic_words(model, path, topn=15):
    """Topic ID -> top words (with probabilities) for one model, for labeling."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["topic_id", "top_words"])

        for topic_id in range(model.num_topics):
            word_probs = model.show_topic(topic_id, topn=topn)
            word_list_str = ", ".join(f"{word} [{prob:.4f}]" for word, prob in word_probs)
            writer.writerow([topic_id, word_list_str])


def main():
    print("Loading data into memory...")
    df = pd.read_parquet("/mnt/data/cevaw/corpus_1998_to_2025_v8.parquet")
    df = df.query("`body`.str.len() > 250")

    # 1. Preprocess each document into a token list. doc_ids and bodies are
    #    pulled out as two plain, position-aligned lists (both come from the
    #    same df in the same row order); pool.map then preserves that order
    #    when it runs preprocess_string in parallel, so texts[i] still lines
    #    up with doc_ids[i] on the other side without having to pair them up
    #    by hand.
    doc_ids = df.index.tolist()
    bodies = df["body"].tolist()
    n_preprocess = os.cpu_count() or 1
    print(f"Preprocessing {len(bodies)} documents across {n_preprocess} process(es)...")
    with mp.Pool(processes=n_preprocess) as pool:
        texts = pool.map(preprocess_string, bodies)

    # 2. Build a dictionary (id <-> word mapping) from the tokenized docs
    print("Building dictionary mapping...")
    dictionary = Dictionary(texts)

    # 3. Filter the dictionary
    print("Filtering the dictionary...")
    dictionary.filter_extremes(no_below=3, no_above=0.2)

    # 4. Convert each document to bag-of-words
    print("Converting to a bag-of-words...")
    corpus_bow = [dictionary.doc2bow(text) for text in texts]

    ks = [20, 25, 30, 35, 40, 45, 50, 55, 60, 80] # 100, 150, 200]
    print("Beginning coherence sweep across...")
    coherence_scores, models = coherence_sweep(
        corpus_bow, dictionary, texts, ks
    )

    # Plot the curve
    plt.plot(ks, coherence_scores, marker="o")
    plt.xlabel("Number of topics")
    plt.ylabel("Coherence (c_v)")
    plt.title("Topic count vs. coherence")
    plt.grid(True, alpha=0.3)
    plt.show()

    # Grab the best-scoring model without retraining
    best_idx = coherence_scores.index(max(coherence_scores))
    best_k = ks[best_idx]
    best_model = models[best_idx]
    print(f"\nHighest coherence at k={best_k} ({coherence_scores[best_idx]:.4f})")

    # Readable printout: each topic as its top words with weights
    for topic_id in range(best_model.num_topics):
        words = [word for word, prob in best_model.show_topic(topic_id, topn=10)]
        print(f"Topic {topic_id}: {', '.join(words)}")

    # -------------------------------------------------------------------
    # Exports: one pair of CSVs per topic count, plus a coherence summary
    # -------------------------------------------------------------------
    # Every model from the sweep is written out, with k in the filename
    # (zero-padded so the files sort naturally), e.g.:
    #   doc_topic_assignments_k020.csv
    #   topic_word_lists_k020.csv
    # doc_ids comes from the tokenization step above (not df.index), so
    # this stays correct regardless of any filtering/reordering of df.

    basepath = Path("output/gensim")
    basepath.mkdir(parents=True, exist_ok=True)

    # Summary of the sweep itself, for quick side-by-side comparison
    scores_path = basepath / "coherence_scores.csv"
    with open(scores_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["num_topics", "coherence_c_v", "is_best"])
        for k, score in zip(ks, coherence_scores):
            writer.writerow([k, f"{score:.4f}", k == best_k])
    print(f"Wrote coherence summary to {scores_path}")

    for k, model in zip(ks, models):
        doc_topics_path = basepath / f"doc_topic_assignments_k{k:03d}.csv"
        topic_words_path = basepath / f"topic_word_lists_k{k:03d}.csv"

        export_doc_topics(model, corpus_bow, doc_ids, doc_topics_path)
        export_topic_words(model, topic_words_path, topn=15)

        print(f"k={k:>3}: wrote {doc_topics_path.name} and {topic_words_path.name}")


if __name__ == "__main__":
    main()