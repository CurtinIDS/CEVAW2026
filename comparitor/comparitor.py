#!/usr/bin/env python3
"""
comparitor.py

Pairwise comparison tool for rating how "emotional" Hansard speeches are.

Each speech is split into excerpts, and a rater is repeatedly shown two
excerpts and asked which is more emotional (or whether they tie). A
Bradley-Terry model is fitted to all the answers, giving every excerpt a
log-strength score, and a speech's score is the mean of its excerpts'.

Raters (--model):
    human     answers typed at the terminal
    fast      Gemma 4 E4B
    balanced  Gemma 4 26B-A4B in FP8
    accurate  Qwen3.8-Flash-Next in 4-bit

The LLM raters run in vLLM's offline mode, judging --nbatch pairs per call;
their settings are in models.py. fast and balanced each fit on one
A100-40GB, and accurate spans four (TP=4), with its n-gram table in host
RAM. --thinking turns on the model's thinking mode (with "medium" effort
for accurate); the thought is stripped from the reply before it is parsed.
The prompt (--prompt, given as text) must ask
for a JSON reply of the form {"winner": "A" | "B" | "TIE", "reason":
"..."}; the two excerpts are appended to it as TEXT A and TEXT B.

Pairs are drawn at random, favouring the least-compared excerpts, and the
two excerpts' positions are swapped at random to avoid a position bias.
Each comparison is appended to <corpus>.<colname>.<rater>.comparisons.jsonl
as soon as it is made, where <corpus> is the parquet path without its
extension, --colname names the experiment, and <rater> is the --model, with
"-thinking" added when --thinking is on (e.g. fast-thinking). So every
experiment and rater, with and without thinking, has its own log, and a
session can be stopped at any point (q, Ctrl-C, or too many consecutive LLM
errors) and resumed later. For a fair comparison between raters,
--max-comparisons stops `compare` once the log holds that many comparisons;
parallel processes can overshoot it by up to a batch each, so `scores` then
fits only the first that many. The settings that affect the judgements (model,
thinking and prompt) are saved to <corpus>.<colname>.<rater>.config.json
when the log is first written to, and a later `compare` or `scores` with
different settings is refused, so that one log never mixes judgements made
with different prompts. --prompt is required for a human rater too, though
the human is never shown it. To use several GPUs, run one `compare` process
per GPU, each with its own CUDA_VISIBLE_DEVICES; they share the log through
file locks. accurate already takes four GPUs, so give its process four.

Every option can also be set in a TOML file (-c/--config), under a
[comparitor] table, with the option name without its leading dashes as the
key (e.g. max-errors); the command line takes precedence. Keeping one such
file per experiment records exactly how its judgements were made, and the
same file serves every mode, which is always given on the command line:

    [comparitor]
    corpus = "hansard.parquet"
    colname = "emotion"
    model = "fast"
    thinking = true
    prompt = '''
    Decide which of the two excerpts expresses more emotion. ...
    '''

Relative paths in the file are taken relative to the current directory,
not to the file.

Usage, from the top-level folder, so that models.py can be imported:
    # judge pairs -- an LLM rater runs until Ctrl-C, a human one until q
    python -m comparitor.comparitor compare -c emotion.toml

    # fit the scores and print them with reliability diagnostics; saves
    # {source_id: (score, error)} to <corpus>.<colname>.<rater>.scores.pkl,
    # and writes the scores and errors to the <colname> and <colname>_error
    # columns of the corpus file
    python -m comparitor.comparitor scores -c emotion.toml

    # delete a comparisons log and its saved settings
    python -m comparitor.comparitor reset -c emotion.toml

    # without a config file, the required options go on the command line
    python -m comparitor.comparitor compare --corpus hansard.parquet --colname emotion \
        --model fast --prompt "Decide which ..." --thinking

Corpus format:
    A parquet file with one speech per row: the row index is the speech id
    and the `body` column its text. Rows whose `name` contains "speaker"
    (the Speaker and their deputies) are skipped. Speeches longer than 12
    sentences are split into excerpts of at least 6 sentences each; shorter
    ones are compared whole.
"""

import argparse
import fcntl
import json
import math
import os
import pickle
import random
import re
import textwrap
import time
from collections import Counter
from pathlib import Path

import configargparse
import numpy as np
import pandas as pd
from vllm import LLM

from models import MAX_MODEL_LEN, MODELS

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
VIRTUAL_ANCHOR = "__anchor__"  # fixed reference point used to regularise the fit

WINNER_TO_RESULT = {"A": "a", "B": "b", "TIE": "tie"}

MAX_TOKENS = 1024  # room for the JSON answer when not thinking

# Removes, in one pass: everything up to the end of the thought block (Gemma 4: <|channel>thought ... <channel|>;
# Qwen: ... </think>, as its chat template opens the block in the prompt), end-of-turn sentinels, a stray leading
# "thought" label, and markdown code fences / backticks.
CLEAN = re.compile(
    r"^.*?(?:<channel\|>|</think>)|<turn\|>|<eos>|<\|im_end\|>|<\|endoftext\|>|^thought\n|```(?:json)?|`",
    re.IGNORECASE | re.DOTALL,
)


def load_corpus(path):
    """Load the speeches in a parquet file as {"id", "text"} dicts, with the row
    index as the id and the `body` column as the text, skipping rows whose
    `name` contains "speaker"."""
    df = pd.read_parquet(path)
    df = df.query("not name.str.lower().str.contains('speaker')")
    print(f"Loaded {len(df)} rows of text")
    return [{"id": row.Index, "text": row.body} for row in df.itertuples()]


def split_sentences(text):
    return [s.strip() for s in SENTENCE_RE.split(text.strip()) if s.strip()]


def chunk_sentences(sentences, target=6):
    """Split sentences into excerpts as evenly as possible.
    If chunking occurs at all, every resulting chunk has at least `target`
    sentences."""
    n = len(sentences)
    if n <= 2 * target:
        return [sentences] if sentences else []
    num_chunks = min(max(1, round(n / (target - 0.5))), n // target)
    base, extra = divmod(n, num_chunks)
    chunks, i = [], 0
    for k in range(num_chunks):
        size = base + (1 if k < extra else 0)
        chunks.append(sentences[i : i + size])
        i += size
    return chunks


def build_excerpts(corpus):
    """Turn each source text into one or more excerpt records for comparison."""
    excerpts = []
    for row in corpus:
        for idx, sentences in enumerate(chunk_sentences(split_sentences(row["text"]))):
            excerpts.append(
                {
                    "excerpt_id": f"{row['id']}#{idx}",
                    "source_id": row["id"],
                    "text": " ".join(sentences),
                }
            )
    return excerpts


def rater_name(args):
    """The rater's name, used in its file names and log records: the --model,
    with "-thinking" added when thinking is on."""
    return f"{args.model}-thinking" if args.thinking else args.model


def output_path(args, kind):
    """<corpus>.<colname>.<rater>.<kind>, next to the corpus file, so that the
    files of each experiment and rater are kept apart."""
    corpus = Path(args.corpus)
    return corpus.with_name(f"{corpus.stem}.{args.colname}.{rater_name(args)}.{kind}")


def check_settings(args, create):
    """Refuse to mix judgements made with different settings in one log: the
    settings that affect the judgements are saved next to the log the first
    time it is written to (when `create`), and must match on every later run."""
    path = output_path(args, "config.json")
    settings = {
        "model": MODELS[args.model].name if args.model in MODELS else args.model,
        "thinking": args.thinking,
        "prompt": args.prompt.strip(),
    }
    if path.exists():
        if json.loads(path.read_text()) != settings:
            raise SystemExit(
                f"{path} holds different settings (model, thinking or prompt) from "
                "these. Use another --colname, or reset this one to start afresh."
            )
    elif create:
        path.write_text(json.dumps(settings, indent=2) + "\n")


def load_comparisons(log_path):
    if not log_path.exists():
        return []
    with log_path.open(encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_SH)  # Blocks only while someone holds LOCK_EX
        try:
            return [json.loads(line) for line in f if line.strip()]
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def append_comparison(log_path, a, b, result, rater="human", reason=""):
    record = {
        "a": a,
        "b": b,
        "result": result,
        "rater": rater,
        "reason": reason,
        "ts": time.time(),
    }
    with log_path.open("a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(json.dumps(record) + "\n")
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def comparison_counts(excerpt_ids, comparisons):
    counts = {e: 0 for e in excerpt_ids}
    for c in comparisons:
        counts[c["a"]] = counts.get(c["a"], 0) + 1
        counts[c["b"]] = counts.get(c["b"], 0) + 1
    return counts


def fit_scores(excerpt_ids, comparisons, warm_start=None, tolerance=1e-3):
    # Extract the comparisions list into numpy arrays
    excerpts_to_i = {id: i for i, id in enumerate(excerpt_ids)}
    a = np.array([excerpts_to_i[c["a"]] for c in comparisons])
    b = np.array([excerpts_to_i[c["b"]] for c in comparisons])

    points = {"a": 1, "b": 0, "tie": 0.5}
    wa = np.array([points[c["result"]] for c in comparisons])
    wb = 1 - wa

    # Add one fictitious game as an anchor: every excerpt draws with the anchor.
    # The anchor does two things: it avoids zeros in the numerator when computing
    # scores for excerpts without any games; and by fixing its associted p value
    # equal to 1, we ensure the normalisation of the values.
    n = len(excerpt_ids)
    a = np.concatenate([a, range(n)])
    b = np.concatenate([b, np.full(n, n)])
    wa = np.concatenate([wa, np.full(n, 0.5)])
    wb = np.concatenate([wb, np.full(n, 0.5)])

    # Initialise ps as unity to begin with and then perform fitting
    p_next = np.ones(n + 1) if warm_start is None else np.append(warm_start, 1.0)
    for iter in range(1000):
        p = p_next

        # Force anchor back to 1
        p[-1] = 1

        d = 1 / (p[a] + p[b])
        numerator = np.bincount(
            a, weights=wa * p[b] * d, minlength=n + 1
        ) + np.bincount(b, weights=wb * p[a] * d, minlength=n + 1)
        denominator = np.bincount(a, weights=wb * d, minlength=n + 1) + np.bincount(
            b, weights=wa * d, minlength=n + 1
        )

        # Use a damping factor to set p_next
        alpha = 0.2
        p_next = (1 - alpha) * p + alpha * (numerator / denominator)

        # Break early if we have converged
        maxdelta = np.abs(np.log(p) - np.log(p_next)).max()
        if maxdelta < tolerance:
            print(
                f"Bradley-Terry fitting breaking early due to convergence (iter={iter})"
            )
            break
    else:
        print(f"Bradley-Terry fitting did not converge early (max delta = {maxdelta})")

    return dict(zip(excerpt_ids, p_next))


def aggregate_to_sources(excerpts, excerpt_scores):
    """Average excerpt-level log-strength up to one score per source text."""
    by_source = {}
    for ex in excerpts:
        by_source.setdefault(ex["source_id"], []).append(
            excerpt_scores[ex["excerpt_id"]]
        )
    return {
        sid: sum(math.log(s) for s in scores) / len(scores)
        for sid, scores in by_source.items()
    }


def reliability(excerpt_ids, comparisons, pi=None):
    """
    Scale Separation Reliability: 1 - (average sampling noise in the scores)
    / (observed spread of the scores). Rises toward 1 as more comparisons
    make the ordering trustworthy; stays near 0 while it's still noise.
    """
    pi = pi or fit_scores(excerpt_ids, comparisons)
    theta = {e: math.log(p) for e, p in pi.items()}

    info = {e: 0.0 for e in excerpt_ids}
    for c in comparisons:
        i, j = c["a"], c["b"]
        p_ij = pi[i] / (pi[i] + pi[j])
        fisher = p_ij * (
            1 - p_ij
        )  # information one comparison carries about theta_i - theta_j
        info[i] += fisher
        info[j] += fisher

    compared = [e for e in excerpt_ids if info[e] > 0]
    if len(compared) < 2:
        return 0.0

    se_sq = {e: 1.0 / info[e] for e in compared}
    mean_theta = sum(theta[e] for e in compared) / len(compared)
    observed_var = sum((theta[e] - mean_theta) ** 2 for e in compared) / len(compared)
    mean_error_var = sum(se_sq.values()) / len(compared)
    return max(0.0, 1.0 - mean_error_var / observed_var) if observed_var > 0 else 0.0


def bootstrap_errors(excerpts, comparisons, n_boot):
    """
    Empirical per-source error via bootstrap: resample the comparisons with
    replacement n_boot times, refit, and average each replicate's excerpt
    log-strengths up to sources. Returns {source_id: standard deviation of
    its score across the replicates}. Aggregating inside each replicate
    accounts for any correlation between a source's excerpts.
    """
    excerpt_ids = [e["excerpt_id"] for e in excerpts]

    rng = np.random.default_rng()
    m = len(comparisons)
    base_pi = np.array(
        list(fit_scores(excerpt_ids, comparisons).values())
    )  # warm-start reference

    replicates = []
    for _ in range(n_boot):
        sample = [comparisons[i] for i in rng.integers(0, m, size=m)]
        excerpt_scores = fit_scores(
            excerpt_ids, sample, warm_start=base_pi, tolerance=5e-2
        )
        replicates.append(aggregate_to_sources(excerpts, excerpt_scores))

    # a DataFrame lines the replicates up by source id, so every error is
    # paired with its own source
    return pd.DataFrame(replicates).std(ddof=0).to_dict()


def bootstrap_summary(errors):
    errors = list(errors.values())
    return {
        "median_error": float(np.median(errors)),
        "p90_error": float(np.percentile(errors, 90)),
        "p95_error": float(np.percentile(errors, 95)),
    }


def pick_pairs(excerpt_ids, counts, k):
    """
    Select up to k pairs: draw excerpts without replacement, favouring those
    with the fewest comparisons so far, then pair them up at random
    regardless of their scores. When there are fewer than 2k excerpts, this
    is repeated, with each round's picks counted before the next, until there
    are k pairs. Pairs that have already been compared are not avoided -- in
    a large corpus a repeat is unlikely.
    """
    count = np.array([counts.get(e, 0) for e in excerpt_ids])
    pairs = []
    while len(excerpt_ids) >= 2 and len(pairs) < k:
        # Efraimidis-Spirakis weighted sampling without replacement: orders
        # excerpts favoring low comparison counts. Computed in log-space to
        # avoid underflows to bit-exact 0.0. In linear space, this would be:
        # np.random.random(len(count)) ** weight
        weight = (count + 1) ** 2
        log_keys = np.log(np.random.random(len(count))) * weight
        top = np.argsort(-log_keys)[: 2 * (k - len(pairs))]
        top = top[: len(top) // 2 * 2]  # an odd one out would go unpaired

        np.random.shuffle(top)
        pairs += [
            (excerpt_ids[top[i]], excerpt_ids[top[i + 1]])
            for i in range(0, len(top), 2)
        ]
        count[top] += 1
    return pairs


def ask_human(text_a, text_b, width=90):
    """Show a pair in the terminal and return the human's choice ('1'/'2'/'e'/'q')."""
    print("\n" + "-" * width)
    print("[1]")
    print(textwrap.fill(text_a, width))
    print("\n[2]")
    print(textwrap.fill(text_b, width))
    print("-" * width)
    while True:
        choice = input("Which is more emotional? [1/2/e=equal/q=quit] ").strip().lower()
        if choice in ("1", "2", "e", "q"):
            return choice
        print("Please enter 1, 2, e, or q.")


def build_llm_prompt(prompt, text_a, text_b):
    return f"{prompt}\n\nTEXT A\n\n{text_a}\n\nTEXT B\n\n{text_b}"


def parse_llm_response(raw):
    """Parse the model's JSON reply, after stripping any thought block,
    end-of-turn tokens and code fences, tolerating other text around it."""
    text = CLEAN.sub("", raw).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def ask_llm_batch(pairs, text_by_id, prompt, llm, sampling_params, template_args):
    """Ask the LLM to judge a whole batch of pairs in a single vLLM call;
    returns a (result, reason) tuple per pair, in the same order. result is
    None when the reply could not be parsed, and reason then says why."""
    messages = [
        [
            {
                "role": "user",
                "content": build_llm_prompt(prompt, text_by_id[a], text_by_id[b]),
            }
        ]
        for a, b in pairs
    ]
    outputs = llm.chat(
        messages,
        sampling_params=sampling_params,
        chat_template_kwargs=template_args,
    )

    results = []
    for output in outputs:
        try:
            parsed = parse_llm_response(output.outputs[0].text)
            winner = str(parsed.get("winner", "")).strip().upper()
            result = WINNER_TO_RESULT.get(winner)
            results.append((result, parsed.get("reason", "")))
        # a reply that isn't JSON, or isn't a JSON object
        except (ValueError, AttributeError) as exc:
            results.append((None, f"request failed: {exc}"))
    return results


def run_compare(args):
    excerpts = build_excerpts(load_corpus(args.corpus))
    excerpt_ids = [e["excerpt_id"] for e in excerpts]
    text_by_id = {e["excerpt_id"]: e["text"] for e in excerpts}
    if len(excerpt_ids) < 2:
        raise SystemExit(
            "The corpus has fewer than 2 excerpts, so there is nothing to compare."
        )
    rater = rater_name(args)
    log_path = output_path(args, "comparisons.jsonl")
    check_settings(args, create=True)
    prompt = args.prompt.strip()

    if args.model != "human":
        model = MODELS[args.model]
        llm = LLM(
            model=model.name,
            max_model_len=MAX_MODEL_LEN,
            generation_config="auto",
            **model.llm_args,
        )
        # the model's own generation_config.json values, except for the length...
        sampling_params = llm.get_default_sampling_params()
        sampling_params.max_tokens = (
            model.max_tokens_thinking if args.thinking else MAX_TOKENS
        )
        # ...and the thought delimiters, which must not be stripped when thinking
        sampling_params.skip_special_tokens = not args.thinking
        template_args = {"enable_thinking": args.thinking, **model.chat_args}
    else:
        llm, sampling_params, template_args = None, None, None

    done, consecutive_errors = 0, 0
    round_pairs = []
    try:
        while True:
            if not round_pairs:
                comparisons = load_comparisons(log_path)
                k = args.nbatch
                if args.max_comparisons:
                    k = min(k, args.max_comparisons - len(comparisons))
                    if k <= 0:
                        print(
                            f"The log holds {len(comparisons)} comparisons, reaching "
                            f"--max-comparisons ({args.max_comparisons})."
                        )
                        break
                counts = comparison_counts(excerpt_ids, comparisons)
                round_pairs = pick_pairs(excerpt_ids, counts, k)

            if args.model != "human":
                batch = [
                    (b, a) if random.random() < 0.5 else (a, b) for a, b in round_pairs
                ]  # avoid a systematic left/right or A/B bias
                round_pairs = []
                for (a, b), (result, reason) in zip(
                    batch,
                    ask_llm_batch(
                        batch, text_by_id, prompt, llm, sampling_params, template_args
                    ),
                ):
                    if result is None:
                        consecutive_errors += 1
                        print(
                            f"[llm error, {consecutive_errors}/{args.max_errors}] {reason}"
                        )
                        if consecutive_errors >= args.max_errors:
                            print("Too many consecutive LLM errors -- aborting.")
                            return
                        continue
                    consecutive_errors = 0
                    print(f"{a} vs {b}: {result.upper()} -- {reason}")
                    append_comparison(
                        log_path, a, b, result, rater=rater, reason=reason
                    )
                    done += 1
                continue

            a, b = round_pairs.pop()
            if random.random() < 0.5:  # avoid a systematic left/right or A/B bias
                a, b = b, a
            text_a, text_b = text_by_id[a], text_by_id[b]

            choice = ask_human(text_a, text_b)
            if choice == "q":
                print(f"Stopped after {done} comparisons this session.")
                return
            result, reason = {"1": "a", "2": "b", "e": "tie"}[choice], ""

            append_comparison(log_path, a, b, result, rater=rater, reason=reason)
            done += 1
    except KeyboardInterrupt:
        print(f"\nInterrupted -- {done} comparisons logged this session -> {log_path}")
        return

    print(f"Logged {done} comparisons this session -> {log_path}")


def run_scores(args):
    excerpts = build_excerpts(load_corpus(args.corpus))
    excerpt_ids = [e["excerpt_id"] for e in excerpts]
    check_settings(args, create=False)
    comparisons = load_comparisons(output_path(args, "comparisons.jsonl"))
    if not comparisons:
        print("No comparisons logged yet -- run the 'compare' command first.")
        return
    logged = len(comparisons)
    # the log is in the order the comparisons were made, so this keeps the
    # first ones, and every rater is fitted on the same number of them
    comparisons = comparisons[: args.max_comparisons]

    counts = comparison_counts(excerpt_ids, comparisons)
    print("Fitting scores...")
    excerpt_scores = fit_scores(excerpt_ids, comparisons)
    print("Done.")
    source_scores = aggregate_to_sources(excerpts, excerpt_scores)

    n_excerpts = {}
    for e in excerpts:
        n_excerpts[e["source_id"]] = n_excerpts.get(e["source_id"], 0) + 1

    if len(source_scores) > 250:
        stride = len(source_scores) // 250
        sorted_scores = sorted(
            list(source_scores.items())[::stride], key=lambda kv: -kv[1]
        )
    else:
        sorted_scores = sorted(source_scores.items(), key=lambda kv: -kv[1])

    print(f"{'source_id':<20}{'score':>10}{'excerpts':>10}{'comparisons':>13}")
    for sid, score in sorted_scores:
        n_comp = sum(counts[e["excerpt_id"]] for e in excerpts if e["source_id"] == sid)
        print(f"{sid:<20}{score:>10.3f}{n_excerpts[sid]:>10}{n_comp:>13}")

    by_rater = Counter(c.get("rater", "human") for c in comparisons)
    breakdown = ", ".join(f"{rater}: {n}" for rater, n in sorted(by_rater.items()))
    used = f", the first {len(comparisons)} used" if len(comparisons) < logged else ""
    print(
        f"\n{logged} comparisons logged in total{used} ({breakdown}). "
        "Score is average log-strength: 0 = about average, higher = more emotional."
    )

    print(
        f"\nExcerpt comparisons (min|median|max|total): {min(counts.values())} | {np.median(list(counts.values()))} | {max(counts.values())} | {2 * len(comparisons)}/{len(excerpts)} (x{2 * len(comparisons) / len(excerpts):.1f})"
    )

    print(
        f"\nScale separation reliability: {reliability(excerpt_ids, comparisons, pi=excerpt_scores):.2f} "
        "(rule of thumb: >0.7 reliable, >0.9 diminishing returns)"
    )

    errors = bootstrap_errors(excerpts, comparisons, n_boot=100)
    summary = bootstrap_summary(errors)
    print(
        f"95% of items have error below {summary['p95_error']:.2f} (log-strength units); "
        f"median error {summary['median_error']:.2f}"
    )

    source_scores = {sid: (score, errors[sid]) for sid, score in source_scores.items()}

    filename = output_path(args, "scores.pkl")
    with open(filename, "wb") as f:
        pickle.dump(source_scores, f)

    df = pd.read_parquet(args.corpus)
    df.loc[list(source_scores.keys()), args.colname] = [
        val for val, _ in source_scores.values()
    ]
    df.loc[list(source_scores.keys()), args.colname + "_error"] = [
        err for _, err in source_scores.values()
    ]
    # written under another name first, so that a crash can't leave a
    # half-written corpus file behind
    tmp = Path(f"{args.corpus}.tmp")
    df.to_parquet(tmp)
    os.replace(tmp, args.corpus)


def run_reset(args):
    paths = [
        p
        for p in (
            output_path(args, "comparisons.jsonl"),
            output_path(args, "config.json"),
        )
        if p.exists()
    ]
    if not paths:
        print("No comparisons log to remove.")
        return
    names = " and ".join(str(p) for p in paths)
    if input(f"Delete {names}? [y/N] ").strip().lower() in ("y", "yes"):
        for p in paths:
            p.unlink()
        print("Removed.")


def main():
    parser = configargparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,  # so that a mistyped key in the TOML file is an error, not a guess
        config_file_parser_class=configargparse.TomlConfigParser(["comparitor"]),
        args_for_setting_config_path=["-c", "--config"],
    )
    # one parser for every mode, rather than subcommands, because configargparse
    # doesn't pass config file values on to subcommands; so the same file serves
    # every mode, and each mode ignores the options it doesn't use
    parser.add_argument(
        "mode",
        choices=["compare", "scores", "reset"],
        help="compare: judge pairs (human or LLM rater); scores: fit and print the "
        "scores, and write them to the corpus file; reset: delete a comparisons log",
    )
    parser.add_argument("--corpus", required=True, help="parquet file of speeches")
    parser.add_argument(
        "--colname",
        required=True,
        help="name of the experiment, which names its files and the columns "
        "(<colname> and <colname>_error) that scores writes to the corpus file",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["human", *MODELS],
        help="who makes the judgements, which also names the files used: human "
        "(at the terminal), fast (Gemma 4 E4B), balanced (Gemma 4 26B-A4B in FP8) or "
        "accurate (Qwen3.8-Flash-Next in 4-bit, on four GPUs)",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="enable the LLM's thinking mode (with medium effort for accurate); this "
        "also names the files used, so judgements with and without thinking are "
        "kept apart",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="the prompt given to an LLM rater, as text; the two excerpts are "
        "appended to it (required for a human rater too, though it is not shown)",
    )
    parser.add_argument(
        "--nbatch",
        type=int,
        default=500,
        help="compare: pairs drawn per round, and so prompts per vLLM call",
    )
    parser.add_argument(
        "--max-comparisons",
        type=int,
        help="compare: stop once the log holds this many comparisons (by default it "
        "never stops); scores: fit only the first this many, as parallel compare "
        "processes can overshoot the limit by up to a batch each",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=5,
        help="compare: abort after this many consecutive LLM errors",
    )
    args = parser.parse_args()

    if args.model == "human" and args.thinking:
        parser.error("--thinking needs an LLM --model")
    if args.max_comparisons is not None and args.max_comparisons < 1:
        parser.error("--max-comparisons must be at least 1")

    {"compare": run_compare, "scores": run_scores, "reset": run_reset}[args.mode](args)


if __name__ == "__main__":
    main()