"""Ask Gemma 4 E4B (vLLM, offline) a yes/no question about every row of a parquet file. Each row is judged
--evals times independently with the prompt given in --prompt. Writes --column (the fraction of evaluations
that answered 1, from 0 to 1) and <column>_reason (every evaluation's reason, joined with ' | ') back to
the same parquet file.

Every option except the parquet path can also be set in a TOML file (-c/--config), under a [vote] table, with
the option name without its dashes as the key. The command line takes precedence over the file. This keeps
the full prompt, and every other setting, in a file you can store:
    [vote]
    column = "is_topic"
    prompt = '''
    Does the text discuss ...?
    '''

The work is split into batches that --gpus independent vLLM replicas (one per GPU) pull from a shared queue.
Each finished batch is saved to <parquet>.<column>.parts/, so an interrupted or failed run resumes where it
stopped. The parquet file itself is only rewritten once every batch is done."""

import argparse
import json
import multiprocessing as mp
import os
import queue
import random
import re
import shutil
import time
import traceback
from pathlib import Path

import configargparse
import numpy as np
import pandas as pd

MODEL = "google/gemma-4-E4B-it"
MAX_MODEL_LEN = 65536
BATCH = 500
MAX_CONSECUTIVE_ERRORS = 5
HEAD_CHARS, TAIL_CHARS = 12000, 4000  # longer fields keep only their start and end
MAX_TOKENS = 256  # room for a JSON answer with a one-sentence reason
MAX_TOKENS_THINKING = 4096  # room for the thought as well
SAMPLE = 5  # documents printed per batch for monitoring
SNIPPET_CHARS = 500  # head and tail of each column shown for those documents

OUTPUT_FORMAT = """
NOTE: Very long texts may have been truncated and this will be represented by "[... n characters omitted ...]". You are to ignore this.

OUTPUT FORMAT: respond with a single JSON object with exactly two keys, in this order: first "reason", then "result".
- "reason": one short sentence (under 25 words) explaining your answer.
- "result": the integer 0 or 1, as defined above.
Example: {"reason": "One short sentence.", "result": 1}
No other output is allowed: no text before or after the JSON object, no code block, no backticks, no explanation. Any response that is not exactly this JSON object is invalid.
"""

# Removes, in one pass: everything up to the end of Gemma 4's thought block (<|channel>thought ... <channel|>),
# end-of-turn sentinels, a stray leading "thought" label, and markdown code fences / backticks.
CLEAN = re.compile(
    r"^.*?<channel\|>|<turn\|>|<eos>|^thought\n|```(?:json)?|`",
    re.IGNORECASE | re.DOTALL,
)


def system_message(prompt):
    return {"role": "system", "content": prompt.strip() + "\n\n" + OUTPUT_FORMAT}


def select_rows(full, query):
    if not query:
        return list(range(len(full)))
    # on a fresh 0..n-1 index, so that the labels of the matches are their positions whatever the real index is
    return full.reset_index(drop=True).query(query).index.tolist()


def to_text(value):
    # list / array cells are not scalars, and pd.isna would return an array for them
    if not pd.api.types.is_scalar(value):
        return str(value)
    return "" if pd.isna(value) else str(value)


def format_record(record, columns):
    lines = []
    for key in columns:
        text = to_text(record[key])
        if len(text) > HEAD_CHARS + TAIL_CHARS:
            omitted = len(text) - HEAD_CHARS - TAIL_CHARS
            text = f"{text[:HEAD_CHARS]}\n[... {omitted} characters omitted ...]\n{text[-TAIL_CHARS:]}"
        lines.append(f"{key}: {text.strip() or '(empty)'}")
    return "\n".join(lines) + "\n\nRespond with the JSON object only:"


def parse_response(raw):
    try:
        answer = json.loads(CLEAN.sub("", raw).strip())
        reason, result = answer["reason"], answer["result"]
    except (ValueError, KeyError, TypeError):
        return None
    # `type(...) is int` rather than isinstance, so that JSON true/false are rejected
    if isinstance(reason, str) and type(result) is int and result in (0, 1):
        return " ".join(reason.split()), result
    return None


def evaluate(llm, conversations, sampling, evals, thinking):
    votes = [[] for _ in conversations]
    reasons = [[] for _ in conversations]
    # (conversation index, invalid responses in a row) for every evaluation still to do
    pending = [(i, 0) for i in range(len(conversations)) for _ in range(evals)]
    while pending:
        outputs = llm.chat(
            [conversations[i] for i, _ in pending],
            sampling,
            chat_template_kwargs={"enable_thinking": thinking},
            use_tqdm=False,
        )
        retry = []
        for (i, errors), output in zip(pending, outputs):
            raw = output.outputs[0].text
            if parsed := parse_response(raw):
                reasons[i].append(parsed[0])
                votes[i].append(parsed[1])
            elif errors + 1 >= MAX_CONSECUTIVE_ERRORS:
                raise RuntimeError(
                    f"one evaluation gave {MAX_CONSECUTIVE_ERRORS} invalid responses in a row "
                    f"(last: {raw[-300:]!r})"
                )
            else:
                retry.append((i, errors + 1))
        pending = retry
    return votes, reasons


def print_sample(gpu, row, record, columns, votes, reasons):
    fields = []  # (label, text) pairs rather than a dict, as a column could itself be called "reasons"
    for key in columns:
        text = " ".join(to_text(record[key]).split())
        if len(text) > 2 * SNIPPET_CHARS:
            text = f"{text[:SNIPPET_CHARS]} [...] {text[-SNIPPET_CHARS:]}"
        fields.append((str(key), text or "(empty)"))
    fields.append(("reasons", " | ".join(reasons)))
    width = max(len(label) for label, _ in fields)
    # a single print call, so that lines from different GPUs are not interleaved
    lines = [f"\n[gpu {gpu}] row {row}: {sum(votes)}/{len(votes)} voted 1"]
    lines += [f"  {label + ':':<{width + 1}} {text}" for label, text in fields]
    print("\n".join(lines), flush=True)


def save_part(parts_dir, start, votes, reasons):
    part = pd.DataFrame(
        {
            "row": range(start, start + len(votes)),
            "fraction": [sum(v) / len(v) for v in votes],
            "reason": [" | ".join(r) for r in reasons],
        }
    )
    # written under another name first, so a crash never leaves a half-written part behind
    tmp = parts_dir / f"{start:09d}.tmp"
    part.to_parquet(tmp)
    os.replace(tmp, parts_dir / f"{start:09d}.parquet")


def worker(gpu, args, positions, shown, parts_dir, tasks, results):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    from vllm import LLM  # imported only now, after the GPU has been pinned

    try:
        df = pd.read_parquet(args.parquet, columns=shown).iloc[positions]
        system = system_message(args.prompt)
        llm = LLM(MODEL, max_model_len=MAX_MODEL_LEN, generation_config="auto")
        # the model's own generation_config.json values, except for the length...
        sampling = llm.get_default_sampling_params()
        sampling.max_tokens = MAX_TOKENS_THINKING if args.thinking else MAX_TOKENS
        # ...and the thought delimiters, which must not be stripped when thinking
        sampling.skip_special_tokens = not args.thinking

        while (start := tasks.get()) is not None:
            batch = df.iloc[start : start + BATCH].to_dict("records")
            conversations = [
                [system, {"role": "user", "content": format_record(r, shown)}]
                for r in batch
            ]
            votes, reasons = evaluate(
                llm, conversations, sampling, args.evals, args.thinking
            )
            save_part(parts_dir, start, votes, reasons)
            for i in sorted(random.sample(range(len(batch)), min(SAMPLE, len(batch)))):
                print_sample(gpu, start + i, batch[i], shown, votes[i], reasons[i])
            results.put(("done", gpu, None))
    except Exception:  # noqa: BLE001 - any failure must reach the parent, which aborts the run
        results.put(("error", gpu, traceback.format_exc()))


def run_workers(args, positions, shown, parts_dir, todo):
    context = mp.get_context("spawn")  # a forked process cannot safely use CUDA
    tasks, results = context.Queue(), context.Queue()
    n_workers = min(args.gpus, len(todo))
    for start in todo:
        tasks.put(start)
    for _ in range(n_workers):
        tasks.put(None)  # tells a worker that there is nothing left to do
    workers = [
        context.Process(
            target=worker,
            args=(gpu, args, positions, shown, parts_dir, tasks, results),
        )
        for gpu in range(n_workers)
    ]
    for w in workers:
        w.start()

    try:
        began, finished = time.monotonic(), 0
        while finished < len(todo):
            try:
                kind, gpu, details = results.get(timeout=10)
            except queue.Empty:
                # a worker killed from outside (e.g. out of memory) reports nothing, so check on them
                if crashed := [w for w in workers if w.exitcode not in (None, 0)]:
                    raise RuntimeError(
                        f"a worker process died with exit code {crashed[0].exitcode}"
                    ) from None
                continue
            if kind == "error":
                raise RuntimeError(f"worker on GPU {gpu} failed:\n{details}")
            finished += 1
            elapsed = time.monotonic() - began
            remaining = elapsed / finished * (len(todo) - finished)
            print(
                f"\n{finished}/{len(todo)} batches done in {elapsed / 60:.1f} min, about {remaining / 60:.1f} min left",
                flush=True,
            )
        for w in workers:
            w.join(timeout=120)
    finally:
        for w in workers:
            if w.is_alive():
                w.terminate()
            w.join()


def open_parts_dir(parts_dir, fingerprint):
    parts_dir.mkdir(exist_ok=True)
    fingerprint_file = parts_dir / "config.json"
    if not fingerprint_file.exists():
        fingerprint_file.write_text(json.dumps(fingerprint))
    elif json.loads(fingerprint_file.read_text()) != fingerprint:
        raise SystemExit(
            f"{parts_dir} holds results from a run with different settings (prompt, columns, query, evals, "
            f"thinking, batch size or input file). Delete it to start afresh."
        )


def main():
    parser = configargparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,  # so that a mistyped key in the TOML file is an error, not a guess
        config_file_parser_class=configargparse.TomlConfigParser(["vote"]),
        args_for_setting_config_path=["-c", "--config"],
    )
    parser.add_argument(
        "parquet", help="input dataframe; results are written back to this file"
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="the full text of the task; the JSON output format is appended automatically",
    )
    parser.add_argument(
        "--column",
        required=True,
        help="name of the result column; reasons go to <column>_reason",
    )
    parser.add_argument(
        "--columns", nargs="+", help="columns shown to the model (default: all)"
    )
    parser.add_argument(
        "--evals", type=int, default=5, help="independent evaluations per document"
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=4,
        help="number of GPUs, each running its own model replica",
    )
    parser.add_argument(
        "--thinking", action="store_true", help="enable Gemma 4 thinking mode"
    )
    parser.add_argument(
        "--query",
        help="a pandas DataFrame.query expression; only the matching rows are judged, the others are left empty",
    )
    args = parser.parse_args()
    if args.evals < 1:
        parser.error("--evals must be at least 1")
    if args.gpus < 1:
        parser.error("--gpus must be at least 1")
    if not args.prompt.strip():
        parser.error("--prompt must not be empty")
    if not Path(args.parquet).is_file():
        parser.error(f"{args.parquet} does not exist")

    full = pd.read_parquet(args.parquet)
    positions = select_rows(full, args.query)
    if not positions:
        parser.error(
            "no rows to process: the dataframe is empty or the query matches nothing"
        )
    print(f"{len(positions)} of {len(full)} rows selected for processing", flush=True)

    reason_column = f"{args.column}_reason"
    shown = [
        c
        for c in (args.columns or full.columns)
        if c not in (args.column, reason_column)
    ]
    if missing := [c for c in shown if c not in full.columns]:
        parser.error(f"columns not in the dataframe: {missing}")
    if not shown:
        parser.error("no columns left to show the model")

    parts_dir = Path(f"{args.parquet}.{args.column}.parts")
    open_parts_dir(
        parts_dir,
        {
            "model": MODEL,
            "system": system_message(args.prompt)["content"],
            "columns": shown,
            "query": args.query,
            "evals": args.evals,
            "thinking": args.thinking,
            "batch": BATCH,
            "rows": len(positions),
            "total_rows": len(full),
        },
    )
    done = {int(p.stem) for p in parts_dir.glob("*.parquet")}
    todo = [start for start in range(0, len(positions), BATCH) if start not in done]
    print(f"{len(done)} batches already done, {len(todo)} to run", flush=True)
    if todo:
        try:
            run_workers(args, positions, shown, parts_dir, todo)
        except RuntimeError as error:
            raise SystemExit(
                f"Aborting: {error}\nFinished batches are kept in {parts_dir}; rerun to resume."
            ) from None

    parts = pd.concat(
        [pd.read_parquet(p) for p in parts_dir.glob("*.parquet")]
    ).sort_values("row")
    if parts["row"].tolist() != list(range(len(positions))):
        raise SystemExit(
            f"The parts in {parts_dir} do not cover every row exactly once."
        )
    # rows that the query left out stay empty
    fraction = np.full(len(full), np.nan)
    fraction[positions] = parts["fraction"].to_numpy()
    reason = np.full(len(full), pd.NA, dtype=object)
    reason[positions] = parts["reason"].to_numpy()
    full[args.column] = fraction
    full[reason_column] = pd.array(reason, dtype="string")
    tmp = args.parquet + ".tmp"
    full.to_parquet(tmp)
    os.replace(tmp, args.parquet)  # atomic, so a crash can't leave a half-written file
    shutil.rmtree(parts_dir)
    print(f"mean {args.column} = {full[args.column].mean():.3f} -> {args.parquet}")


if __name__ == "__main__":
    main()