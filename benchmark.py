"""Throwaway benchmark: time vLLM for each model in models.py, thinking off and on. It sits next to models.py,
in the top-level folder:

python bench.py                  all models, one subprocess each, then a summary table
python bench.py --only fast      a single model, in this process, then the summary table with the
                                 earlier results of the other models; a rerun replaces that model's results

fast and balanced run on one GPU, accurate (Flash-Next) on four (TP=4), with its n-gram table in host RAM. The
"4-GPU req/s" column puts them on the same footing: one-GPU models times four, accurate as measured."""

import argparse
import itertools
import json
import os
import re
import statistics
import subprocess
import sys
import time

import numpy as np
import pandas as pd

import models

MODELS = models.MODELS
TASK = """
BACKGROUND

You are a text classifier: you will be given a text and will be asked to determine if it matches a given criteria.

The text is from the Australian Hansard corpus, containing speeches from both Australia Parliament. The corpus has been filtered to speeches made by members of the house and excludes the Speaker.

## Your task

Identify whether the given text is rasing a point of order. Set "result" to 1 if it is, and to 0 if it is not.

A point of order is raised when a member interrupts proceedings to draw the Speaker’s attention to an alleged breach of the Standing Orders, rules of debate, or parliamentary practice, and asks for a ruling. It is a procedural objection, not a contribution to the debate itself.

Key features (a text is a point of order if it):

- Is introduced by or clearly signals “point of order”, or is an interjected procedural objection directed to the Chair
- Alleges a specific rule breach, such as relevance, imputations or offensive words, unparliamentary language, misleading the House, tedious repetition, a member speaking out of turn, a breach of the sessional orders, or a quorum issue
- Often cites a standing order number or asks the Chair to direct a member (e.g. to withdraw, be relevant, or resume their seat)
- Is addressed to the Chair (“Mr Speaker”, “Madam Deputy President”), not to other members

Examples of points of order (result 1):

- “Mr SPEAKER, on a point of order, the member is not being relevant to the question.”
- “Point of order, Madam Deputy President. The minister is reflecting on a vote of the House.”
- “On a point of order under standing order 64, the member has used an unparliamentary word and should withdraw.”
- “I raise a point of order. The member for Grayndler is misleading the House.”
- “A point of order, Mr Deputy Speaker: the member is debating the question.”

Similar but NOT points of order (result 0):

- “I rise to speak on the bill.” (the start of a speech, not a procedural objection)
- “Mr Speaker, I seek leave to table a document.” (a request for leave, a different procedure)
- “I move that the question be put.” (a motion)
- “The member’s claims are completely false.” (a debating point or disagreement, not a rule breach alleged to the Chair)
- “I seek to make a personal explanation.” (a separate procedure)
- “Will the minister take an interjection?” or “Will the member give way?” (a request to the speaker, not the Chair)
- “Is the member aware of the point of order raised earlier?” (merely mentions the term, does not raise one)
- “Mr Speaker, I ask a supplementary question.” (a question, not an objection)
- “I thank the Speaker for his ruling on the point of order.” (refers to a past point of order, does not raise one)

Edge-case rules:

- If the text merely references or discusses a point of order without raising one, result 0.
- If a member raises a point of order and then continues with its substance, result 1.

## How to answer

Do not rely on your own notions of a point of order: strictly adhere to the definition given above.

Respond only with the JSON object described under OUTPUT FORMAT below: a one-sentence "reason", then "result", which is 1 or 0 as defined above. Do not answer with a bare digit, and do not write anything outside the JSON object.
"""

PARAGRAPHS = [
    row.body
    for row in pd.read_parquet("/mnt/data/cevaw/corpus_1998_to_2025_v8.parquet")
    .query("body.str.len() > 250")
    .sample(5000, random_state=np.random.RandomState(12345))
    .itertuples()
]
RESULTS = "bench_results.jsonl"
MAX_TOKENS = 256  # room for a JSON answer with a one-sentence reason
OUTPUT_FORMAT = """
OUTPUT FORMAT: respond with a single JSON object with exactly two keys, in this order: first "reason", then "result".
- "reason": one short sentence (under 25 words) explaining your answer.
- "result": the integer 0 or 1, as defined above.
Example: {"reason": "One short sentence.", "result": 1}
No other output is allowed: no text before or after the JSON object, no code block, no backticks, no explanation.
"""
# what to remove before judging an answer: the thought block and end-of-turn markers, but not code fences
THOUGHT = re.compile(
    r"^.*?(?:<channel\|>|</think>)|<turn\|>|<eos>|<\|im_end\|>|<\|endoftext\|>|^thought\n",
    re.IGNORECASE | re.DOTALL,
)


def is_valid(text):
    # the answer itself must be a JSON object with a string "reason" and an integer "result" of 0 or 1
    try:
        answer = json.loads(THOUGHT.sub("", text).strip())
        return (
            isinstance(answer, dict)
            and isinstance(answer["reason"], str)
            and type(answer["result"]) is int
            and answer["result"] in (0, 1)
        )
    except (ValueError, KeyError):
        return False


def chat(llm, conversations, sampling, template_args):
    return llm.chat(
        conversations, sampling, chat_template_kwargs=template_args, use_tqdm=False
    )


def conversations(docs, n):
    return [
        [
            {"role": "system", "content": TASK.strip() + "\n\n" + OUTPUT_FORMAT},
            {
                "role": "user",
                "content": f"body: {next(docs)}\n\nRespond with the JSON object only:",
            },
        ]
        for _ in range(n)
    ]


def gpus_of(model):
    return model.llm_args.get("tensor_parallel_size", 1)


def run(name, args):
    from vllm import LLM

    model = MODELS[name]
    docs = itertools.cycle(PARAGRAPHS)
    llm = LLM(
        model.name,
        max_model_len=models.MAX_MODEL_LEN,
        generation_config="auto",
        **model.llm_args,
    )
    for thinking in (False, True):
        sampling = llm.get_default_sampling_params()
        sampling.max_tokens = model.max_tokens_thinking if thinking else MAX_TOKENS
        sampling.skip_special_tokens = not thinking
        template_args = {"enable_thinking": thinking, **model.chat_args}

        chat(
            llm, conversations(docs, 16), sampling, template_args
        )  # warm-up, not timed
        for batch in range(args.batches):
            prompts = conversations(docs, args.batch_size)
            start = time.perf_counter()
            outputs = chat(llm, prompts, sampling, template_args)
            seconds = time.perf_counter() - start
            texts = [o.outputs[0].text for o in outputs]
            row = {
                "model": name,
                "thinking": thinking,
                "batch": batch,
                "seconds": seconds,
                "requests": len(outputs),
                "gpus": gpus_of(model),
                "prompt_tokens": sum(len(o.prompt_token_ids) for o in outputs),
                "output_tokens": sum(len(o.outputs[0].token_ids) for o in outputs),
                # prompt tokens that vLLM says it took from the prefix cache
                "cached_tokens": sum(
                    getattr(o, "num_cached_tokens", None) or 0 for o in outputs
                ),
                "truncated": sum(
                    o.outputs[0].finish_reason == "length" for o in outputs
                ),
                "sample": outputs[0].outputs[0].text[:300],
                "valid": sum(is_valid(t) for t in texts),
                # a bare 0 or 1, which the old prompt asked for
                "bare": sum(THOUGHT.sub("", t).strip() in ("0", "1") for t in texts),
                "invalid_sample": next((t[:300] for t in texts if not is_valid(t)), ""),
            }
            with open(RESULTS, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(row, flush=True)


def forget(name):
    # drop the earlier rows of this model, so that a rerun replaces them instead of being averaged with them
    try:
        with open(RESULTS) as f:
            kept = [line for line in f if json.loads(line)["model"] != name]
    except FileNotFoundError:
        kept = []
    with open(RESULTS, "w") as f:
        f.writelines(kept)


def summarise():
    with open(RESULTS) as f:
        rows = [json.loads(line) for line in f]
    print(
        f"\n{'model':10} {'thinking':8} {'s/batch':>8} {'req/s':>7} {'4-GPU req/s':>11} {'prompt tok':>10} {'prompt tok/s':>12} {'cached':>7} {'output tok':>10} {'cut off':>8} {'valid':>7} {'bare 0/1':>8}"
    )
    for name in MODELS:
        for thinking in (False, True):
            group = [
                r for r in rows if r["model"] == name and r["thinking"] == thinking
            ]
            if not group:
                print(f"{name:10} {thinking!s:8} {'no results':>8}")
                continue
            n = sum(r["requests"] for r in group)
            seconds = statistics.mean(r["seconds"] for r in group)
            rate = n / sum(r["seconds"] for r in group)
            print(
                f"{name:10} {thinking!s:8} {seconds:8.1f} {rate:7.2f} "
                f"{rate * 4 / group[0].get('gpus', 1):11.2f} "
                f"{sum(r['prompt_tokens'] for r in group) / n:10.0f} "
                f"{sum(r['prompt_tokens'] for r in group) / sum(r['seconds'] for r in group):12.0f} "
                f"{sum(r['cached_tokens'] for r in group) / sum(r['prompt_tokens'] for r in group):7.1%} "
                f"{sum(r['output_tokens'] for r in group) / n:10.0f} "
                f"{sum(r['truncated'] for r in group) / n:8.1%} "
                f"{sum(r['valid'] for r in group) / n:7.1%} "
                f"{sum(r['bare'] for r in group) / n:8.1%}"
            )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--only", choices=MODELS, help="run just this model, in this process"
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="with --only, do not print the summary table (used when running every model)",
    )
    parser.add_argument("--gpu", default="0", help="the one GPU to use")
    parser.add_argument(
        "--tp-gpus", default="0,1,2,3", help="the GPUs for models that span several"
    )
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument(
        "--batch-size", type=int, default=100, help="requests per batch"
    )
    args = parser.parse_args()
    if args.only:
        forget(args.only)
        run(args.only, args)
        if not args.no_summary:
            summarise()
        return

    open(RESULTS, "w").close()
    for name, model in MODELS.items():
        print(f"\n===== {name}", flush=True)
        command = [
            sys.executable,
            __file__,
            "--only",
            name,
            "--batches",
            str(args.batches),
        ]
        command += ["--batch-size", str(args.batch_size), "--no-summary"]
        done = subprocess.run(
            command,
            check=False,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": args.tp_gpus
                if gpus_of(model) > 1
                else args.gpu,
            },
        )
        if done.returncode:
            print(f"===== {name} failed with exit code {done.returncode}", flush=True)
    summarise()


if __name__ == "__main__":
    main()