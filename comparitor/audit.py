#!/usr/bin/env python3
"""
audit.py

Checks how well one or more models' emotion scores agree with a trusted
"reference" score -- for example one made by humans, or by the most accurate
model -- so that you can choose a model and understand how far to trust it.

The scores are columns of a parquet file, as written by `comparitor.py
scores`. Only the order of the scores matters: a model that ranks every
speech the same way as the reference agrees perfectly, even if its numbers
are on a different scale. The reference also needs its <reference>_error
column, which says how precisely each of its scores is known. The models'
own error columns are not used: a model's imprecision is part of how
accurate it is, so nothing is done to excuse it.

Usage:
    python audit.py hansard.parquet --reference emotion_human \\
        --candidates emotion_fast emotion_balanced

Only the speeches that have a value in every one of these columns are used.
Make sure that every model's column was made from the same number of
comparisons, as a model's accuracy improves with more comparisons.


WHAT THE NUMBERS MEAN

Everything is about pairs of speeches. For each pair we ask: does the model
put the two in the same order as the reference, that is, agree on which of
the two is more emotional?

reference reliability
    How much of the spread in the reference's scores is real, rather than
    measurement noise. 1 means the reference is precise; 0.5 means that half
    of the differences between its scores are noise. Below about 0.7 the
    reference is too noisy to tell good models apart well, and it needs more
    comparisons before the numbers below mean much.

best possible
    The confident agreement (see below) that a perfect model could expect.
    The reference is itself a little noisy, so even a model that knew the
    true answers would sometimes disagree with it. This depends only on the
    reference, so it is the same for every model.

agreement
    Pick two speeches at random: how often does the model order them the
    same way as the reference? 50% is no better than tossing a coin, and
    100% means always the same order. This counts every pair equally,
    including pairs the reference itself can barely tell apart, so it makes
    every model look somewhat worse than it is. It is shown for context.

confident agreement
    The same, but each pair counts according to how sure the reference is
    of its order: pairs it clearly separates count fully, and near-ties
    count for almost nothing. So a model isn't blamed for disagreeing where
    the reference itself can't really tell.

score
    How far the model gets from a coin toss (0) to the best possible (1).
    This is the main number for choosing a model: 0.9 means the model gets
    90% of the way to the agreement that the reference's own precision
    allows. It can come out a little above 1, as the best possible is itself
    an estimate; read that as "as good as this reference can tell".

clear-cut errors
    Of the pairs the reference is clearly sure about (their two scores at
    least two combined errors apart, so that it is about 95% sure of their
    order), the share that the model puts the other way round. This is the
    plainest measure of how often the model is simply wrong.

95% ranges
    Every number is followed by a range in brackets. The numbers depend on
    which speeches happen to be in the sample; the range is where the value
    would probably fall for another sample of the same size. It is found by
    repeating the whole calculation many times (--resamples), each time on
    speeches drawn at random from the sample, with repeats allowed. Small
    samples give wide ranges.


CHOOSING A MODEL

Compare the models' scores. The report also gives the difference in score
between every two models, with its own range:

- If the range includes 0, this sample can't tell which of the two is
  better, so you might as well use the cheaper one.
- If the whole range is above (or below) 0, the first model really is
  better (or worse) than the second.

To tell apart two models that are close, you need a larger sample of
speeches that the reference has judged.


UNDERSTANDING ONE MODEL'S ERROR

The agreement plot (saved as <parquet>.<reference>.audit.png unless --plot
is given) shows each model's agreement separately for pairs that the reference puts
close together (on the left) and far apart (on the right), in units of the
two speeches' combined error. The dashed line is what a perfect model could
get, and under each group is the share of all pairs that fall in it. The
shaded bands are the 95% ranges.

- A line close to the dashed line: the model is about as good as the
  reference can tell.
- A line that falls short on the left but reaches nearly 100% on the right:
  the model measures the same thing as the reference, but less precisely.
  It mixes up speeches that are close, but not ones that are clearly
  different. More comparisons per speech would make it more precise.
- A line that stays well below the dashed line even on the right: the model
  disagrees with the reference about speeches that aren't close at all, so
  it is judging something different from what the reference judges. More
  comparisons won't fix that; a different prompt or model might.


FINDING OUTLIERS

The rank plot (saved as <parquet>.<reference>.ranks.png unless --rank-plot
is given) has one panel per model. Each dot is a speech: across, its rank by
the reference (1 is the least emotional); up, its rank by the model. If the
two agreed perfectly, every dot would sit on the dashed diagonal line.

The reference's own ranks are uncertain, so each dot has a grey bar across
it showing the range of ranks the reference's errors allow: the speech must
rank above every speech that the reference is clearly sure is less
emotional (by two combined errors or more, as for clear-cut errors), and
below every one that it is clearly sure is more emotional.

The speeches the model gets clearly wrong are shown in red, and labelled
with the row's index in the parquet file; these are the speeches worth
reading to see why the model gets them wrong. The report lists them for
each model, worst first. Which speeches count as clearly wrong depends on
--highlight:

share (the default)
    Every speech is in many clear-cut pairs: pairs that the reference is
    clearly sure about. A speech is red when the model orders more than half
    of its clear-cut pairs the other way round, so that whenever the
    reference is sure about this speech, the model is more often wrong than
    right. This is the clear-cut errors number worked out for one speech at
    a time, so the two can be read together.

range
    A speech is red when its grey bar misses the diagonal: the model ranks
    it where the reference is clearly sure it doesn't belong, above more
    speeches than the reference is sure it is above, or below more than it
    is sure it is below. How far the dot is beyond the end of its bar is the
    least number of clear-cut pairs the model must have got wrong for it.

Either way, a noisier model scatters further from the diagonal, but a
speech only turns red when the model contradicts what the reference is
clearly sure about. With share, a red dot's bar can still cross the
diagonal: the model may give the speech a plausible rank while still
ordering it the wrong way against the particular speeches the reference is
sure about.


CAVEATS

- The best possible, the score and the confidence of each pair all rely on
  the reference's _error column being roughly right.
- If the reference reliability is low, the best possible is uncertain, and
  so are the scores; their ranges will be wide to show it.
"""

import argparse
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write the plot to a file, with no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

# groups of pairs by how far apart the reference puts them, in combined errors
GAP_BINS = [0, 0.5, 1, 1.5, 2, 3, np.inf]
GAP_LABELS = ["0–0.5", "0.5–1", "1–1.5", "1.5–2", "2–3", "3+"]
CLEAR_GAP = 2  # pairs further apart than this are "clear-cut"
# with --highlight share, a speech is red when the model gets more than this
# share of its clear-cut pairs wrong
HIGHLIGHT_SHARE = 0.05
MARKERS = "osD^vP*X"


def pick_pairs(n, rng, max_pairs):
    """Every pair of n speeches, or a random sample of max_pairs of them when
    there are more, as the number of pairs grows with the square of n."""
    if n * (n - 1) // 2 <= max_pairs:
        return np.triu_indices(n, 1)
    i = rng.integers(0, n, max_pairs)
    j = rng.integers(0, n - 1, max_pairs)
    j += j >= i  # so that j is never i
    return i, j


def evaluate(ref, err, candidates, i, j):
    """Every statistic for one set of speeches, compared over the pairs (i, j).
    ref and err are the reference's scores and errors, and candidates maps each
    model's column name to its scores."""
    # the variance of the reference's true scores: its observed variance less
    # the part that is noise
    true_var = max(ref.var() - (err**2).mean(), 0.0)
    gap = ref[i] - ref[j]
    pair_var = err[i] ** 2 + err[j] ** 2
    z = np.abs(gap) / np.sqrt(pair_var)  # the gap in combined errors
    # how likely the reference is to have the pair in the right order; the gap
    # is shrunk by the share of it that is likely to be real, since a large
    # observed gap is partly noise (regression to the mean)
    right = norm.cdf(z * np.sqrt(2 * true_var / (2 * true_var + pair_var)))
    weight = 2 * right - 1  # 0 for a near-tie, 1 for a certain order
    group = np.digitize(z, GAP_BINS) - 1
    groups = [group == g for g in range(len(GAP_LABELS))]
    clear = z > CLEAR_GAP

    with np.errstate(invalid="ignore", divide="ignore"):
        best = (weight * right).sum() / weight.sum()
        stats = {
            "reliability": true_var / ref.var() if ref.var() > 0 else np.nan,
            "best": best,
            "group_share": [g.mean() for g in groups],
            "group_best": [right[g].mean() if g.any() else np.nan for g in groups],
        }
        for name, scores in candidates.items():
            # 1 if the model orders the pair as the reference does, 0 if not,
            # and 0.5 if either of them calls it a tie
            agree = (np.sign(gap) * np.sign(scores[i] - scores[j]) + 1) / 2
            confident = (weight * agree).sum() / weight.sum()
            stats[name] = {
                "agreement": agree.mean(),
                "confident": confident,
                "score": (confident - 0.5) / (best - 0.5),
                "clear_errors": 1 - agree[clear].mean() if clear.any() else np.nan,
                "groups": [agree[g].mean() if g.any() else np.nan for g in groups],
            }
    return stats


def resample(ref, err, candidates, n_resamples, max_pairs, rng):
    """evaluate() on n_resamples random resamples of the speeches, drawn with
    repeats allowed, to see how much each statistic depends on the sample.
    Every model is judged on the same resamples, so that they can be compared
    with each other fairly."""
    n = len(ref)
    replicates = []
    for _ in range(n_resamples):
        picked = rng.integers(0, n, n)
        i, j = pick_pairs(n, rng, max_pairs)
        distinct = picked[i] != picked[j]  # a speech paired with its own repeat
        replicates.append(
            evaluate(
                ref[picked],
                err[picked],
                {name: s[picked] for name, s in candidates.items()},
                i[distinct],
                j[distinct],
            )
        )
    return replicates


def spread(values, axis=None):
    """The 95% range of a statistic across the resamples."""
    with warnings.catch_warnings():
        # a statistic that no resample could compute, e.g. an empty group
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanpercentile(np.array(values, dtype=float), [2.5, 97.5], axis=axis)


def report(reference, n, n_pairs, full, replicates, names):
    def pct(value, low, high):
        return f"{value:.0%} ({low:.0%}–{high:.0%})"

    def num(value, low, high):
        return f"{value:.2f} ({low:.2f} to {high:.2f})"

    print(f"Reference: {reference} -- {n} speeches, {n_pairs:,} pairs\n")
    low, high = spread([r["reliability"] for r in replicates])
    print(f"  reference reliability   {num(full['reliability'], low, high)}")
    if full["reliability"] < 0.7:
        print(
            "    (below 0.7: the reference is too noisy to tell models apart well;\n"
            "     it needs more comparisons)"
        )
    low, high = spread([r["best"] for r in replicates])
    print(f"  best possible           {pct(full['best'], low, high)}\n")

    columns = [
        ("agreement", "agreement", pct),
        ("confident", "confident agreement", pct),
        ("score", "score", num),
        ("clear_errors", "clear-cut errors", pct),
    ]
    width = max(len(name) for name in names) + 2
    cells = {
        name: [
            fmt(
                full[name][key],
                *spread([r[name][key] for r in replicates]),
            )
            for key, _, fmt in columns
        ]
        for name in names
    }
    widths = [
        max(len(title), *(len(cells[name][c]) for name in names)) + 3
        for c, (_, title, _) in enumerate(columns)
    ]
    rows = [(" ", [title for _, title, _ in columns])]
    rows += [(name, cells[name]) for name in names]
    for label, row in rows:
        line = f"{label:<{width}}" + "".join(f"{c:<{w}}" for c, w in zip(row, widths))
        print(line.rstrip())

    if len(names) > 1:
        print("\nDifference in score (first model minus second):")
        for a, b in combinations(names, 2):
            diff = full[a]["score"] - full[b]["score"]
            low, high = spread([r[a]["score"] - r[b]["score"] for r in replicates])
            if low > 0:
                verdict = f"{a} is better"
            elif high < 0:
                verdict = f"{b} is better"
            else:
                verdict = "can't tell which is better from this sample"
            print(f"  {a} - {b}: {num(diff, low, high)} -- {verdict}")


def rank_ranges(ref, err):
    """For each speech, the lowest and highest rank (1 is the least emotional)
    the reference's errors allow it: it must rank above every speech that the
    reference is clearly sure is less emotional, and below every one that it is
    clearly sure is more emotional."""
    n = len(ref)
    below, above = np.empty(n, int), np.empty(n, int)
    rows = max(1, 5_000_000 // n)  # rows at a time, to bound memory for a large sample
    for start in range(0, n, rows):
        part = slice(start, start + rows)
        gap = ref[part, None] - ref[None, :]
        clear = CLEAR_GAP * np.sqrt(err[part, None] ** 2 + err[None, :] ** 2)
        below[part] = (gap > clear).sum(axis=1)
        above[part] = (-gap > clear).sum(axis=1)
    return below + 1, n - above


def clear_cut_shares(ref, err, rank):
    """For each speech, how many of its clear-cut pairs (those the reference is
    clearly sure about) the model orders the other way, and how many clear-cut
    pairs it is in. A tie by the model counts as half wrong, as for the
    clear-cut errors statistic."""
    n = len(ref)
    wrong, total = np.zeros(n), np.zeros(n, int)
    rows = max(1, 5_000_000 // n)  # rows at a time, to bound memory for a large sample
    for start in range(0, n, rows):
        part = slice(start, start + rows)
        gap = ref[part, None] - ref[None, :]
        clear = np.abs(gap) > CLEAR_GAP * np.sqrt(
            err[part, None] ** 2 + err[None, :] ** 2
        )
        error = (1 - np.sign(gap) * np.sign(rank[part, None] - rank[None, :])) / 2
        wrong[part] = (error * clear).sum(axis=1)
        total[part] = clear.sum(axis=1)
    return wrong, total


def highlighted(mode, ref, err, low, high, rank):
    """The speeches to show in red, as positions in the sample, worst first,
    with a note on each. mode "share": the model gets more than HIGHLIGHT_SHARE
    of the speech's clear-cut pairs wrong; mode "range": the model ranks it
    outside the range the reference's errors allow."""
    if mode == "range":
        distance = np.maximum(low - rank, rank - high)
        red = np.flatnonzero(distance > 0)
        red = red[np.argsort(-distance[red])]
        return red, [f"{distance[k]:g} ranks outside its range" for k in red]
    wrong, total = clear_cut_shares(ref, err, rank)
    share = np.divide(wrong, total, out=np.zeros(len(ref)), where=total > 0)
    red = np.flatnonzero(share > HIGHLIGHT_SHARE)
    red = red[np.lexsort((-wrong[red], -share[red]))]  # by share, then by count
    notes = [
        f"wrong on {wrong[k]:g} of its {total[k]} clear-cut pairs ({share[k]:.0%})"
        for k in red
    ]
    return red, notes


HEADINGS = {
    "share": "Speeches a model gets wrong on more than half of their clear-cut pairs",
    "range": "Speeches a model ranks outside the range the reference's errors allow",
}
RED_LABELS = {
    "share": ("other speeches", "wrong on over half its clear-cut pairs"),
    "range": ("within the reference's range", "outside the reference's range"),
}


def report_outliers(mode, reference, ids, ref, err, ref_rank, low, high, ranks):
    print(f"\n{HEADINGS[mode]}, worst first (rank 1 is the least emotional):")
    for name, rank in ranks.items():
        red, notes = highlighted(mode, ref, err, low, high, rank)
        print(f"  {name}: {len(red)} of {len(rank)} speeches")
        for k, note in zip(red, notes):
            print(
                f"    row {ids[k]}: {note}; {reference} rank {ref_rank[k]:g} (could "
                f"be {low[k]}–{high[k]}), {name} rank {rank[k]:g}"
            )


def rank_plot(
    path, mode, reference, ids, ref, err, ref_rank, low, high, ranks, max_labels=20
):
    n = len(ref_rank)
    ncols = min(3, len(ranks))
    nrows = -(-len(ranks) // ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.5 * ncols, 5.5 * nrows), squeeze=False
    )
    for ax, (name, rank) in zip(axes.flat, ranks.items()):
        out, _ = highlighted(mode, ref, err, low, high, rank)
        red = np.zeros(n, bool)
        red[out] = True
        ax.plot(
            [1, n], [1, n], "--", color="gray", linewidth=1, label="perfect agreement"
        )
        for mask, colour, size, label in (
            (~red, "tab:blue", 3, RED_LABELS[mode][0]),
            (red, "tab:red", 5, RED_LABELS[mode][1]),
        ):
            ax.errorbar(
                ref_rank[mask],
                rank[mask],
                xerr=[ref_rank[mask] - low[mask], high[mask] - ref_rank[mask]],
                fmt="o",
                markersize=size,
                color=colour,
                ecolor="tab:red" if colour == "tab:red" else "lightgray",
                elinewidth=1,
                alpha=0.8,
                label=label,
            )
        for k in out[:max_labels]:
            ax.annotate(
                str(ids[k]),
                (ref_rank[k], rank[k]),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7,
                color="tab:red",
            )
        more = f" ({max_labels} labelled)" if len(out) > max_labels else ""
        ax.set_title(f"{name}: {len(out)} of {n} speeches in red{more}", fontsize=10)
        ax.set_xlabel(f"rank by {reference}\n(grey bar: the ranks its errors allow)")
        ax.set_ylabel(f"rank by {name}")
        ax.set_xlim(0, n + 1)
        ax.set_ylim(0, n + 1)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(loc="upper left", fontsize=8)
    for ax in axes.flat[len(ranks) :]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Rank plot saved to {path}")


def plot(path, reference, full, replicates, names):
    x = np.arange(len(GAP_LABELS))
    fig, ax = plt.subplots(figsize=(9, 5.5))

    def band(values, **style):
        low, high = spread(values, axis=0)
        ax.fill_between(x, low, high, alpha=0.15, linewidth=0, **style)

    ax.plot(x, full["group_best"], "--", color="gray", label="best possible")
    for k, name in enumerate(names):
        (line,) = ax.plot(
            x, full[name]["groups"], marker=MARKERS[k % len(MARKERS)], label=name
        )
        band([r[name]["groups"] for r in replicates], color=line.get_color())

    ax.set_xticks(
        x,
        [
            f"{label}\n{share:.0%} of pairs"
            for label, share in zip(GAP_LABELS, full["group_share"])
        ],
    )
    ax.set_xlabel(
        f"how far apart {reference} puts the two speeches, in combined errors"
    )
    ax.set_ylabel(f"pairs ordered the same way as {reference}")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylim(0.4, 1.02)
    ax.axhline(0.5, color="lightgray", linewidth=0.8, label="coin toss")
    ax.set_title(f"Agreement with {reference}")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\nPlot saved to {path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("parquet", help="file with the score columns")
    parser.add_argument(
        "--reference",
        required=True,
        help="the trusted score column; <reference>_error must also exist",
    )
    parser.add_argument(
        "--candidates", nargs="+", required=True, help="the score columns to check"
    )
    parser.add_argument(
        "--resamples",
        type=int,
        default=1000,
        help="resamples used for the 95%% ranges; more give steadier ranges but take longer",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=200_000,
        help="above this many pairs, a random sample of them is used",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        help="where to save the agreement plot (default: next to the parquet)",
    )
    parser.add_argument(
        "--rank-plot",
        type=Path,
        help="where to save the rank plot (default: next to the parquet)",
    )
    parser.add_argument(
        "--highlight",
        choices=["share", "range"],
        default="share",
        help="which speeches the rank plot and report mark as clearly wrong: share, "
        "those the model gets wrong on more than half of their clear-cut pairs; "
        "range, those it ranks outside the range the reference's errors allow",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed, so that results can be repeated",
    )
    args = parser.parse_args()

    error = f"{args.reference}_error"
    columns = [args.reference, error, *args.candidates]
    df = pd.read_parquet(args.parquet)
    if missing := [c for c in columns if c not in df.columns]:
        parser.error(f"columns not in {args.parquet}: {missing}")
    if args.reference in args.candidates:
        parser.error("the reference can't also be a candidate")
    df = df[columns].dropna()
    if len(df) < 3:
        parser.error("fewer than 3 speeches have a value in every one of these columns")

    rng = np.random.default_rng(args.seed)
    ref = df[args.reference].to_numpy(float)
    err = df[error].to_numpy(float)
    candidates = {name: df[name].to_numpy(float) for name in args.candidates}

    i, j = pick_pairs(len(df), rng, args.max_pairs)
    full = evaluate(ref, err, candidates, i, j)
    if np.isnan(full["best"]):
        raise SystemExit(
            f"Every difference between {args.reference}'s scores is within its errors, so "
            "it can't be used to judge any model yet: it needs more comparisons."
        )
    replicates = resample(ref, err, candidates, args.resamples, args.max_pairs, rng)

    report(args.reference, len(df), len(i), full, replicates, args.candidates)
    ids = df.index.to_numpy()
    ref_rank = rankdata(ref)
    low, high = rank_ranges(ref, err)
    ranks = {name: rankdata(scores) for name, scores in candidates.items()}
    report_outliers(
        args.highlight, args.reference, ids, ref, err, ref_rank, low, high, ranks
    )

    stem = Path(args.parquet).stem
    path = args.plot or Path(args.parquet).with_name(
        f"{stem}.{args.reference}.audit.png"
    )
    plot(path, args.reference, full, replicates, args.candidates)
    path = args.rank_plot or Path(args.parquet).with_name(
        f"{stem}.{args.reference}.ranks.png"
    )
    rank_plot(
        path, args.highlight, args.reference, ids, ref, err, ref_rank, low, high, ranks
    )


if __name__ == "__main__":
    main()