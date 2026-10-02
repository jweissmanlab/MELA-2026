#!/usr/bin/env python3
"""
Annotate the strongest ancestral-linkage relationships using headless Claude Code.

Ancestral linkage (`norm_value`) measures shared ancestry between two cell types
in the E9.5 mouse embryo: positive = more shared ancestry than expected by chance,
negative = less. This script selects every statistically significant, positive
linkage above a cutoff (default: p_value < 0.05, norm_value >= 0.5, excluding
self-pairs) and asks Claude to add:

  * proposed_explanation : one sentence hypothesising why the two cell types share
                           ancestry, grounded in mouse developmental biology.
  * novelty              : an integer 1 (well-established / textbook) to
                           5 (totally novel / unexpected).

By default the input file is modified in place: all rows are preserved and
reordered by linkage (norm_value) descending. The two annotation columns are
filled only on qualifying pairs; annotations on rows that do not satisfy the
criteria are cleared, so only statistically significant pairs above the cutoff
ever carry an annotation. Pass --output to write to a separate file instead.
Either way the run is resumable: qualifying rows that already carry a
proposed_explanation and novelty are preserved and never re-queried.

Usage:
    python annotate_linkage.py                              # modify input in place
    python annotate_linkage.py --min-linkage 0.5           # cutoff (default 0.5)
    python annotate_linkage.py --output results/sig.csv    # write a separate file
    python annotate_linkage.py --model claude-opus-4-8      # override model
    python annotate_linkage.py --dry-run                    # print prompts only
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_INPUT = RESULTS_DIR / "e95_type_linkage_annotated.csv"

FAILED_MARKER = "Annotation failed after retries."

NEW_COLUMNS = ["proposed_explanation", "novelty"]
OBSOLETE_COLUMNS = ["top5_for"]  # from the earlier top-5-per-cell-type scheme

JSON_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "proposed_explanation": {"type": "string"},
            "novelty": {"type": "integer", "minimum": 1, "maximum": 5},
        },
        "required": ["proposed_explanation", "novelty"],
        "additionalProperties": False,
    }
)

SYSTEM_PROMPT = (
    "You are an expert mouse developmental biologist and single-cell lineage-tracing "
    "researcher. You are interpreting an ancestral-linkage analysis of an E9.5 mouse "
    "embryo. Ancestral linkage (norm_value) quantifies shared ancestry between two cell "
    "types measured from single-cell lineage-tracing trees: a positive value means the "
    "two cell types share more recent common ancestors than expected by chance (they tend "
    "to arise from a common progenitor pool or clone), and a negative value means less "
    "shared ancestry than expected. You will be given a pair of cell types with a "
    "significant, positive linkage. Using your knowledge of mouse gastrulation and early "
    "organogenesis (germ layers, lineage hierarchies, known bipotent/multipotent "
    "progenitors, and spatial/temporal co-emergence), do two things:\n"
    "1. proposed_explanation: ONE sentence hypothesising why these two cell types share "
    "ancestry (e.g. a shared progenitor, a common germ-layer origin, or co-emergence from "
    "the same embryonic region). Name the relevant progenitor or lineage when you can.\n"
    "2. novelty: an integer from 1 to 5 rating how surprising this shared ancestry is:\n"
    "   1 = textbook / well-established shared lineage (e.g. two neuronal subtypes from "
    "neural progenitors);\n"
    "   2 = expected given known biology but not always stated explicitly;\n"
    "   3 = plausible but not well characterised;\n"
    "   4 = surprising, only weakly supported by existing literature;\n"
    "   5 = totally novel / unexpected, no obvious lineage explanation.\n"
    "Base novelty on the biology, not on the statistics. Be specific and concise."
)


def build_prompt(row: dict) -> str:
    """Build the prompt describing a linked cell-type pair and its linkage statistics."""
    return (
        f"Cell type A: {row['source']}\n"
        f"Cell type B: {row['target']}\n"
        f"Ancestral linkage (norm_value): {row['norm_value']}\n"
        f"z-score: {row['z_score']}\n"
        f"p-value: {row['p_value']}\n"
        f"Number of cells (A): {row['source_n']}\n"
        f"Number of cells (B): {row['target_n']}\n\n"
        f"These two cell types share significantly more ancestry than expected by chance.\n"
        f"Provide:\n"
        f"1. proposed_explanation: one sentence on why they share ancestry.\n"
        f"2. novelty: integer 1 (well-established) to 5 (totally novel)."
    )


def annotate_pair(row: dict, model: str, max_retries: int = 3) -> dict:
    """Query Claude for a proposed explanation and novelty score for one cell-type pair.

    Retries up to ``max_retries`` times with exponential backoff and raises
    ``RuntimeError`` if every attempt fails.
    """
    prompt = build_prompt(row)
    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                [
                    "claude",
                    "--print",
                    "--output-format",
                    "json",
                    "--json-schema",
                    JSON_SCHEMA,
                    "--append-system-prompt",
                    SYSTEM_PROMPT,
                    "--no-session-persistence",
                    "--model",
                    model,
                    prompt,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                raise RuntimeError(f"claude exited with code {result.returncode}: {result.stderr}")

            outer = json.loads(result.stdout)
            if outer.get("is_error"):
                raise RuntimeError(f"claude returned error: {outer.get('result', result.stdout)}")

            structured = outer.get("structured_output")
            if not structured:
                content = outer.get("result", "")
                if not content:
                    raise RuntimeError(f"Unexpected claude output: {result.stdout[:200]}")
                structured = json.loads(content)

            return {
                "proposed_explanation": structured["proposed_explanation"],
                "novelty": int(structured["novelty"]),
            }

        except subprocess.TimeoutExpired:
            print(f"  Timeout on attempt {attempt + 1}. Retrying...")
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            print(f"  Parse error on attempt {attempt + 1}: {e}")
        except RuntimeError as e:
            print(f"  Error on attempt {attempt + 1}: {e}")

        if attempt < max_retries - 1:
            time.sleep(2**attempt)

    raise RuntimeError(f"Failed to annotate after {max_retries} attempts")


def _norm_value(row: dict) -> float:
    try:
        return float(row["norm_value"])
    except (ValueError, KeyError, TypeError):
        return float("-inf")


def qualifies(row: dict, min_linkage: float) -> bool:
    """A pair is annotated iff it is significant, positive above cutoff, non-self."""
    try:
        pval = float(row["p_value"])
    except (ValueError, KeyError, TypeError):
        return False
    return row["source"] != row["target"] and pval < 0.05 and _norm_value(row) >= min_linkage


def load_and_select(input_path: Path, min_linkage: float):
    """Load all rows, clear stale annotations, and select qualifying pairs.

    Returns (fieldnames, all_rows, selected_rows, n_cleared). Every statistically
    significant, positive pair with norm_value >= `min_linkage` (excluding
    self-pairs) is selected. Annotations on rows that do not qualify are cleared,
    so only qualifying pairs ever carry an annotation. All rows (and the selected
    subset) are sorted by norm_value descending.
    """
    with open(input_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    # Drop obsolete columns from earlier selection schemes.
    for col in OBSOLETE_COLUMNS:
        if col in fieldnames:
            fieldnames.remove(col)
        for row in rows:
            row.pop(col, None)

    # Ensure the annotation columns exist on every row.
    for col in NEW_COLUMNS:
        if col not in fieldnames:
            fieldnames.append(col)
    for row in rows:
        for col in NEW_COLUMNS:
            row.setdefault(col, "")

    selected = []
    n_cleared = 0
    for row in rows:
        if qualifies(row, min_linkage):
            selected.append(row)
        else:
            # Remove annotations for pairs that no longer satisfy the criteria.
            if (row.get("proposed_explanation") or "").strip():
                n_cleared += 1
            row["proposed_explanation"] = ""
            row["novelty"] = ""

    # Reorder the whole file (and the selected subset) by linkage descending.
    rows.sort(key=_norm_value, reverse=True)
    selected.sort(key=_norm_value, reverse=True)
    return fieldnames, rows, selected, n_cleared


def is_annotated(row: dict) -> bool:
    """Return True if the row has a (non-failed) proposed explanation and a novelty score."""
    expl = (row.get("proposed_explanation") or "").strip()
    nov = (row.get("novelty") or "").strip()
    return bool(expl) and bool(nov) and expl != FAILED_MARKER


def write_csv(path: Path, fieldnames, rows) -> None:
    """Atomically write all rows to path, preserving column order."""
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def pair_key(row: dict) -> tuple:
    """Return the ``(source, target)`` key identifying a cell-type pair."""
    return (row["source"], row["target"])


def main():
    """Select significant positive linkage pairs, annotate them with Claude, and write the CSV."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="claude-opus-4-8", help="Claude model alias (default: claude-opus-4-8)")
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT, help=f"Linkage CSV to select from (default: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Annotated CSV to write (default: modify the input in place)"
    )
    parser.add_argument(
        "--min-linkage", type=float, default=0.5, help="Minimum norm_value cutoff to annotate (default: 0.5)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without calling Claude")
    args = parser.parse_args()

    output = args.output or args.input

    fieldnames, all_rows, selected, n_cleared = load_and_select(args.input, args.min_linkage)
    print(
        f"Selected {len(selected)} significant positive pairs "
        f"(p<0.05, norm_value>={args.min_linkage}) from {args.input}"
    )
    if n_cleared:
        print(f"Cleared {n_cleared} annotations on pairs that no longer qualify")

    # Resume: when writing to a separate, pre-existing output, carry over its
    # annotations. (When modifying in place, annotations are already in all_rows.)
    if args.output and output != args.input and output.exists():
        with open(output, newline="") as f:
            prior = {pair_key(r): r for r in csv.DictReader(f)}
        carried = 0
        for row in selected:
            old = prior.get(pair_key(row))
            if old and is_annotated(old):
                row["proposed_explanation"] = old["proposed_explanation"]
                row["novelty"] = old["novelty"]
                carried += 1
        print(f"Carried over {carried} existing annotations from {output}")

    todo = [row for row in selected if not is_annotated(row)]
    print(f"Annotating {len(todo)} pairs (writing to {output})...")

    errors = []
    for i, row in enumerate(todo, 1):
        print(
            f"[{i}/{len(todo)}] {row['source']} <-> {row['target']} " f"(norm={float(row['norm_value']):.3f})",
            end=" ",
            flush=True,
        )

        if args.dry_run:
            print("\n" + build_prompt(row) + "\n---")
            continue

        try:
            ann = annotate_pair(row, model=args.model)
            print(f"-> [nov {ann['novelty']}] {ann['proposed_explanation'][:70]}...", flush=True)
            row["proposed_explanation"] = ann["proposed_explanation"]
            row["novelty"] = str(ann["novelty"])
        except Exception as e:  # noqa: BLE001
            print(f"ERROR: {e}", flush=True)
            errors.append((pair_key(row), str(e)))
            row["proposed_explanation"] = FAILED_MARKER
            row["novelty"] = ""

        # Persist after every row so progress survives interruptions.
        write_csv(output, fieldnames, all_rows)

    if args.dry_run:
        return

    print(f"\nDone. Results written to {output}")
    if errors:
        print(f"\n{len(errors)} errors:")
        for key, err in errors:
            print(f"  {key}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
