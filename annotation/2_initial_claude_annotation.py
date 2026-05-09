#!/usr/bin/env python3
"""
Annotate scRNA-seq clusters using Claude headlessly.

For each cluster in subclusters.csv, calls claude -p with marker gene and
linked cell type information to predict a cell type and provide rationale.
Results are saved incrementally to avoid losing progress on failure.
"""

import csv
import json
import subprocess
import sys
import os
import argparse
from pathlib import Path

INPUT_CSV = Path(__file__).parent / "results" / "subclusters.csv"
OUTPUT_CSV = Path(__file__).parent / "results" / "subclusters_annotated.csv"

JSON_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "predicted_cell_type": {
            "type": "string",
            "description": "Concise cell type name (e.g. 'cardiac neural crest cell')"
        },
        "rationale": {
            "type": "string",
            "description": "2-4 sentence explanation citing specific marker genes and linked cell types"
        }
    },
    "required": ["predicted_cell_type", "rationale"]
})

SYSTEM_PROMPT = (
    "You are an expert in mouse embryo development and single-cell RNA sequencing. "
    "You will be given information about a cluster of cells from a mouse embryo scRNA-seq "
    "dataset spanning E7.5 to E10. Your task is to predict the cell type based on "
    "marker genes and developmental context. Use standard developmental biology "
    "nomenclature. Be specific where the evidence supports it."
)

def build_prompt(row: dict) -> str:
    time_val = float(row["time"])
    # Convert numeric time to approximate embryonic day string
    time_str = f"E{time_val:.1f}"

    return f"""Predict the cell type for the following mouse embryo scRNA-seq cluster.

Developmental time: {time_str} (average pseudotime/stage of cells in this cluster)

Global marker genes (highly expressed relative to all other cells):
{row['global_markers']}

Local marker genes (highly expressed relative to similar/neighboring clusters):
{row['local_markers']}

Linked cell types (clusters that likely share a developmental origin with this one):
{row['linked_cell_types']}

Based on the marker genes and developmental context, predict the cell type and provide your rationale. \
Reference specific marker genes and linked cell types in your reasoning."""


def call_claude(prompt: str, model: str = "sonnet") -> dict:
    cmd = [
        "claude",
        "--print",
        "--output-format", "json",
        "--json-schema", JSON_SCHEMA,
        "--append-system-prompt", SYSTEM_PROMPT,
        "--no-session-persistence",
        "--model", model,
        prompt,
    ]
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude exited with code {result.returncode}: {result.stderr}")

    # --output-format json wraps the response; structured output is in "structured_output"
    outer = json.loads(result.stdout)
    if outer.get("is_error"):
        raise RuntimeError(f"claude returned error: {outer.get('result', result.stdout)}")
    structured = outer.get("structured_output")
    if structured:
        return structured
    # Fallback: try to parse the result field as JSON
    content = outer.get("result", "")
    if content:
        return json.loads(content)
    raise RuntimeError(f"Unexpected claude output: {result.stdout[:200]}")


def load_existing_results(output_path: Path) -> dict:
    """Load already-processed rows keyed by subcluster id."""
    if not output_path.exists():
        return {}
    results = {}
    with open(output_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pt = row.get("predicted_cell_type", "")
            if pt and pt != "ERROR":
                results[row["subcluster"]] = {
                    "predicted_cell_type": pt,
                    "rationale": row["rationale"],
                }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="claude-opus-4-5",
                        help="Claude model alias (default: claude-opus-4-5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompts without calling claude")
    parser.add_argument("--start-at", type=int, default=0,
                        help="Skip the first N rows (0-indexed, for debugging)")
    args = parser.parse_args()

    # Read all input rows
    with open(INPUT_CSV, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    # Load any previously saved results
    existing = load_existing_results(OUTPUT_CSV)
    print(f"Loaded {len(existing)} previously annotated clusters.", flush=True)

    # Determine output fieldnames
    out_fieldnames = list(fieldnames) + ["predicted_cell_type", "rationale"]

    # Open output file in append or write mode
    write_header = not OUTPUT_CSV.exists() or len(existing) == 0
    out_file = open(OUTPUT_CSV, "a" if not write_header else "w", newline="")
    writer = csv.DictWriter(out_file, fieldnames=out_fieldnames)
    if write_header:
        writer.writeheader()
        # Write rows that already have annotations
        for row in rows:
            subcluster = row["subcluster"]
            if subcluster in existing:
                out_row = dict(row)
                out_row["predicted_cell_type"] = existing[subcluster]["predicted_cell_type"]
                out_row["rationale"] = existing[subcluster]["rationale"]
                writer.writerow(out_row)
        out_file.flush()

    total = len(rows)
    errors = []

    for i, row in enumerate(rows):
        if i < args.start_at:
            continue

        subcluster = row["subcluster"]

        # Skip if already processed
        if subcluster in existing:
            print(f"[{i+1}/{total}] Skipping {subcluster} (already annotated)", flush=True)
            continue

        print(f"[{i+1}/{total}] Annotating cluster {subcluster} ...", end=" ", flush=True)

        if args.dry_run:
            print("\n" + build_prompt(row) + "\n---")
            continue

        try:
            prompt = build_prompt(row)
            annotation = call_claude(prompt, model=args.model)
            predicted_cell_type = annotation["predicted_cell_type"]
            rationale = annotation["rationale"]
            print(f"-> {predicted_cell_type}", flush=True)

            out_row = dict(row)
            out_row["predicted_cell_type"] = predicted_cell_type
            out_row["rationale"] = rationale
            writer.writerow(out_row)
            out_file.flush()

            # Track so we don't re-process if script is resumed
            existing[subcluster] = {
                "predicted_cell_type": predicted_cell_type,
                "rationale": rationale,
            }

        except Exception as e:
            print(f"ERROR: {e}", flush=True)
            errors.append((subcluster, str(e)))
            # Write a placeholder so we can identify failures
            out_row = dict(row)
            out_row["predicted_cell_type"] = "ERROR"
            out_row["rationale"] = str(e)
            writer.writerow(out_row)
            out_file.flush()

    out_file.close()

    print(f"\nDone. Results written to {OUTPUT_CSV}")
    if errors:
        print(f"\n{len(errors)} errors:")
        for subcluster, err in errors:
            print(f"  {subcluster}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
