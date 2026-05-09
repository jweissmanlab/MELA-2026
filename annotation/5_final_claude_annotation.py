#!/usr/bin/env python3
"""
Annotate cell subtypes using headless Claude Code.

Reads results/cell_type_info.csv and queries Claude to add a biological
description and naming rationale to each cell subtype. Writes results to
results/cell_subtypes_annotated.csv.

Usage:
    python 5_final_claude_annotation.py
    python 5_final_claude_annotation.py --model claude-opus-4-7   # override model
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

INPUT_CSV = Path(__file__).parent / "results" / "cell_type_info.csv"
OUTPUT_CSV = Path(__file__).parent / "results" / "cell_subtypes_annotated.csv"

JSON_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "rationale":   {"type": "string"},
    },
    "required": ["description", "rationale"],
    "additionalProperties": False,
})

SYSTEM_PROMPT = (
    "You are an expert developmental biologist and single-cell genomics researcher. "
    "You are annotating manually-curated cell subtypes from a mouse embryo scRNA-seq "
    "dataset spanning stages E7.5 to E10.0. Each subtype has a name already assigned "
    "by the researcher; your job is to (1) explain what this cell type is biologically "
    "and (2) justify the chosen name based on the provided marker genes. "
    "Global markers are the top differentially-expressed genes across the entire dataset. "
    "Local markers are the top differentially-expressed genes relative to the three most "
    "similar cell types in the dataset. "
    "Write 1-3 sentences for description and 2-4 sentences for rationale. "
    "Be specific — cite actual marker genes from the lists when justifying the name."
)


def build_prompt(row: dict) -> str:
    return (
        f"Cell subtype: {row['cell_subtype']}\n"
        f"Parent cell type: {row['cell_type']}\n"
        f"Lineage: {row['lineage']}\n"
        f"Germ layer: {row['germ_layer']}\n"
        f"Stages present: {row['stages']}\n"
        f"Number of cells: {row['n_cells']}\n"
        f"Global markers: {row['global_markers']}\n"
        f"Local markers: {row['local_markers']}\n\n"
        f"Provide:\n"
        f"1. description: 1-3 sentences explaining what this cell type is biologically.\n"
        f"2. rationale: 2-4 sentences explaining why this name was chosen based on the marker genes."
    )


def annotate_subtype(row: dict, model: str, max_retries: int = 3) -> dict:
    prompt = build_prompt(row)
    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                [
                    "claude",
                    "--print",
                    "--output-format", "json",
                    "--json-schema", JSON_SCHEMA,
                    "--append-system-prompt", SYSTEM_PROMPT,
                    "--no-session-persistence",
                    "--model", model,
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
            if structured:
                return {"description": structured["description"], "rationale": structured["rationale"]}
            content = outer.get("result", "")
            if content:
                parsed = json.loads(content)
                return {"description": parsed["description"], "rationale": parsed["rationale"]}
            raise RuntimeError(f"Unexpected claude output: {result.stdout[:200]}")

        except subprocess.TimeoutExpired:
            print(f"  Timeout on attempt {attempt + 1}. Retrying...")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Parse error on attempt {attempt + 1}: {e}")
        except RuntimeError as e:
            print(f"  Error on attempt {attempt + 1}: {e}")

        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)

    raise RuntimeError(f"Failed to annotate after {max_retries} attempts")


def load_existing_annotations(path: Path) -> set:
    """Load already-processed rows keyed by subtype_id."""
    if not path.exists():
        return set()
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return {row["subtype_id"] for row in reader if row.get("description") and row["description"] != "Annotation failed after retries."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="claude-opus-4-6",
                        help="Claude model alias (default: claude-opus-4-6)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompts without calling Claude")
    args = parser.parse_args()

    with open(INPUT_CSV, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        subtypes = list(reader)
    print(f"Loaded {len(subtypes)} subtypes from {INPUT_CSV}")

    existing = load_existing_annotations(OUTPUT_CSV)
    if existing:
        print(f"Resuming: {len(existing)} subtypes already annotated.")

    out_fieldnames = list(fieldnames) + ["description", "rationale"]
    write_header = not OUTPUT_CSV.exists() or len(existing) == 0
    out_f = open(OUTPUT_CSV, "a" if not write_header else "w", newline="")
    writer = csv.DictWriter(out_f, fieldnames=out_fieldnames)
    if write_header:
        writer.writeheader()
        for row in subtypes:
            if row["subtype_id"] in existing:
                writer.writerow(row)
        out_f.flush()

    todo = [s for s in subtypes if s["subtype_id"] not in existing]
    print(f"Annotating {len(todo)} subtypes...")

    errors = []
    try:
        for i, row in enumerate(todo, 1):
            sid = row["subtype_id"]
            print(f"[{i}/{len(todo)}] {sid}: {row['cell_subtype']}", end=" ", flush=True)

            if args.dry_run:
                print("\n" + build_prompt(row) + "\n---")
                continue

            try:
                annotation = annotate_subtype(row, model=args.model)
                print(f"-> {annotation['description'][:80]}...", flush=True)
                writer.writerow({**row, "description": annotation["description"], "rationale": annotation["rationale"]})
                out_f.flush()
                existing.add(sid)
            except Exception as e:
                print(f"ERROR: {e}", flush=True)
                errors.append((sid, str(e)))
                writer.writerow({**row, "description": "Annotation failed after retries.", "rationale": str(e)})
                out_f.flush()
    finally:
        out_f.close()

    print(f"\nDone. Results written to {OUTPUT_CSV}")
    if errors:
        print(f"\n{len(errors)} errors:")
        for sid, err in errors:
            print(f"  {sid}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
