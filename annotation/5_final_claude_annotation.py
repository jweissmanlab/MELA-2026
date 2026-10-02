#!/usr/bin/env python3
"""
Annotate cell types/subtypes using headless Claude Code.

Reads results/cell_types_annotated.csv and queries Claude to add a biological
description and naming rationale to each row that does not already have one.
Rows that already contain a description and rationale are left untouched — the
file is updated in place and existing annotations are never overwritten.

Usage:
    python 5_final_claude_annotation.py
    python 5_final_claude_annotation.py --model claude-opus-4-7   # override model
    python 5_final_claude_annotation.py --input results/other.csv # override file
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

DEFAULT_CSV = Path(__file__).parent / "results" / "cell_types_annotated.csv"

FAILED_MARKER = "Annotation failed after retries."

JSON_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "description": {"type": "string"},
            "rationale": {"type": "string"},
        },
        "required": ["description", "rationale"],
        "additionalProperties": False,
    }
)

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
    """Build the prompt for a cell type/subtype row from its name, lineage, stages, and markers."""
    subtype = (row.get("cell_subtype") or "").strip()
    cell_type = (row.get("cell_type") or "").strip()
    # Parent cell-type rows have no cell_subtype; annotate the cell type itself.
    if subtype:
        name_lines = f"Cell subtype: {subtype}\nParent cell type: {cell_type}\n"
    else:
        name_lines = f"Cell type: {cell_type}\n"
    return (
        f"{name_lines}"
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
    """Query Claude for a description and naming rationale for one row.

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
            time.sleep(2**attempt)

    raise RuntimeError(f"Failed to annotate after {max_retries} attempts")


def id_column(fieldnames) -> str:
    """Return the row-identifier column, preferring 'id' over legacy 'subtype_id'."""
    for candidate in ("id", "subtype_id"):
        if candidate in fieldnames:
            return candidate
    raise KeyError("Input CSV must have an 'id' or 'subtype_id' column")


def is_annotated(row: dict) -> bool:
    """A row is already annotated if it has both a description and rationale."""
    desc = (row.get("description") or "").strip()
    rat = (row.get("rationale") or "").strip()
    return bool(desc) and bool(rat) and desc != FAILED_MARKER


def write_csv(path: Path, fieldnames, rows) -> None:
    """Atomically write all rows to path, preserving column order."""
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def main():
    """Annotate unannotated rows of the input CSV in place, preserving existing annotations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="claude-opus-4-6", help="Claude model alias (default: claude-opus-4-6)")
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_CSV, help=f"CSV to annotate in place (default: {DEFAULT_CSV})"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without calling Claude")
    args = parser.parse_args()

    csv_path = args.input
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)
    id_col = id_column(fieldnames)
    print(f"Loaded {len(rows)} rows from {csv_path}")

    # Ensure the annotation columns exist without disturbing existing values/order.
    for col in ("description", "rationale"):
        if col not in fieldnames:
            fieldnames.append(col)
            for row in rows:
                row.setdefault(col, "")

    already = sum(1 for row in rows if is_annotated(row))
    todo = [row for row in rows if not is_annotated(row)]
    print(f"{already} rows already annotated (preserved); annotating {len(todo)} rows...")

    errors = []
    for i, row in enumerate(todo, 1):
        rid = row[id_col]
        name = (row.get("cell_subtype") or "").strip() or row.get("cell_type", "")
        print(f"[{i}/{len(todo)}] {rid}: {name}", end=" ", flush=True)

        if args.dry_run:
            print("\n" + build_prompt(row) + "\n---")
            continue

        try:
            annotation = annotate_subtype(row, model=args.model)
            print(f"-> {annotation['description'][:80]}...", flush=True)
            row["description"] = annotation["description"]
            row["rationale"] = annotation["rationale"]
        except Exception as e:  # noqa: BLE001
            print(f"ERROR: {e}", flush=True)
            errors.append((rid, str(e)))
            row["description"] = FAILED_MARKER
            row["rationale"] = str(e)

        # Persist after every row so progress survives interruptions.
        write_csv(csv_path, fieldnames, rows)

    if args.dry_run:
        return

    print(f"\nDone. Results written to {csv_path}")
    if errors:
        print(f"\n{len(errors)} errors:")
        for rid, err in errors:
            print(f"  {rid}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
