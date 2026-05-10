#!/usr/bin/env python3
"""
Annotate gene programs using headless Claude Code.

Reads results/gene_programs.csv and queries Claude Opus to assign a short name
(<30 characters) and biological description to each program. Writes results
to results/annotated_programs.csv.

Usage:
    python annotate_programs.py
    python annotate_programs.py --resume   # skip already-annotated programs
"""

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

INPUT_CSV = Path("results/gene_programs.csv")
OUTPUT_CSV = Path("results/annotated_programs.csv")
MODEL = "claude-opus-4-6"

JSON_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "name":        {"type": "string"},
        "description": {"type": "string"},
    },
    "required": ["name", "description"],
    "additionalProperties": False,
})

# Tools to block so Claude doesn't read local files during annotation
DISALLOWED_TOOLS = "Bash,Edit,Write,Glob,Grep,Read,Agent,WebSearch,WebFetch"

SYSTEM_CONTEXT = (
    "You are an expert developmental biologist and single-cell genomics researcher. "
    "You are annotating heritable gene expression programs identified by lineage tracing "
    "in mouse embryos from stages E7.5 to E10. These programs drive cell type "
    "diversification and are defined by their co-expressed genes and the cell types in "
    "which they are most active. "
    "Focus on the dominant biological theme: signaling pathways, transcription factor "
    "programs, cell type identity, or developmental processes. "
    "name must be ≤30 characters; abbreviations are encouraged."
)


def build_prompt(program_id: str, genes: str, active_in: str, size: int) -> str:
    return (
        f"{SYSTEM_CONTEXT}\n\n"
        f"Program: {program_id}\n"
        f"Size: {size} genes\n"
        f"Active in: {active_in}\n"
        f"Genes: {genes}\n\n"
        f"Provide a concise biological name (≤30 chars) and a 1-3 sentence description "
        f"of what this gene program does."
    )


def annotate_program(
    program_id: str,
    genes: str,
    active_in: str,
    size: int,
    max_retries: int = 3,
) -> dict:
    prompt = build_prompt(program_id, genes, active_in, size)
    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                [
                    "claude", "-p", prompt,
                    "--model", MODEL,
                    "--output-format", "json",
                    "--no-session-persistence",
                    "--disallowed-tools", DISALLOWED_TOOLS,
                    "--json-schema", JSON_SCHEMA,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or f"exit code {result.returncode}")

            response = json.loads(result.stdout)

            if response.get("is_error"):
                raise RuntimeError(response.get("result", "unknown error"))

            annotation = response.get("structured_output")
            if not annotation:
                # Fallback: try parsing result text as JSON
                text = response.get("result", "").strip()
                if text.startswith("```"):
                    text = text.split("```")[1].lstrip("json").strip()
                annotation = json.loads(text)

            name = annotation["name"][:30]
            description = annotation["description"]
            return {"name": name, "description": description}

        except subprocess.TimeoutExpired:
            print(f"  Timeout on attempt {attempt + 1}. Retrying...")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Parse error on attempt {attempt + 1}: {e}")
        except RuntimeError as e:
            print(f"  Error on attempt {attempt + 1}: {e}")

        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)

    return {
        "name": f"{program_id}_unannotated",
        "description": "Annotation failed after retries.",
    }


def load_existing_annotations(path: Path) -> set:
    """Return set of already-annotated program IDs."""
    if not path.exists():
        return set()
    with open(path, newline="") as f:
        return {row["program"] for row in csv.DictReader(f)}


def main():
    parser = argparse.ArgumentParser(description="Annotate gene programs with Claude.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip programs already present in output CSV.")
    args = parser.parse_args()

    programs = []
    with open(INPUT_CSV, newline="") as f:
        for row in csv.DictReader(f):
            programs.append(row)
    print(f"Loaded {len(programs)} programs from {INPUT_CSV}")

    existing = load_existing_annotations(OUTPUT_CSV) if args.resume else set()
    if existing:
        print(f"Resuming: {len(existing)} programs already annotated.")

    fieldnames = ["program", "genes", "size", "active_in", "name", "description"]
    mode = "a" if (args.resume and OUTPUT_CSV.exists()) else "w"
    out_f = open(OUTPUT_CSV, mode, newline="")
    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
    if mode == "w":
        writer.writeheader()

    try:
        todo = [p for p in programs if p["program"] not in existing]
        print(f"Annotating {len(todo)} programs...")

        for i, row in enumerate(todo, 1):
            pid = row["program"]
            print(f"[{i}/{len(todo)}] {pid} "
                  f"(size={row['size']}, active_in={row['active_in'][:60]}...)")

            annotation = annotate_program(
                pid,
                row["genes"],
                row["active_in"],
                int(row["size"]),
            )
            print(f"  -> {annotation['name']}")

            writer.writerow({
                "program":     pid,
                "genes":       row["genes"],
                "size":        row["size"],
                "active_in":   row["active_in"],
                "name":        annotation["name"],
                "description": annotation["description"],
            })
            out_f.flush()
    finally:
        out_f.close()

    print(f"\nDone. Results written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
