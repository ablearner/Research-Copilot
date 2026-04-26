from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.metrics import informative_tokens  # noqa: E402


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert chart VQA JSONL files into Research-Copilot evaluation cases."
    )
    parser.add_argument("--input", required=True, help="Source VQA JSONL file.")
    parser.add_argument("--output", required=True, help="Output evaluation cases JSON file.")
    parser.add_argument(
        "--base-root",
        default=None,
        help="Root used to resolve relative image paths. Defaults to an inferred dataset root.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of cases to write.",
    )
    parser.add_argument(
        "--first-turn-only",
        action="store_true",
        help="Only convert the first QA turn from each source item.",
    )
    parser.add_argument(
        "--skip-missing-images",
        action="store_true",
        help="Skip items whose image file cannot be found after path resolution.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy referenced images into this project and write project-local image paths.",
    )
    parser.add_argument(
        "--asset-dir",
        default=None,
        help="Directory for copied images. Defaults to <output-dir>/<output-stem>_images.",
    )
    parser.add_argument(
        "--skill-name",
        default="chart_qa",
        help="Optional skill_name to place on generated cases.",
    )
    parser.add_argument(
        "--reasoning-style",
        default=None,
        help="Optional reasoning_style to place on generated cases, for example react.",
    )
    parser.add_argument(
        "--min-keyword-recall",
        type=float,
        default=0.5,
        help="Minimum expected keyword recall used by Task Success Rate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    base_root = Path(args.base_root).expanduser().resolve() if args.base_root else infer_base_root(input_path)
    asset_dir = (
        Path(args.asset_dir).expanduser().resolve()
        if args.asset_dir
        else output_path.parent / f"{output_path.stem}_images"
    )

    cases = convert_jsonl(
        input_path=input_path,
        base_root=base_root,
        limit=args.limit,
        first_turn_only=args.first_turn_only,
        skip_missing_images=args.skip_missing_images,
        copy_images=args.copy_images,
        asset_dir=asset_dir,
        skill_name=args.skill_name,
        reasoning_style=args.reasoning_style,
        min_keyword_recall=args.min_keyword_recall,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"cases": cases}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "input": str(input_path),
                "output": str(output_path),
                "base_root": str(base_root),
                "asset_dir": str(asset_dir) if args.copy_images else None,
                "case_count": len(cases),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def convert_jsonl(
    *,
    input_path: Path,
    base_root: Path,
    limit: int | None,
    first_turn_only: bool,
    skip_missing_images: bool,
    copy_images: bool,
    asset_dir: Path,
    skill_name: str | None,
    reasoning_style: str | None,
    min_keyword_recall: float,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    copied_images: dict[Path, Path] = {}
    with input_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if limit is not None and len(cases) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            image_path = resolve_image_path(item.get("image"), base_root=base_root)
            if skip_missing_images and (image_path is None or not image_path.exists()):
                continue
            if copy_images and image_path is not None:
                if not image_path.exists():
                    raise FileNotFoundError(f"Image not found for item at line {line_number}: {image_path}")
                source_id = str(item.get("source_id") or item.get("id") or f"line-{line_number}")
                image_path = copy_image_once(
                    image_path=image_path,
                    asset_dir=asset_dir,
                    stable_name=sanitize_case_id(source_id),
                    copied_images=copied_images,
                )
            for turn_index, turn in enumerate(extract_turns(item), start=1):
                if limit is not None and len(cases) >= limit:
                    break
                if first_turn_only and turn_index > 1:
                    break
                case = build_case(
                    item=item,
                    turn=turn,
                    turn_index=turn_index,
                    line_number=line_number,
                    image_path=image_path,
                    skill_name=skill_name,
                    reasoning_style=reasoning_style,
                    min_keyword_recall=min_keyword_recall,
                )
                if case:
                    cases.append(case)
    return cases


def infer_base_root(input_path: Path) -> Path:
    if input_path.parent.name == "benchmark_eval" and input_path.parent.parent.name == "data":
        return input_path.parent.parent.parent
    if input_path.parent.name in {"train", "test", "valid", "val"}:
        return input_path.parent.parent.parent
    return input_path.parent


def resolve_image_path(image: Any, *, base_root: Path) -> Path | None:
    if not image:
        return None
    image_path = Path(str(image))
    if image_path.is_absolute():
        return image_path
    candidates = [
        base_root / image_path,
        base_root / "data" / image_path,
        base_root / "images" / image_path,
    ]
    for candidate in candidates:
        if candidate.suffix.lower() in _IMAGE_EXTENSIONS and candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def copy_image_once(
    *,
    image_path: Path,
    asset_dir: Path,
    stable_name: str,
    copied_images: dict[Path, Path],
) -> Path:
    resolved = image_path.resolve()
    if resolved in copied_images:
        return copied_images[resolved]
    asset_dir.mkdir(parents=True, exist_ok=True)
    suffix = resolved.suffix.lower() or ".png"
    target = asset_dir / f"{stable_name}{suffix}"
    shutil.copy2(resolved, target)
    copied_images[resolved] = target.resolve()
    return copied_images[resolved]


def extract_turns(item: dict[str, Any]) -> list[dict[str, Any]]:
    turns = item.get("turns")
    if isinstance(turns, list) and turns:
        return [turn for turn in turns if isinstance(turn, dict)]
    question = item.get("question") or item.get("query") or item.get("prompt")
    answer = item.get("answer") or item.get("reference_answer") or item.get("label")
    if question is None or answer is None:
        return []
    return [{"question": question, "answer": answer}]


def build_case(
    *,
    item: dict[str, Any],
    turn: dict[str, Any],
    turn_index: int,
    line_number: int,
    image_path: Path | None,
    skill_name: str | None,
    reasoning_style: str | None,
    min_keyword_recall: float,
) -> dict[str, Any] | None:
    question = clean_text(turn.get("question"))
    reference_answer = clean_text(turn.get("answer"))
    if not question or not reference_answer or image_path is None:
        return None

    source_id = str(item.get("source_id") or item.get("id") or f"line-{line_number}")
    case_id = source_id if turn_index == 1 else f"{source_id}-turn-{turn_index}"
    expected_keywords = expected_keywords_from_answer(reference_answer)

    case: dict[str, Any] = {
        "id": sanitize_case_id(case_id),
        "kind": "ask_fused",
        "question": question,
        "document_id": source_id,
        "image_path": project_local_path(image_path),
        "expected_route": "ask_fused",
        "expected_keywords": expected_keywords,
        "min_keyword_recall": min_keyword_recall,
        "grounding_keywords": expected_keywords,
        "expected_tool_names": ["ChartAgent.ask_chart"],
        "require_evidence": False,
        "metadata": {
            "dataset": item.get("dataset"),
            "dataset_split": item.get("dataset_split"),
            "source_file": item.get("source_file"),
            "source_id": source_id,
            "turn_index": turn_index,
            "reference_answer": reference_answer,
            "context": item.get("context") or {},
            "history": item.get("history") or turn.get("history") or [],
        },
    }
    if skill_name:
        case["skill_name"] = skill_name
    if reasoning_style:
        case["reasoning_style"] = reasoning_style
    return case


def expected_keywords_from_answer(answer: str) -> list[str]:
    normalized = clean_text(answer)
    if not normalized:
        return []
    tokens = informative_tokens(normalized)
    if len(normalized) <= 48 or len(tokens) <= 4:
        return [normalized]
    return tokens[:8]


def clean_text(value: Any) -> str:
    if isinstance(value, list):
        value = " ".join(str(item) for item in value)
    return re.sub(r"\s+", " ", str(value or "")).strip()


def project_local_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def sanitize_case_id(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.:-]+", "-", value).strip("-")
    return sanitized or "case"


if __name__ == "__main__":
    main()
