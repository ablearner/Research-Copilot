from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download, list_repo_files

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.metrics import informative_tokens  # noqa: E402
from evaluation.schemas import EvaluationCase  # noqa: E402


DEFAULT_OUTPUT_ROOT = ROOT / "evaluation" / "benchmarks"
RAGBENCH_REPO_ALIASES = ("galileo-ai/ragbench", "rungalileo/ragbench")
SPECIAL_RAGBENCH_SUPPORT_KEYS = {
    "general",
    "numerical_reasoning",
    "supported_without_sentence",
    "well_known_fact",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Research-Copilot evaluation sets from BEIR SciFact and RAGBench."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--skip-scifact", action="store_true")
    parser.add_argument("--skip-ragbench", action="store_true")
    parser.add_argument("--scifact-repo-id", default="BeIR/scifact")
    parser.add_argument("--scifact-qrels-repo-id", default="BeIR/scifact-qrels")
    parser.add_argument("--scifact-qrels-split", default="test")
    parser.add_argument("--local-scifact-corpus", type=Path)
    parser.add_argument("--local-scifact-queries", type=Path)
    parser.add_argument("--local-scifact-qrels", type=Path)
    parser.add_argument("--limit-scifact-cases", type=int, default=100)
    parser.add_argument(
        "--limit-scifact-corpus",
        type=int,
        default=0,
        help="Optional distractor corpus cap. Relevant documents for selected cases are always kept.",
    )
    parser.add_argument("--ragbench-repo-id", default=RAGBENCH_REPO_ALIASES[0])
    parser.add_argument("--ragbench-subset", default="pubmedqa")
    parser.add_argument("--ragbench-split", default="test")
    parser.add_argument("--local-ragbench-file", type=Path)
    parser.add_argument("--limit-ragbench-cases", type=int, default=100)
    parser.add_argument("--hf-cache-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []

    if not args.skip_scifact:
        summaries.append(build_scifact(args))
    if not args.skip_ragbench:
        summaries.append(build_ragbench(args))

    print(json.dumps({"benchmarks": summaries}, ensure_ascii=False, indent=2))


def build_scifact(args: argparse.Namespace) -> dict[str, Any]:
    corpus_paths = [args.local_scifact_corpus] if args.local_scifact_corpus else download_scifact_files(
        repo_id=args.scifact_repo_id,
        include_tokens=("corpus",),
        suffixes=(".parquet", ".jsonl", ".jsonl.gz"),
        cache_dir=args.hf_cache_dir,
    )
    query_paths = [args.local_scifact_queries] if args.local_scifact_queries else download_scifact_files(
        repo_id=args.scifact_repo_id,
        include_tokens=("queries",),
        suffixes=(".parquet", ".jsonl", ".jsonl.gz"),
        cache_dir=args.hf_cache_dir,
    )
    qrels_paths = [args.local_scifact_qrels] if args.local_scifact_qrels else download_scifact_qrels(
        repo_id=args.scifact_qrels_repo_id,
        split=args.scifact_qrels_split,
        cache_dir=args.hf_cache_dir,
    )

    corpus_rows = read_rows(corpus_paths)
    query_rows = read_rows(query_paths)
    qrel_rows = read_rows(qrels_paths)

    raw_doc_to_project_doc: dict[str, str] = {}
    corpus_by_raw_id: dict[str, dict[str, Any]] = {}
    for row in corpus_rows:
        raw_doc_id = clean_text(first_value(row, "_id", "id", "doc_id", "corpus_id"))
        if not raw_doc_id:
            continue
        project_doc_id = f"scifact_doc_{safe_id(raw_doc_id)}"
        raw_doc_to_project_doc[raw_doc_id] = project_doc_id
        corpus_by_raw_id[raw_doc_id] = row

    qrels_by_query: dict[str, list[str]] = defaultdict(list)
    for row in qrel_rows:
        qid, doc_id, score = parse_qrel_row(row)
        if not qid or not doc_id or score <= 0:
            continue
        if doc_id in raw_doc_to_project_doc and doc_id not in qrels_by_query[qid]:
            qrels_by_query[qid].append(doc_id)

    query_by_id = {
        clean_text(first_value(row, "_id", "id", "query_id", "qid")): row
        for row in query_rows
    }
    selected_qids = [
        qid
        for qid in sorted(qrels_by_query)
        if qid in query_by_id and qrels_by_query[qid]
    ]
    if args.limit_scifact_cases > 0:
        selected_qids = selected_qids[: args.limit_scifact_cases]

    relevant_raw_doc_ids = {
        raw_doc_id
        for qid in selected_qids
        for raw_doc_id in qrels_by_query[qid]
    }
    selected_raw_doc_ids = select_scifact_corpus_ids(
        all_raw_doc_ids=list(corpus_by_raw_id),
        relevant_raw_doc_ids=relevant_raw_doc_ids,
        limit=args.limit_scifact_corpus,
    )

    documents = [
        scifact_document(corpus_by_raw_id[raw_doc_id], raw_doc_to_project_doc[raw_doc_id])
        for raw_doc_id in selected_raw_doc_ids
        if raw_doc_id in corpus_by_raw_id
    ]
    included_project_doc_ids = {document["document_id"] for document in documents}
    cases = []
    for qid in selected_qids:
        query_row = query_by_id[qid]
        expected_source_ids = [
            raw_doc_to_project_doc[raw_doc_id]
            for raw_doc_id in qrels_by_query[qid]
            if raw_doc_to_project_doc.get(raw_doc_id) in included_project_doc_ids
        ]
        if not expected_source_ids:
            continue
        cases.append(
            validate_case(
                {
                    "id": f"scifact_{safe_id(qid)}",
                    "kind": "ask_document",
                    "question": clean_text(first_value(query_row, "text", "query", "title")),
                    "top_k": 5,
                    "expected_route": "ask_document",
                    "expected_source_ids": expected_source_ids,
                    "require_nonempty_answer": False,
                    "require_evidence": True,
                    "metadata": {
                        "benchmark": "beir_scifact",
                        "source_query_id": qid,
                        "source_relevant_doc_ids": qrels_by_query[qid],
                    },
                }
            )
        )

    target = args.output_root / "scifact_v1"
    write_benchmark(
        target=target,
        knowledge_base={
            "metadata": {
                "benchmark": "beir_scifact",
                "source_repo": args.scifact_repo_id,
                "qrels_repo": args.scifact_qrels_repo_id,
                "qrels_split": args.scifact_qrels_split,
                "case_count": len(cases),
                "document_count": len(documents),
            },
            "documents": documents,
        },
        cases=cases,
    )
    return {
        "name": "scifact_v1",
        "path": relpath(target),
        "cases": len(cases),
        "documents": len(documents),
    }


def build_ragbench(args: argparse.Namespace) -> dict[str, Any]:
    if args.local_ragbench_file:
        ragbench_paths = [args.local_ragbench_file]
        repo_id = "local"
    else:
        repo_id, ragbench_paths = download_ragbench_files(
            repo_id=args.ragbench_repo_id,
            subset=args.ragbench_subset,
            split=args.ragbench_split,
            cache_dir=args.hf_cache_dir,
        )
    rows = read_rows(ragbench_paths)
    if args.limit_ragbench_cases > 0:
        rows = rows[: args.limit_ragbench_cases]

    documents: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        mapped = ragbench_case_and_documents(
            row=row,
            row_index=index,
            subset=args.ragbench_subset,
            split=args.ragbench_split,
        )
        if not mapped:
            continue
        cases.append(validate_case(mapped["case"]))
        documents.extend(mapped["documents"])

    target = args.output_root / f"ragbench_{safe_id(args.ragbench_subset)}_{args.ragbench_split}_v1"
    write_benchmark(
        target=target,
        knowledge_base={
            "metadata": {
                "benchmark": "ragbench",
                "source_repo": repo_id,
                "subset": args.ragbench_subset,
                "split": args.ragbench_split,
                "case_count": len(cases),
                "document_count": len(documents),
            },
            "documents": documents,
        },
        cases=cases,
    )
    return {
        "name": target.name,
        "path": relpath(target),
        "cases": len(cases),
        "documents": len(documents),
    }


def download_scifact_files(
    *,
    repo_id: str,
    include_tokens: tuple[str, ...],
    suffixes: tuple[str, ...],
    cache_dir: Path | None,
) -> list[Path]:
    files = list_repo_files(repo_id, repo_type="dataset")
    matches = select_files(files, include_tokens=include_tokens, suffixes=suffixes)
    if not matches:
        raise SystemExit(f"No files matched {include_tokens} in Hugging Face dataset {repo_id}")
    return [
        Path(download_with_retries(repo_id=repo_id, filename=filename, cache_dir=cache_dir))
        for filename in matches
    ]


def download_scifact_qrels(*, repo_id: str, split: str, cache_dir: Path | None) -> list[Path]:
    files = list_repo_files(repo_id, repo_type="dataset")
    suffixes = (".parquet", ".csv", ".tsv", ".txt")
    split_matches = select_files(files, include_tokens=(split,), suffixes=suffixes)
    matches = split_matches or select_files(files, include_tokens=(), suffixes=suffixes)
    if not matches:
        raise SystemExit(f"No qrels files found in Hugging Face dataset {repo_id}")
    return [
        Path(download_with_retries(repo_id=repo_id, filename=filename, cache_dir=cache_dir))
        for filename in matches
    ]


def download_ragbench_files(
    *,
    repo_id: str,
    subset: str,
    split: str,
    cache_dir: Path | None,
) -> tuple[str, list[Path]]:
    errors: list[str] = []
    for candidate_repo in unique_values([repo_id, *RAGBENCH_REPO_ALIASES]):
        try:
            files = list_repo_files(candidate_repo, repo_type="dataset")
            matches = select_files(
                files,
                include_tokens=(subset, split),
                suffixes=(".parquet",),
            )
            if not matches:
                matches = select_files(
                    files,
                    include_tokens=(split,),
                    suffixes=(".parquet",),
                    path_prefix=f"{subset}/",
                )
            if matches:
                return candidate_repo, [
                    Path(
                        download_with_retries(
                            repo_id=candidate_repo,
                            filename=filename,
                            cache_dir=cache_dir,
                        )
                    )
                    for filename in matches
                ]
        except Exception as exc:  # pragma: no cover - depends on network/HF state.
            errors.append(f"{candidate_repo}: {exc}")
    raise SystemExit(
        "No RAGBench parquet files found. Tried: "
        + "; ".join(errors or [repo_id])
    )


def download_with_retries(
    *,
    repo_id: str,
    filename: str,
    cache_dir: Path | None,
    retries: int = 5,
) -> str:
    for attempt in range(1, retries + 1):
        try:
            return hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=filename,
                cache_dir=str(cache_dir) if cache_dir else None,
            )
        except Exception:
            if attempt == retries:
                raise
            time.sleep(min(2**attempt, 20))
    raise RuntimeError("unreachable")


def select_files(
    files: list[str],
    *,
    include_tokens: tuple[str, ...],
    suffixes: tuple[str, ...],
    path_prefix: str | None = None,
) -> list[str]:
    normalized_tokens = [token.lower() for token in include_tokens if token]
    candidates = []
    for filename in files:
        lower = filename.lower()
        if path_prefix and not lower.startswith(path_prefix.lower()):
            continue
        if lower.endswith((".md", ".json", ".lock")):
            continue
        if suffixes and not lower.endswith(suffixes):
            continue
        if all(token in lower for token in normalized_tokens):
            candidates.append(filename)
    return sorted(candidates)


def read_rows(paths: list[Path | None]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if path is None:
            continue
        rows.extend(read_rows_from_path(Path(path)))
    return rows


def read_rows_from_path(path: Path) -> list[dict[str, Any]]:
    suffix = "".join(path.suffixes[-2:]).lower()
    if path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise SystemExit("pyarrow is required to read parquet files") from exc
        table = pq.read_table(path)
        return [dict(row) for row in table.to_pylist()]
    if suffix == ".jsonl.gz":
        import gzip

        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.suffix.lower() in {".csv", ".tsv", ".txt"}:
        delimiter = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, delimiter=delimiter)
            return [dict(row) for row in reader]
    raise SystemExit(f"Unsupported benchmark file format: {path}")


def parse_qrel_row(row: dict[str, Any]) -> tuple[str, str, int]:
    qid = clean_text(
        first_value(
            row,
            "query-id",
            "query_id",
            "queryid",
            "qid",
            "query",
        )
    )
    doc_id = clean_text(
        first_value(
            row,
            "corpus-id",
            "corpus_id",
            "corpusid",
            "doc_id",
            "document_id",
            "pid",
        )
    )
    score = int(float(first_value(row, "score", "relevance", "label") or 1))
    if (not qid or not doc_id) and len(row) >= 3:
        values = list(row.values())
        qid = qid or clean_text(values[0])
        doc_id = doc_id or clean_text(values[1])
        score = int(float(values[2] or 1))
    return qid, doc_id, score


def select_scifact_corpus_ids(
    *,
    all_raw_doc_ids: list[str],
    relevant_raw_doc_ids: set[str],
    limit: int,
) -> list[str]:
    if limit <= 0:
        return all_raw_doc_ids
    selected = all_raw_doc_ids[:limit]
    for raw_doc_id in sorted(relevant_raw_doc_ids):
        if raw_doc_id not in selected:
            selected.append(raw_doc_id)
    return selected


def scifact_document(row: dict[str, Any], document_id: str) -> dict[str, Any]:
    raw_doc_id = clean_text(first_value(row, "_id", "id", "doc_id", "corpus_id"))
    title = clean_text(first_value(row, "title"))
    text = clean_text(first_value(row, "text", "abstract"))
    block_text = "\n".join(part for part in [title, text] if part)
    page_id = f"{document_id}_page_1"
    return {
        "document_id": document_id,
        "source_id": raw_doc_id,
        "title": title,
        "metadata": {
            "benchmark": "beir_scifact",
            "source_doc_id": raw_doc_id,
        },
        "text_blocks": [
            {
                "id": f"{document_id}_text",
                "document_id": document_id,
                "page_id": page_id,
                "page_number": 1,
                "text": block_text,
                "block_type": "paragraph",
                "metadata": {
                    "benchmark": "beir_scifact",
                    "source_doc_id": raw_doc_id,
                    "title": title,
                },
            }
        ],
    }


def ragbench_case_and_documents(
    *,
    row: dict[str, Any],
    row_index: int,
    subset: str,
    split: str,
) -> dict[str, Any] | None:
    row_id = clean_text(first_value(row, "id", "_id")) or str(row_index)
    case_prefix = f"ragbench_{safe_id(subset)}_{safe_id(row_id)}"
    question = clean_text(first_value(row, "question", "query", "prompt"))
    response = clean_text(first_value(row, "response", "answer", "reference_answer"))
    if not question:
        return None

    parsed_docs = parse_ragbench_documents(
        row=row,
        case_prefix=case_prefix,
        subset=subset,
        split=split,
    )
    if not parsed_docs["documents"]:
        return None

    support_keys = ragbench_support_keys(row)
    expected_source_ids = [
        parsed_docs["key_to_block_id"][key]
        for key in support_keys
        if key in parsed_docs["key_to_block_id"]
    ]
    expected_keywords = informative_tokens(response)[:8]
    case = {
        "id": case_prefix,
        "kind": "ask_document",
        "question": question,
        "document_ids": parsed_docs["document_ids"],
        "top_k": 5,
        "expected_route": "ask_document",
        "expected_keywords": expected_keywords,
        "min_keyword_recall": 0.3,
        "expected_source_ids": unique_values(expected_source_ids),
        "grounding_keywords": expected_keywords,
        "require_nonempty_answer": True,
        "require_evidence": True,
        "metadata": {
            "benchmark": "ragbench",
            "subset": subset,
            "split": split,
            "source_row_id": row_id,
            "reference_response": response,
            "support_sentence_keys": support_keys,
            "adherence_score": first_value(row, "adherence_score", "overall_supported"),
            "trulens_groundedness": first_value(row, "trulens_groundedness"),
            "ragas_faithfulness": first_value(row, "ragas_faithfulness"),
            "relevance_score": first_value(row, "relevance_score"),
            "utilization_score": first_value(row, "utilization_score"),
            "completeness_score": first_value(row, "completeness_score"),
        },
    }
    if not case["expected_source_ids"]:
        case["expected_retrieval_keywords"] = informative_tokens(response)[:5]
    return {"case": case, "documents": parsed_docs["documents"]}


def parse_ragbench_documents(
    *,
    row: dict[str, Any],
    case_prefix: str,
    subset: str,
    split: str,
) -> dict[str, Any]:
    sentence_groups = first_value(row, "documents_sentences")
    documents_text = first_value(row, "documents", "contexts", "passages") or []
    documents: list[dict[str, Any]] = []
    document_ids: list[str] = []
    key_to_block_id: dict[str, str] = {}

    if isinstance(sentence_groups, list) and sentence_groups:
        for doc_index, sentence_group in enumerate(sentence_groups):
            document_id = f"{case_prefix}_doc_{doc_index}"
            page_id = f"{document_id}_page_1"
            blocks = []
            for sentence_index, sentence_item in enumerate(as_list(sentence_group)):
                sentence_key, sentence_text = parse_sentence_item(
                    sentence_item,
                    doc_index=doc_index,
                    sentence_index=sentence_index,
                )
                sentence_text = clean_text(sentence_text)
                if not sentence_text:
                    continue
                block_id = f"{document_id}_sent_{safe_id(sentence_key)}"
                key_to_block_id[sentence_key] = block_id
                for alias in sentence_key_aliases(doc_index, sentence_index):
                    key_to_block_id.setdefault(alias, block_id)
                blocks.append(
                    {
                        "id": block_id,
                        "document_id": document_id,
                        "page_id": page_id,
                        "page_number": 1,
                        "text": sentence_text,
                        "block_type": "paragraph",
                        "metadata": {
                            "benchmark": "ragbench",
                            "subset": subset,
                            "split": split,
                            "sentence_key": sentence_key,
                            "sentence_index": sentence_index,
                            "source_doc_index": doc_index,
                        },
                    }
                )
            if blocks:
                documents.append(
                    {
                        "document_id": document_id,
                        "title": f"RAGBench {subset} {case_prefix} context {doc_index}",
                        "metadata": {
                            "benchmark": "ragbench",
                            "subset": subset,
                            "split": split,
                            "source_doc_index": doc_index,
                        },
                        "text_blocks": blocks,
                    }
                )
                document_ids.append(document_id)
        return {
            "documents": documents,
            "document_ids": document_ids,
            "key_to_block_id": key_to_block_id,
        }

    for doc_index, text in enumerate(as_list(documents_text)):
        text = clean_text(text)
        if not text:
            continue
        document_id = f"{case_prefix}_doc_{doc_index}"
        page_id = f"{document_id}_page_1"
        block_id = f"{document_id}_context"
        documents.append(
            {
                "document_id": document_id,
                "title": f"RAGBench {subset} {case_prefix} context {doc_index}",
                "metadata": {
                    "benchmark": "ragbench",
                    "subset": subset,
                    "split": split,
                    "source_doc_index": doc_index,
                },
                "text_blocks": [
                    {
                        "id": block_id,
                        "document_id": document_id,
                        "page_id": page_id,
                        "page_number": 1,
                        "text": text,
                        "block_type": "paragraph",
                        "metadata": {
                            "benchmark": "ragbench",
                            "subset": subset,
                            "split": split,
                            "source_doc_index": doc_index,
                        },
                    }
                ],
            }
        )
        document_ids.append(document_id)
    return {
        "documents": documents,
        "document_ids": document_ids,
        "key_to_block_id": key_to_block_id,
    }


def parse_sentence_item(sentence_item: Any, *, doc_index: int, sentence_index: int) -> tuple[str, str]:
    if isinstance(sentence_item, dict):
        key = clean_text(first_value(sentence_item, "key", "sentence_key", "id"))
        text = clean_text(first_value(sentence_item, "text", "sentence", "content"))
        return key or default_sentence_key(doc_index, sentence_index), text
    if isinstance(sentence_item, (list, tuple)):
        values = [clean_text(value) for value in sentence_item if clean_text(value)]
        if len(values) >= 2 and looks_like_sentence_key(values[0]):
            return values[0], values[1]
        if len(values) >= 2 and looks_like_sentence_key(values[-1]):
            return values[-1], values[0]
        return default_sentence_key(doc_index, sentence_index), " ".join(values)
    return default_sentence_key(doc_index, sentence_index), clean_text(sentence_item)


def ragbench_support_keys(row: dict[str, Any]) -> list[str]:
    keys = [
        clean_text(key)
        for key in as_list(first_value(row, "all_relevant_sentence_keys"))
        if keep_support_key(key)
    ]
    if keys:
        return unique_values(keys)

    keys = [
        clean_text(key)
        for key in as_list(first_value(row, "all_utilized_sentence_keys"))
        if keep_support_key(key)
    ]
    if keys:
        return unique_values(keys)

    support_info = first_value(row, "sentence_support_information") or []
    for item in as_list(support_info):
        if not isinstance(item, dict):
            continue
        for key in as_list(item.get("supporting_sentence_keys")):
            if keep_support_key(key):
                keys.append(clean_text(key))
    return unique_values(keys)


def keep_support_key(value: Any) -> bool:
    key = clean_text(value)
    return bool(key and key.lower() not in SPECIAL_RAGBENCH_SUPPORT_KEYS)


def sentence_key_aliases(doc_index: int, sentence_index: int) -> list[str]:
    letter_key = default_sentence_key(doc_index, sentence_index)
    return [
        letter_key,
        f"{doc_index}_{sentence_index}",
        f"{doc_index}:{sentence_index}",
        f"{doc_index}-{sentence_index}",
        f"doc{doc_index}_sent{sentence_index}",
        str(sentence_index),
    ]


def default_sentence_key(doc_index: int, sentence_index: int) -> str:
    return f"{doc_index}{letter_index(sentence_index)}"


def letter_index(index: int) -> str:
    letters = ""
    current = index
    while True:
        letters = chr(ord("a") + (current % 26)) + letters
        current = current // 26 - 1
        if current < 0:
            return letters


def looks_like_sentence_key(value: str) -> bool:
    return bool(re.fullmatch(r"\d+[a-z]+|\d+[_:-]\d+|doc\d+_sent\d+", value.strip().lower()))


def validate_case(payload: dict[str, Any]) -> dict[str, Any]:
    return EvaluationCase.model_validate(payload).model_dump(mode="json", exclude_none=True)


def write_benchmark(*, target: Path, knowledge_base: dict[str, Any], cases: list[dict[str, Any]]) -> None:
    target.mkdir(parents=True, exist_ok=True)
    (target / "knowledge_base.json").write_text(
        json.dumps(knowledge_base, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (target / "cases.json").write_text(
        json.dumps({"cases": cases}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def first_value(row: dict[str, Any], *keys: str) -> Any:
    lowered = {key.lower(): key for key in row}
    for key in keys:
        actual_key = lowered.get(key.lower())
        if actual_key is not None and row.get(actual_key) is not None:
            return row.get(actual_key)
    return None


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    return " ".join(str(value).strip().split())


def safe_id(value: Any) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", clean_text(value))
    cleaned = cleaned.strip("._-")
    return cleaned[:120] or "unknown"


def unique_values(values: list[Any]) -> list[Any]:
    unique: list[Any] = []
    seen: set[str] = set()
    for value in values:
        key = clean_text(value)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(value)
    return unique


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    main()
