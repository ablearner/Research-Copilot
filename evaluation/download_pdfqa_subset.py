from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = [
    "FinanceBench",
    "Tat-QA",
    "ClimateFinanceBench",
    "ClimRetrieve",
]
DATASET_ALIASES = {
    "ClimRetrievb": "ClimRetrieve",
    "ClimRetriev": "ClimRetrieve",
}
PDF_PREFIX = "real-pdfQA/01.2_Input_Files_PDF"
CURATED_SMALL_FILES = {
    "FinanceBench": [
        "3M_2015_10K.pdf",
        "3M_2016_10K.pdf",
        "3M_2017_10K.pdf",
        "3M_2018_10K.pdf",
        "3M_2019_10K.pdf",
    ],
    "Tat-QA": [
        "Auto-Trader_2019.pdf",
        "Hansen-Technologies_2019.pdf",
        "Intu-Properties_2019.pdf",
        "Monolithic-Power-Systems-Inc_2019.pdf",
        "Nextdc-Ltd_2019.pdf",
    ],
    "ClimateFinanceBench": [
        "015157_LVMH_RSE_committed_to_positive_impact_2023_GB_SR_MEL_080724 (6)_2024-09-19_10_56.pdf",
        "2017-2018-NTPC-Sustainability-Report-Final_opt.pdf",
        "2019.pdf",
        "2022.pdf",
        "2023-advancing-climate-solutions-progress-report.pdf",
    ],
    "ClimRetrieve": [
        "2022 Microsoft Environmental Sustainability Report.pdf",
        "2022_BXP_ESG_Report.pdf",
        "2023 Lloyds sustainability report.pdf",
        "AEO 2022 ESG Report.pdf",
        "AT&T 2022 Sustainability Summary.pdf",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download selected pdfQA-Benchmark PDF folders.")
    parser.add_argument(
        "--target-dir",
        default=str(ROOT / "evaluation" / "datasets" / "pdfqa_benchmark"),
        help="Project-local directory where the selected PDF folders will be downloaded.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Dataset folder names under real-pdfQA/01.2_Input_Files_PDF.",
    )
    parser.add_argument(
        "--repo-id",
        default="pdfqa/pdfQA-Benchmark",
        help="Hugging Face dataset repository id.",
    )
    parser.add_argument(
        "--limit-per-dataset",
        type=int,
        default=5,
        help="Maximum number of PDFs to download from each dataset folder. Use 0 for all files.",
    )
    parser.add_argument(
        "--use-curated-small",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the built-in 5-file curated list for the default datasets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_dir = Path(args.target_dir).expanduser().resolve()
    datasets = [DATASET_ALIASES.get(name, name) for name in args.datasets]

    target_dir.mkdir(parents=True, exist_ok=True)
    if args.use_curated_small and args.limit_per_dataset == 5 and all(name in CURATED_SMALL_FILES for name in datasets):
        selected_remote_files = {
            dataset: [f"{PDF_PREFIX}/{dataset}/{filename}" for filename in CURATED_SMALL_FILES[dataset]]
            for dataset in datasets
        }
    else:
        repo_files = list_repo_files(args.repo_id, repo_type="dataset")
        selected_remote_files = {}
        for dataset in datasets:
            prefix = f"{PDF_PREFIX}/{dataset}/"
            remote_files = sorted(path for path in repo_files if path.startswith(prefix) and path.lower().endswith(".pdf"))
            if args.limit_per_dataset > 0:
                remote_files = remote_files[: args.limit_per_dataset]
            selected_remote_files[dataset] = remote_files

    for remote_files in selected_remote_files.values():
        for filename in remote_files:
            download_with_retries(
                repo_id=args.repo_id,
                repo_type="dataset",
                filename=filename,
                local_dir=target_dir,
            )

    manifest = {
        "repo_id": args.repo_id,
        "target_dir": str(target_dir.relative_to(ROOT)),
        "limit_per_dataset": args.limit_per_dataset,
        "datasets": {},
    }
    missing: list[str] = []
    for dataset in datasets:
        remote_files = selected_remote_files[dataset]
        local_files = [target_dir / path for path in remote_files]
        missing.extend(str(path.relative_to(target_dir)) for path in local_files if not path.exists())
        manifest["datasets"][dataset] = {
            "selected_count": len(remote_files),
            "local_count": sum(1 for path in local_files if path.exists()),
            "folder": f"{PDF_PREFIX}/{dataset}",
        }

    manifest_path = target_dir / "pdf_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path.relative_to(ROOT)), **manifest}, ensure_ascii=False, indent=2))
    if missing:
        print("Missing downloaded files:", file=sys.stderr)
        for path in missing:
            print(path, file=sys.stderr)
        raise SystemExit(1)


def download_with_retries(*, repo_id: str, repo_type: str, filename: str, local_dir: Path, retries: int = 5) -> None:
    for attempt in range(1, retries + 1):
        try:
            hf_hub_download(
                repo_id=repo_id,
                repo_type=repo_type,
                filename=filename,
                local_dir=local_dir,
            )
            return
        except Exception:
            if attempt == retries:
                raise
            time.sleep(min(2**attempt, 20))


if __name__ == "__main__":
    main()
