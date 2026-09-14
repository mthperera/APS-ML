import shutil
from pathlib import Path

import kagglehub

DATASET = "fedesoriano/stroke-prediction-dataset"
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
CSV_PATH = DATA_DIR / "healthcare-dataset-stroke-data.csv"


def main():
    if CSV_PATH.exists():
        print(f"Dataset já existe em {CSV_PATH}")
        return

    cache_dir = Path(kagglehub.dataset_download(DATASET))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(cache_dir / CSV_PATH.name, CSV_PATH)
    print(f"Dataset salvo em {CSV_PATH}")


if __name__ == "__main__":
    main()