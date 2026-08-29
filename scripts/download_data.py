import zipfile
from pathlib import Path

import kaggle

from honest_model.config import load_config


def download_and_extract_data():
    """
    Download the data from Kaggle and extract it to the data directory.
    """
    # Step 1: Authenticate with Kaggle API
    try:
        kaggle.api.authenticate()
    except Exception as e:
        raise RuntimeError(
            """
            Kaggle API authentication failed. Ensure that you have either:
            1. Place the kaggle.json file in the ~/.kaggle/
            2. Set the KAGGLE_USERNAME and KAGGLE_KEY environment variables
            """
        ) from e
    # Step 2: Check the path to keep the raw data
    raw_data_path = load_config().paths.raw_data_dir
    Path(raw_data_path).mkdir(parents=True, exist_ok=True)

    # Step 3: Download the dataset from Kaggle and unzip
    kaggle.api.competition_download_files(
        "home-credit-default-risk",
        path=raw_data_path,
    )

    # Step 4: Extract all files and remove the downloaded zip
    zip_path = Path(raw_data_path) / "home-credit-default-risk.zip"
    if zip_path.is_file():
        print(f"Extracting {zip_path.name} to {raw_data_path}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(raw_data_path)
        zip_path.unlink()
        print("Extraction complete. Archive removed.")


if __name__ == "__main__":
    download_and_extract_data()
