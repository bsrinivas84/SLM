import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
ZIP_PATH = DATA_DIR / "sms_spam_collection.zip"
DATA_FILE = DATA_DIR / "SMSSpamCollection.tsv"
TRAIN_FILE = DATA_DIR / "train.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"
TEST_FILE = DATA_DIR / "test.csv"
PRIMARY_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
BACKUP_URL = (
    "https://f001.backblazeb2.com/file/LLMs-from-scratch/"
    "sms%2Bspam%2Bcollection.zip"
)


def download_and_extract(url: str) -> None:
    if DATA_FILE.exists():
        print(f"{DATA_FILE} already exists. Skipping download.")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=30) as response, ZIP_PATH.open("wb") as archive:
        shutil.copyfileobj(response, archive)

    with zipfile.ZipFile(ZIP_PATH) as archive:
        with archive.open("SMSSpamCollection") as source, DATA_FILE.open("wb") as target:
            shutil.copyfileobj(source, target)

    print(f"Downloaded dataset to {DATA_FILE}")


def create_balanced_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    spam_rows = dataframe[dataframe["Label"] == "spam"]
    ham_subset = dataframe[dataframe["Label"] == "ham"].sample(
        n=len(spam_rows), random_state=123
    )
    return pd.concat([ham_subset, spam_rows]).reset_index(drop=True)


def random_split(
    dataframe: pd.DataFrame, train_fraction: float, validation_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataframe = dataframe.sample(frac=1, random_state=123).reset_index(drop=True)
    train_end = int(len(dataframe) * train_fraction)
    validation_end = train_end + int(len(dataframe) * validation_fraction)

    train_dataframe = dataframe[:train_end]
    validation_dataframe = dataframe[train_end:validation_end]
    test_dataframe = dataframe[validation_end:]
    return train_dataframe, validation_dataframe, test_dataframe


def main() -> None:
    try:
        download_and_extract(PRIMARY_URL)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        print("Primary URL failed. Trying backup URL...")
        download_and_extract(BACKUP_URL)

    dataframe = pd.read_csv(DATA_FILE, sep="\t", header=None, names=["Label", "Text"])
    balanced_dataframe = create_balanced_dataset(dataframe)
    print(balanced_dataframe["Label"].value_counts())

    train_dataframe, validation_dataframe, test_dataframe = random_split(
        balanced_dataframe, train_fraction=0.7, validation_fraction=0.1
    )
    for split, path in (
        (train_dataframe, TRAIN_FILE),
        (validation_dataframe, VALIDATION_FILE),
        (test_dataframe, TEST_FILE),
    ):
        split.to_csv(path, index=None)
        print(f"Saved {len(split)} rows to {path}")


if __name__ == "__main__":
    main()
