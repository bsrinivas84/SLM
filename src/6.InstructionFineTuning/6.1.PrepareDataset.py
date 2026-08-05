"""Download and load the instruction fine-tuning dataset."""

import json
import urllib.request
from pathlib import Path
from typing import Any


DATA_FILE = Path(__file__).resolve().parents[2] / "data" / "raw" / "instruction-data.json"
DATA_URL = (
	"https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/"
	"main/ch07/01_main-chapter-code/instruction-data.json"
)


def download_and_load_file(file_path: str | Path, url: str) -> list[dict[str, Any]]:
	"""Download a JSON file when needed, then return its contents."""
	path = Path(file_path)

	if not path.exists():
		path.parent.mkdir(parents=True, exist_ok=True)
		with urllib.request.urlopen(url, timeout=30) as response:
			text_data = response.read().decode("utf-8")
		path.write_text(text_data, encoding="utf-8")

	with path.open("r", encoding="utf-8") as file:
		return json.load(file)


def format_input(entry: dict[str, Any]) -> str:
	"""Format an instruction dataset entry as a model prompt."""
	instruction_text = (
		"Below is an instruction that describes a task. "
		"Write a response that appropriately completes the request."
		f"\n\n### Instruction:\n{entry['instruction']}"
	)
	input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

	return instruction_text + input_text


def split_data(
	data: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
	"""Split data sequentially into 85% training, 10% testing, and 5% validation."""
	train_portion = int(len(data) * 0.85)
	test_portion = int(len(data) * 0.1)

	train_data = data[:train_portion]
	test_data = data[train_portion : train_portion + test_portion]
	val_data = data[train_portion + test_portion :]

	return train_data, test_data, val_data



    
def main() -> None:
	"""Download the dataset and print its number of entries."""
	data = download_and_load_file(DATA_FILE, DATA_URL)
	train_data, test_data, val_data = split_data(data)
	print("Number of entries:", len(data))
	print("Training entries:", len(train_data))
	print("Testing entries:", len(test_data))
	print("Validation entries:", len(val_data))


if __name__ == "__main__":
	main()
