"""Check that Ollama is available before scoring generated responses."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any
import urllib.request

import psutil
from tqdm import tqdm


RESPONSE_PATH = (
	Path(__file__).resolve().parents[2]
	/ "data"
	/ "raw"
	/ "instruction-data-with-response.json"
)
SCORE_LOG_PATH = (
	Path(__file__).resolve().parents[2]
	/ "data"
	/ "raw"
	/ "instruction-response-scores.json"
)


def import_neighbor(module_name: str, filename: str) -> ModuleType:
	"""Load a module from the instruction fine-tuning chapter."""
	module_path = Path(__file__).with_name(filename)
	spec = importlib.util.spec_from_file_location(module_name, module_path)
	if spec is None or spec.loader is None:
		raise ImportError(f"Could not load module from {module_path}")

	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


prepare_dataset = import_neighbor("prepare_dataset", "6.1.PrepareDataset.py")
format_input = prepare_dataset.format_input


def load_response_data(file_path: Path = RESPONSE_PATH) -> list[dict[str, Any]]:
	"""Load generated test responses from JSON."""
	if not file_path.exists():
		raise FileNotFoundError(
			f"Response data not found: {file_path}. "
			"Run 6.6ExtractingResponses.py first."
		)

	with file_path.open("r", encoding="utf-8") as file:
		data = json.load(file)

	if not isinstance(data, list) or not all(isinstance(entry, dict) for entry in data):
		raise ValueError("Response data must be a JSON list of objects")
	return data


def check_if_running(process_name: str) -> bool:
	"""Return whether a running process name contains the requested name."""
	normalized_name = process_name.casefold()

	for process in psutil.process_iter(["name"]):
		name = process.info["name"]
		if name and normalized_name in name.casefold():
			return True

	return False


def query_model(
	prompt: str,
	model: str = "llama3.2",
	url: str = "http://localhost:11434/api/chat",
) -> str:
	"""Send a prompt to Ollama and return its streamed response text."""
	data = {
		"model": model,
		"messages": [{"role": "user", "content": prompt}],
		"options": {
			"seed": 123,
			"temperature": 0,
			"num_ctx": 2048,
		},
	}
	payload = json.dumps(data).encode("utf-8")
	request = urllib.request.Request(
		url,
		data=payload,
		headers={"Content-Type": "application/json"},
		method="POST",
	)

	response_parts = []
	with urllib.request.urlopen(request, timeout=120) as response:
		for raw_line in response:
			if not raw_line.strip():
				continue
			response_json = json.loads(raw_line.decode("utf-8"))
			if "error" in response_json:
				raise RuntimeError(f"Ollama error: {response_json['error']}")
			content = response_json.get("message", {}).get("content", "")
			if not isinstance(content, str):
				raise ValueError("Ollama response content must be a string")
			response_parts.append(content)

	return "".join(response_parts)


def generate_model_scores(
	json_data: list[dict[str, Any]],
	json_key: str,
	model: str = "llama3.2",
) -> list[int]:
	"""Ask Ollama to score each generated response from 0 to 100."""
	scores = []
	for entry in tqdm(json_data, desc="Scoring entries"):
		if json_key not in entry:
			entry["score_error"] = f"Missing response key {json_key!r}"
			print(f"Missing response key {json_key!r}; skipping entry")
			continue

		prompt = (
			f"Given the input `{format_input(entry)}` and correct output "
			f"`{entry['output']}`, score the model response "
			f"`{entry[json_key]}` on a scale from 0 to 100, where 100 is "
			"the best score. Respond with the integer number only."
		)
		score_text = query_model(prompt, model).strip()
		entry["scoring_model"] = model
		entry["scoring_model_response"] = score_text
		try:
			score = int(score_text)
		except ValueError:
			entry["score_error"] = "Scoring model response is not an integer"
			print(f"Could not convert score: {score_text}")
			continue

		if not 0 <= score <= 100:
			entry["score_error"] = "Score is outside the 0-100 range"
			print(f"Score outside 0-100 range: {score}")
			continue
		entry["score"] = score
		entry.pop("score_error", None)
		scores.append(score)

	return scores


def save_score_log(
	entries: list[dict[str, Any]],
	scores: list[int],
	output_path: Path = SCORE_LOG_PATH,
) -> None:
	"""Save both model responses, per-entry scores, and aggregate results."""
	output_path.parent.mkdir(parents=True, exist_ok=True)
	log_data = {
		"summary": {
			"entries": len(entries),
			"valid_scores": len(scores),
			"average_score": sum(scores) / len(scores) if scores else None,
		},
		"results": entries,
	}
	with output_path.open("w", encoding="utf-8") as file:
		json.dump(log_data, file, indent=4, ensure_ascii=False)


def main() -> None:
	"""Require a running Ollama process before response scoring."""
	ollama_running = check_if_running("ollama")
	if not ollama_running:
		raise RuntimeError("Ollama not running. Launch Ollama before proceeding.")

	print("Ollama running:", ollama_running)
	test_data = load_response_data()
	scores = generate_model_scores(test_data, "model_response")
	save_score_log(test_data, scores)
	print(f"Number of scores: {len(scores)} of {len(test_data)}")
	if scores:
		print(f"Average score: {sum(scores) / len(scores):.2f}")
	else:
		print("Average score: unavailable (no valid scores)")
	print(f"Score log saved to {SCORE_LOG_PATH}")


if __name__ == "__main__":
	main()
