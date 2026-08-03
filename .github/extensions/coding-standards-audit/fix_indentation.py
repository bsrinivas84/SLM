import ast
import io
import json
import re
import sys
import tokenize
from pathlib import Path


def string_lines(source: str) -> set[int]:
    lines: set[int] = set()
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    for token in tokens:
        if token.type == tokenize.STRING and token.end[0] > token.start[0]:
            # Indentation before an opening quote is code; subsequent physical
            # lines belong to the string token and must remain byte-for-byte.
            lines.update(range(token.start[0] + 1, token.end[0] + 1))
    return lines


def normalize_indentation(source: str) -> tuple[str, int]:
    protected_lines = string_lines(source)
    changed_lines = 0
    normalized_lines = []

    for line_number, line in enumerate(source.splitlines(keepends=True), start=1):
        match = re.match(r"[ \t]+", line)
        if (
            match
            and "\t" in match.group()
            and line_number not in protected_lines
        ):
            indentation = match.group().expandtabs(4)
            line = indentation + line[match.end():]
            changed_lines += 1
        normalized_lines.append(line)

    return "".join(normalized_lines), changed_lines


def normalized_ast(source: str, path: Path) -> str:
    tree = ast.parse(source, filename=str(path), type_comments=True)
    return ast.dump(tree, include_attributes=False)


def prepare_file(path: Path) -> dict[str, object] | None:
    raw_source = path.read_bytes()
    encoding, _ = tokenize.detect_encoding(io.BytesIO(raw_source).readline)
    source = raw_source.decode(encoding)
    normalized, changed_lines = normalize_indentation(source)
    if changed_lines == 0:
        return None

    normalized_tree = normalized_ast(normalized, path)
    repaired_invalid_indentation = False
    try:
        source_tree = normalized_ast(source, path)
    except IndentationError:
        repaired_invalid_indentation = True
    else:
        if source_tree != normalized_tree:
            raise ValueError(f"Refusing AST-changing edit in {path}")

    return {
        "path": path,
        "content": normalized.encode(encoding),
        "changedLines": changed_lines,
        "repairedInvalidIndentation": repaired_invalid_indentation,
    }


def main() -> None:
    source_root = Path(sys.argv[1]).resolve()
    check_only = "--check" in sys.argv[2:]
    prepared = []

    for path in sorted(source_root.rglob("*.py")):
        candidate = prepare_file(path)
        if candidate:
            prepared.append(candidate)

    if not check_only:
        # Validate every candidate before writing any file to avoid partial updates.
        for candidate in prepared:
            candidate["path"].write_bytes(candidate["content"])

    changed = [
        {
            "path": str(candidate["path"].relative_to(source_root.parent)),
            "changedLines": candidate["changedLines"],
            "repairedInvalidIndentation": candidate[
                "repairedInvalidIndentation"
            ],
        }
        for candidate in prepared
    ]

    print(
        json.dumps(
            {
                "changedFiles": len(changed),
                "changedLines": sum(item["changedLines"] for item in changed),
                "checkOnly": check_only,
                "repairedFiles": sum(
                    item["repairedInvalidIndentation"] for item in changed
                ),
                "files": changed,
            }
        )
    )


if __name__ == "__main__":
    main()
