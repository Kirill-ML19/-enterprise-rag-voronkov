import json
from pathlib import Path
from typing import Any, Union

PathLike = Union[str, Path]


def save_json(data: Any, path: PathLike, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_json(path: PathLike) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def read_text(path: PathLike) -> str:
    with Path(path).open("r", encoding="utf-8") as f:
        return f.read()


def write_text(text: str, path: PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        f.write(text)
