from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from brandfit_core.storage import ObjectStore


class MultilingualOCR:
    def __init__(self) -> None:
        self._engine: Any = None

    def _load(self) -> Any:
        if self._engine is None:
            try:
                import easyocr
            except ImportError as exc:
                raise RuntimeError(
                    "easyocr is not installed; install the project media extra"
                ) from exc
            # The v1 worker recognizes English plus Devanagari used by Hindi and related evidence.
            self._engine = easyocr.Reader(["en", "hi"], gpu=False)
        return self._engine

    def extract(self, store: ObjectStore, object_key: str) -> dict[str, object]:
        with tempfile.TemporaryDirectory(prefix="brandfit-ocr-") as directory:
            path = Path(directory) / "frame.jpg"
            store.get_file(object_key, path)
            raw = self._load().readtext(str(path))
            lines: list[dict[str, object]] = []
            for box, text, confidence in raw or []:
                lines.append({"text": text, "confidence": confidence, "box": box})
            return {"text": " ".join(str(item["text"]) for item in lines), "lines": lines}
