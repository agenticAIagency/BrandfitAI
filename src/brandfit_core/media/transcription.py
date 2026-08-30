from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from brandfit_core.storage import ObjectStore


class FasterWhisperTranscriber:
    def __init__(self, model_name: str, *, device: str = "cpu", compute_type: str = "int8") -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError(
                    "faster-whisper is not installed; install the project media extra"
                ) from exc
            self._model = WhisperModel(
                self.model_name, device=self.device, compute_type=self.compute_type
            )
        return self._model

    def transcribe(self, store: ObjectStore, object_key: str) -> dict[str, object]:
        with tempfile.TemporaryDirectory(prefix="brandfit-audio-") as directory:
            path = Path(directory) / "audio.wav"
            store.get_file(object_key, path)
            segments, info = self._load().transcribe(str(path), vad_filter=True)
            values = [
                {
                    "start": round(segment.start, 3),
                    "end": round(segment.end, 3),
                    "text": segment.text.strip(),
                }
                for segment in segments
            ]
            return {
                "language": info.language,
                "language_probability": info.language_probability,
                "segments": values,
                "text": " ".join(item["text"] for item in values),
            }
