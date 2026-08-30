from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from brandfit_core.domain.enums import MediaType
from brandfit_core.storage import ObjectStore


@dataclass(slots=True)
class ExtractedFile:
    kind: str
    path: Path
    locator: str
    mime_type: str


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=True)  # noqa: S603


def _duration(path: Path) -> float:
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ]
    )
    return max(float(json.loads(result.stdout)["format"]["duration"]), 0.1)


def extract_media(
    store: ObjectStore,
    *,
    object_key: str,
    media_type: MediaType,
    mime_type: str,
    output_prefix: str,
    max_frames: int = 12,
) -> list[ExtractedFile]:
    """Download one source object and upload derived scene frames/audio.

    The returned Paths only live until this call finishes, so consumers should use object keys.
    This function returns uploaded key metadata through each item's locator.
    """

    results: list[ExtractedFile] = []
    with tempfile.TemporaryDirectory(prefix="brandfit-media-") as directory:
        root = Path(directory)
        suffix = Path(object_key).suffix or (".jpg" if media_type == MediaType.IMAGE else ".mp4")
        source = root / f"source{suffix}"
        store.get_file(object_key, source)
        if media_type in {MediaType.IMAGE, MediaType.CAROUSEL} or mime_type.startswith("image/"):
            derived_key = f"{output_prefix}/frame-000{suffix}"
            store.put_file(derived_key, source, mime_type)
            results.append(
                ExtractedFile("keyframe", Path(derived_key), "frame:0@0.000s", mime_type)
            )
            return results

        duration = _duration(source)
        frame_pattern = root / "frame-%03d.jpg"
        # FFmpeg scene scores create content-aware frames; fps covers videos with few cuts.
        try:
            _run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(source),
                    "-vf",
                    "select='gt(scene,0.25)',scale='min(1280,iw)':-2",
                    "-vsync",
                    "vfr",
                    "-frames:v",
                    str(max_frames),
                    str(frame_pattern),
                ]
            )
        except subprocess.CalledProcessError:
            pass
        frames = sorted(root.glob("frame-*.jpg"))[:max_frames]
        if not frames:
            interval = max(duration / max_frames, 0.25)
            _run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(source),
                    "-vf",
                    f"fps=1/{interval},scale='min(1280,iw)':-2",
                    "-frames:v",
                    str(max_frames),
                    str(frame_pattern),
                ]
            )
            frames = sorted(root.glob("frame-*.jpg"))[:max_frames]
        for index, frame in enumerate(frames):
            timestamp = min(duration, index * duration / max(len(frames), 1))
            derived_key = f"{output_prefix}/frame-{index:03d}.jpg"
            store.put_file(derived_key, frame, "image/jpeg")
            results.append(
                ExtractedFile(
                    "keyframe", Path(derived_key), f"frame:{index}@{timestamp:.3f}s", "image/jpeg"
                )
            )

        audio = root / "audio.wav"
        try:
            _run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(source),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    str(audio),
                ]
            )
        except subprocess.CalledProcessError:
            audio = Path()
        if audio and audio.is_file() and audio.stat().st_size > 44:
            audio_key = f"{output_prefix}/audio.wav"
            store.put_file(audio_key, audio, "audio/wav")
            results.append(ExtractedFile("audio", Path(audio_key), "audio:0", "audio/wav"))
    return results
