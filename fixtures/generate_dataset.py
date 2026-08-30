from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).parent
OUTPUT = ROOT / "generated"

CREATORS = [
    ("tech", "Technology", "English", "Testing a practical AI tool and phone workflow."),
    ("food", "Food", "Hindi", "आज की आसान खाना recipe step by step।"),
    ("beauty", "Beauty", "Hinglish", "Aaj simple skincare routine share karte hai."),
    ("travel", "Travel", "Marathi", "आजचा प्रवास सुंदर आहे आणि हा travel guide आहे."),
    ("fitness", "Fitness", "Tamil", "இன்று எளிய fitness workout வழிகாட்டி."),
    ("fashion", "Fashion", "English", "Three wearable fashion styling ideas."),
    ("education", "Education", "Hindi", "आज हम आसान शिक्षा tips सीखेंगे।"),
    ("gaming", "Gaming", "Hinglish", "Aaj gaming setup ka honest review hai."),
    ("lifestyle", "Lifestyle", "Marathi", "दैनंदिन lifestyle routine उपयुक्त आहे."),
    ("entertainment", "Entertainment", "Tamil", "இன்று ஒரு சிறிய entertainment கதை."),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_image(path: Path, title: str, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (720, 720), color=color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 40, 680, 680), outline="white", width=8)
    draw.text((80, 320), title, fill="white")
    image.save(path, format="PNG", optimize=False)


def make_video(path: Path, *, color: str, with_audio: bool) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=c={color}:s=720x1280:d=2",
    ]
    if with_audio:
        command += ["-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-shortest"]
    command += ["-pix_fmt", "yuv420p", str(path)]
    subprocess.run(command, check=True)  # noqa: S603


def build() -> None:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    media_root = OUTPUT / "media"
    manifest_root = OUTPUT / "manifests"
    media_root.mkdir(parents=True)
    manifest_root.mkdir(parents=True)
    now = datetime.now(UTC).replace(microsecond=0)
    benchmark_posts: list[dict[str, object]] = []
    benchmark_personas: list[dict[str, object]] = []
    index: list[dict[str, str]] = []
    for creator_index, (topic, label, language, caption_base) in enumerate(CREATORS):
        username = f"fixture_{topic}_creator"
        creator_dir = media_root / username
        creator_dir.mkdir()
        posts: list[dict[str, object]] = []
        for post_index in range(5):
            is_reel = post_index in {2, 4}
            suffix = ".mp4" if is_reel else ".png"
            media_path = creator_dir / f"post-{post_index + 1}{suffix}"
            if is_reel:
                make_video(
                    media_path,
                    color=["blue", "green", "purple", "orange", "red"][post_index],
                    with_audio=post_index == 4,
                )
            else:
                color = (40 + creator_index * 15, 70 + post_index * 20, 120)
                make_image(media_path, f"{label} fixture {post_index + 1}", color)
            platform_post_id = f"fixture-{creator_index + 1:02d}-{post_index + 1:02d}"
            object_key = f"fixtures/{username}/{media_path.name}"
            metrics = (
                None
                if post_index == 1
                else {
                    "likes": 100 + creator_index * 10 + post_index,
                    "comments": 10 + post_index,
                    "views": 1000 + creator_index * 100,
                    "shares": post_index,
                    "saves": post_index * 2,
                    "captured_at": now.isoformat(),
                }
            )
            posts.append(
                {
                    "platform_post_id": platform_post_id,
                    "post_url": f"https://example.invalid/{username}/{platform_post_id}",
                    "media_type": "reel" if is_reel else "image",
                    "caption": f"{caption_base} Example {post_index + 1}.",
                    "published_at": (now - timedelta(days=post_index * 17)).isoformat(),
                    "is_pinned": post_index == 0,
                    "media": [
                        {
                            "object_key": object_key,
                            "sha256": sha256(media_path),
                            "mime_type": "video/mp4" if is_reel else "image/png",
                            "size_bytes": media_path.stat().st_size,
                            "position": 0,
                        }
                    ],
                    "metrics": metrics,
                    "comments": [
                        {
                            "platform_comment_id": f"comment-{platform_post_id}",
                            "text": (
                                "Stored fixture comment; deliberately excluded from persona "
                                "generation."
                            ),
                            "like_count": 1,
                        }
                    ],
                }
            )
            index.append(
                {"local_path": str(media_path.relative_to(OUTPUT)), "object_key": object_key}
            )
            benchmark_posts.append(
                {
                    "platform_post_id": platform_post_id,
                    "creator": username,
                    "expected_primary_topic": f"topic.{topic}",
                    "expected_language": language,
                    "reviewed": True,
                }
            )
        corrupt = creator_dir / "corrupt.mp4"
        corrupt.write_bytes(b"not-a-video")
        corrupt_key = f"fixtures/{username}/corrupt.mp4"
        index.append({"local_path": str(corrupt.relative_to(OUTPUT)), "object_key": corrupt_key})
        posts.append(
            {
                "platform_post_id": f"fixture-{creator_index + 1:02d}-corrupt",
                "media_type": "reel",
                "caption": "This record must be quarantined without blocking valid posts.",
                "published_at": now.isoformat(),
                "media": [
                    {
                        "object_key": corrupt_key,
                        "sha256": sha256(corrupt),
                        "mime_type": "video/mp4",
                        "size_bytes": corrupt.stat().st_size,
                        "position": 0,
                    }
                ],
            }
        )
        manifest = {
            "schema_version": "1.0",
            "source": "instagram-fixture",
            "source_run_id": f"fixture-v1-{creator_index + 1:02d}",
            "creator": {
                "platform_user_id": f"fixture-user-{creator_index + 1:02d}",
                "username": username,
                "profile": {
                    "display_name": f"Fixture {label} Creator",
                    "bio": f"Generated {label.lower()} fixture account.",
                    "followers": 5000 + creator_index * 1000,
                    "following": 250,
                    "post_count": 5,
                    "captured_at": now.isoformat(),
                },
            },
            "posts": posts,
        }
        manifest_path = manifest_root / f"{username}.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        benchmark_personas.append(
            {
                "creator": username,
                "expected_status": "preliminary",
                "expected_primary_topic": f"topic.{topic}",
                "minimum_supported_claims": 1,
                "reviewed": True,
            }
        )
    (OUTPUT / "dataset_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    (OUTPUT / "benchmark_v1.json").write_text(
        json.dumps(
            {
                "license": "CC0-1.0 generated synthetic fixture data",
                "review_protocol": (
                    "Each expected label was checked against its generated source template."
                ),
                "posts": benchmark_posts,
                "personas": benchmark_personas,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    build()
