from __future__ import annotations

import json
import os
import time
from pathlib import Path

import httpx
import pytest
from minio import Minio


@pytest.mark.skipif(
    os.getenv("BRANDFIT_RUN_INTEGRATION") != "1",
    reason="requires the Docker Compose stack",
)
def test_manifest_to_review_required_persona() -> None:
    generated = Path("fixtures/generated")
    index = json.loads((generated / "dataset_index.json").read_text(encoding="utf-8"))
    manifest_path = sorted((generated / "manifests").glob("*.json"))[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Keep the end-to-end test isolated from the benchmark fixture, which may
    # already have passed human approval in a long-lived development stack.
    manifest["source_run_id"] = "integration-e2e-v1"
    manifest["creator"]["platform_user_id"] += "-integration"
    manifest["creator"]["username"] += "_integration"
    for post in manifest["posts"]:
        post["platform_post_id"] += "-integration"
        if post.get("post_url"):
            post["post_url"] += "-integration"
        for comment in post.get("comments", []):
            comment["platform_comment_id"] += "-integration"
    needed = {item["object_key"] for post in manifest["posts"] for item in post["media"]}
    minio = Minio("localhost:9000", access_key="brandfit", secret_key="change-me", secure=False)
    if not minio.bucket_exists("brandfit-evidence"):
        minio.make_bucket("brandfit-evidence")
    for item in index:
        if item["object_key"] in needed:
            minio.fput_object(
                "brandfit-evidence", item["object_key"], str(generated / item["local_path"])
            )
    with httpx.Client(base_url="http://localhost:8000", timeout=180) as client:
        batch = client.post("/v1/ingestion/manifests", json=manifest).json()
        assert batch["accepted_posts"] == 5
        assert len(batch["quarantined_posts"]) == 1
        for _ in range(180):
            job = client.get(f"/v1/jobs/{batch['job_id']}").json()
            if job["status"] in {"review_required", "completed", "failed"}:
                break
            time.sleep(1)
        assert job["status"] == "review_required", job
        creator_id = job["result"]["creator_id"]
        versions = client.get(f"/v1/creators/{creator_id}/persona/versions").json()
        assert versions[0]["status"] == "preliminary"
        assert versions[0]["is_latest"] is False
