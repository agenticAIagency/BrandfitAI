from __future__ import annotations

import json
import os

import httpx
from generate_dataset import OUTPUT, build
from minio import Minio


def main() -> None:
    if not (OUTPUT / "dataset_index.json").exists():
        build()
    endpoint = os.getenv("BRANDFIT_MINIO_ENDPOINT", "localhost:9000")
    access = os.getenv("BRANDFIT_MINIO_ACCESS_KEY", "brandfit")
    secret = os.getenv("BRANDFIT_MINIO_SECRET_KEY", "change-me")
    bucket = os.getenv("BRANDFIT_MINIO_BUCKET", "brandfit-evidence")
    api = os.getenv("BRANDFIT_API_URL", "http://localhost:8000")
    client = Minio(endpoint, access_key=access, secret_key=secret, secure=False)
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
    index = json.loads((OUTPUT / "dataset_index.json").read_text(encoding="utf-8"))
    for item in index:
        path = OUTPUT / item["local_path"]
        client.fput_object(bucket, item["object_key"], str(path))
    with httpx.Client(base_url=api, timeout=180) as http:
        for path in sorted((OUTPUT / "manifests").glob("*.json")):
            response = http.post(
                "/v1/ingestion/manifests", json=json.loads(path.read_text(encoding="utf-8"))
            )
            response.raise_for_status()
            print(path.name, response.json())


if __name__ == "__main__":
    main()
