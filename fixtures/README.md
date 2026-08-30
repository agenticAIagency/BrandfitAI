# Generated fixture dataset

Run `python fixtures/generate_dataset.py` to create 50 valid multilingual posts across ten
synthetic creators, plus corrupt-media and missing-audio/metrics cases. The generated data is
CC0-1.0. `benchmark_v1.json` records the reviewed expected post and persona labels.

With the Docker stack running, `python fixtures/load_dataset.py` uploads media to MinIO and
submits every manifest through the public ingestion API.
