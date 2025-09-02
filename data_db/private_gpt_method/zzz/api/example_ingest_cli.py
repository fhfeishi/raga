# api/example_ingest_cli.py

from __future__ import annotations

import argparse
from pathlib import Path

from api.ingest_service import IngestService


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path", type=str, help="file or directory to ingest")
    args = p.parse_args()

    svc = IngestService()
    target = Path(args.path)

    if target.is_dir():
        pairs = [(p.name, p) for p in target.rglob('*') if p.is_file()]
        docs = svc.bulk_ingest(pairs)
    else:
        docs = svc.ingest_file(target.name, target)

    for d in docs:
        print(d.doc_id, d.doc_metadata)


if __name__ == "__main__":
    main()
