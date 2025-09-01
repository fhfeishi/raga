# cli.py
import argparse
from pathlib import Path
from .container import build_injector
from .service import IngestService

def main():
    parser = argparse.ArgumentParser(description="Ingest a file into PrivateGPT stores.")
    parser.add_argument("file", type=Path, help="Path to the file to ingest")
    args = parser.parse_args()

    injector = build_injector()
    svc = injector.get(IngestService)
    docs = svc.ingest_file(args.file.name, args.file)
    print(f"Ingested {len(docs)} document node(s).")
    for d in docs[:5]:
        print(f"- {d.doc_id}  (metadata keys: {list((d.doc_metadata or {}).keys())})")

if __name__ == "__main__":
    main()
