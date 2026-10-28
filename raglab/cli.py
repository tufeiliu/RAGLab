import argparse
import json
from .core import RAGLab, RAGConfig


def main():
    parser = argparse.ArgumentParser(description="Try RAG configurations")
    parser.add_argument("command", choices=["ingest", "ask", "compare"])
    parser.add_argument("args", nargs="*")
    parser.add_argument("--config", help="Path to JSON with RAGConfig overrides")
    parser.add_argument("--reference", help="Reference answer for compare")
    a = parser.parse_args()
    config = RAGConfig(**json.loads(open(a.config, encoding="utf-8").read())) if a.config else RAGConfig()
    lab = RAGLab(config)
    if a.command == "ingest":
        if not a.args:
            parser.error("Provide .md or .txt filenames")
        print(f"Indexed {lab.ingest_files(a.args)} passages")
    elif a.command == "ask":
        print(json.dumps(lab.ask(" ".join(a.args)).to_dict(), indent=2))
    else:
        question = " ".join(a.args)
        variants = {
            "baseline": {}, "rewrite": {"query_rewrite": True},
            "expansion": {"query_expansion": 3},
            "rerank": {"rerank": True},
            "graph_neighbor": {"graph": True},
            "prompt": {"improved_prompt": True},
        }
        print(json.dumps(lab.compare(question, variants, a.reference), indent=2))


if __name__ == "__main__":
    main()
