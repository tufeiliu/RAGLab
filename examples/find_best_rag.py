"""Find the best RAG setup for the OrthoFinder paper, then use it.

Run from the project root (needs OPENAI_API_KEY in .env):

    python examples/find_best_rag.py --dry-run          # show the setups that would run, no API calls
    python examples/find_best_rag.py --quick            # 2 setups x 11 questions, about 45 model calls
    python examples/find_best_rag.py                    # 12 setups x 11 questions, about 260 model calls
    python examples/find_best_rag.py --quick --tests one    # a single question, no correct answer
    python examples/find_best_rag.py --quick --tests list   # plain questions mixed with one answer key

Indexes and embeddings are cached in raglab_data/, which the app shares, so a rerun is much cheaper.
"""
import argparse

from raglab import find_best_rag, plan_search

PAPER = "examples/orthofinder/orthofinder_nature_methods_2026.pdf"

TESTS = {
    # 11 questions with correct answers, evidence passages and keywords: ranked by accuracy.
    "file": "examples/orthofinder/tests.jsonl",
    # One question, no correct answer: judged on relevance, faithfulness and context precision only.
    "one": "Which methods was OrthoFinder compared against?",
    # Plain questions plus one with a correct answer, which is also scored for accuracy.
    "list": ["What does OrthoFinder infer?",
             "Which methods was OrthoFinder compared against?",
             {"question": "Who is the second author of this paper?", "reference_answer": "Yi Liu."}],
}

SEARCH = {
    "full": {"chunk_size": [400, 800], "top_k": [3, 6], "techniques": [[], ["rerank"], ["rerank", "opening"]]},
    "quick": {"chunk_size": [800], "top_k": [5], "techniques": [[], ["opening"]]},
}


def show_progress(**event):
    print(f"  {event.get('current_run') or ''}: {event.get('stage')} "
          f"{event.get('question_done') or 0}/{event.get('total_questions') or '?'}".ljust(70), end="\r")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tests", choices=TESTS, default="file", help="which test questions to use (default: file)")
    parser.add_argument("--quick", action="store_true", help="search 2 setups instead of 12")
    parser.add_argument("--dry-run", action="store_true", help="list the setups without running them")
    parser.add_argument("--chat", default="openai_gpt5_mini", help="answer and judge model alias from models.yaml")
    parser.add_argument("--report", default="best_rag_report.json", help="where to save the full report")
    args = parser.parse_args()
    search = SEARCH["quick" if args.quick else "full"]

    if args.dry_run:
        for run in plan_search(search, chat=args.chat)["runs"]:
            print(run["name"])
        return

    result = find_best_rag(PAPER, TESTS[args.tests], search, chat=args.chat, progress=show_progress)

    print(f"\nRanked by {result.metric}:")
    print(result.leaderboard.to_string())
    print("\nBest:", result.best["name"], result.best["settings"], result.best["techniques"])
    print("Full report (every answer and retrieved passage):", result.save(args.report))

    rag = result.best_rag()  # the winning pipeline, ready to use
    question = "Which methods was OrthoFinder compared against?"
    print(f"\n{question}\n{rag.ask(question).answer}")


if __name__ == "__main__":
    main()
