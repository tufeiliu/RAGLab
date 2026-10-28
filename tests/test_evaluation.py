import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from langchain_core.documents import Document
from raglab.evaluation import (TestCase, JudgeScores, load_tests, retrieval_metrics,
                               evaluate_dataset, evaluate_result)


class JudgeStub:
    def evaluate(self, question, generated_answer, reference_answer, retrieved_docs):
        return JudgeScores(feedback="Supported", accuracy=5, completeness=5, relevance=5, faithfulness=5)


class RagStub:
    config = SimpleNamespace(top_k=3)
    def ask(self, question):
        return SimpleNamespace(answer="2015", documents=[Document(page_content="Founded in 2015")])


class EvaluationTests(unittest.TestCase):
    def test_metrics_rank(self):
        docs = [Document(page_content="unrelated"), Document(page_content="founder 2015")]
        data = retrieval_metrics(docs, ["founder", "2015"], k=2)
        self.assertEqual(data["mrr"], 0.5)
        self.assertEqual(data["keyword_coverage_pct"], 100)
        self.assertAlmostEqual(data["ndcg"], 1 / 1.5849625007211563)

    def test_truncated_k(self):
        docs = [Document(page_content="nothing"), Document(page_content="keyword")]
        self.assertEqual(retrieval_metrics(docs, ["keyword"], k=1)["keyword_coverage"], 0)

    def test_load_and_dataset(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "tests.jsonl"
            p.write_text(json.dumps({"question":"When founded?", "keywords":["2015"],
                                     "reference_answer":"2015", "category":"fact"}) + "\n")
            tests = load_tests(p)
            report = evaluate_dataset(RagStub(), tests, JudgeStub())
            self.assertEqual(report["summary"]["accuracy"], 5)
            self.assertEqual(report["summary"]["keyword_coverage"], 1)
            self.assertEqual(report["summary"]["failed"], 0)

    def test_no_reference_rejected_for_judge(self):
        with self.assertRaisesRegex(ValueError, "reference_answer"):
            evaluate_result(RagStub().ask("q"), TestCase(question="q"), JudgeStub())


if __name__ == "__main__":
    unittest.main()
