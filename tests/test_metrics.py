import unittest
from langchain_core.documents import Document
from raglab.evaluation import retrieval_metrics

class MetricsTest(unittest.TestCase):
    def test_keyword_metrics(self):
        docs = [Document(page_content="apples"), Document(page_content="revenue 391")]
        data = retrieval_metrics(docs, ["apples", "391"])
        self.assertAlmostEqual(data["mrr"], .75)
        self.assertAlmostEqual(data["keyword_coverage"], 1)
