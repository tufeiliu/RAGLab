"""Run with pytest after installing RAGLab development dependencies."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from financial_retrieval import identify_filing, windows_from_file, bm25_search, financial_search_terms


def test_choose_filing_from_question():
    filings = ['3M_2018_10K', '3M_2022_10K', '3M_2023Q2_10Q']
    assert identify_filing('What is the FY2018 capital expenditure amount for 3M?', filings) == '3M_2018_10K'
    assert identify_filing('Tell me about 3M', filings) is None


def test_table_rows_survive_chunking(tmp_path):
    p = tmp_path / '3M_2018_10K.txt'
    p.write_text('[PDF PAGE 60]\nCash Flows from Investing Activities\n(Millions) 2018 2017 2016\n'
                 'Purchases of property, plant and equipment (PP&E) (1,577) (1,373) (1,420)\n')
    chunks = windows_from_file(p)
    assert any('Purchases of property' in c.page_content and '2018' in c.page_content and '1,577' in c.page_content for c in chunks)
    assert chunks[0].metadata['fiscal_year'] == 2018


def test_synonym_expansion():
    assert 'purchases of property' in financial_search_terms('FY2018 capex for 3M?')


def test_bm25_prioritizes_relevant_row(tmp_path):
    p = tmp_path / '3M_2018_10K.txt'
    p.write_text('[PDF PAGE 60]\n(Millions) 2018 2017 2016\nPurchases of property plant equipment 1,577\n'
                 '[PDF PAGE 61]\nSummary of dividends and shareholders\n')
    docs = windows_from_file(p)
    top = bm25_search(financial_search_terms('2018 capex property equipment'), docs, 2)
    assert top and '1,577' in top[0].page_content
