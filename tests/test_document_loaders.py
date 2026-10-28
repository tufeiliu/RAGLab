"""Offline tests of parsers without importing cloud SDKs."""
from pathlib import Path
import importlib.util
import sys
import types

class Document:
    def __init__(self, page_content, metadata):
        self.page_content, self.metadata = page_content, metadata

if 'langchain_core.documents' not in sys.modules:
    pkg = types.ModuleType('langchain_core')
    pkg.__path__ = []
    sys.modules['langchain_core'] = pkg
    child = types.ModuleType('langchain_core.documents')
    child.Document = Document
    sys.modules['langchain_core.documents'] = child

path = Path(__file__).resolve().parents[1] / 'raglab' / 'loaders.py'
spec = importlib.util.spec_from_file_location('raglab_loader_under_test', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_csv_column_context(tmp_path):
    p = tmp_path / 'data.csv'
    p.write_text('year,revenue\n2018,1577\n2019,2000\n')
    docs = module.load_file(p, max_rows=1)
    assert len(docs) == 2
    assert 'revenue: 1577' in docs[0].page_content
    assert docs[0].metadata['row_start'] == 2
    assert docs[1].metadata['row_start'] == 3


def test_xlsx_preserves_sheet(tmp_path):
    from openpyxl import Workbook
    p = tmp_path / 'financial.xlsx'
    wb = Workbook()
    ws = wb.active
    ws.title = 'Cash Flows'
    ws.append(['year', 'capex'])
    ws.append([2018, -1577])
    wb.save(p)
    rows = module.load_file(p)
    assert len(rows) == 1
    assert rows[0].metadata['sheet'] == 'Cash Flows'
    assert 'capex: -1577' in rows[0].page_content


def test_pdf_pages(tmp_path):
    import pymupdf
    p = tmp_path / 'sample.pdf'
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((50, 50), 'Capital expenditure 1577')
    pdf.save(p)
    docs = module.load_file(p)
    assert len(docs) == 1
    assert docs[0].metadata['page'] == 1


def test_unknown_extension(tmp_path):
    import pytest
    p = tmp_path / 'evil.exe'
    p.write_text('not a document')
    with pytest.raises(ValueError):
        module.load_file(p)
