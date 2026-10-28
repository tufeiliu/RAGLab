"""Format-aware file ingestion for RAGLab.

Optional parsers are imported only when used. Structured records include their
column names; no LLM is used to read the files.
"""
from __future__ import annotations

from pathlib import Path
import csv
import json
import re
from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".doc", ".xlsx", ".xls", ".csv", ".tsv", ".txt", ".md", ".json", ".jsonl", ".html", ".htm")


def _doc(content, path, **metadata):
    return Document(page_content=str(content), metadata={"source": str(path), "file_type": path.suffix.lower().lstrip("."), **metadata})


def _rows_as_docs(rows, path, *, sheet=None, max_rows=25):
    """Batch rows while repeating column headers to avoid losing table context."""
    rows = iter(rows)
    try:
        header = next(rows)
    except StopIteration:
        return []
    columns = [str(v).strip() if v is not None else "" for v in header]
    out, batch, start = [], [], 2
    context = f"Sheet: {sheet}\n" if sheet is not None else ""
    def flush(end):
        if batch:
            text = context + "Columns: " + " | ".join(columns) + "\n" + "\n".join(batch)
            out.append(_doc(text, path, **({"sheet":str(sheet)} if sheet is not None else {}), row_start=start, row_end=end))
    for n, row in enumerate(rows, 2):
        values = list(row)
        if not any(v is not None and str(v).strip() for v in values):
            continue
        pairs = [f"{col or f'column_{i+1}'}: {value}" for i, (col, value) in enumerate(zip(columns, values)) if value is not None and str(value).strip()]
        if len(values) > len(columns):
            pairs.extend(f"column_{i+1}: {value}" for i, value in enumerate(values[len(columns):], len(columns)) if value is not None and str(value).strip())
        batch.append(f"Row {n}: " + "; ".join(pairs))
        if len(batch) >= max_rows:
            flush(n)
            batch, start = [], n+1
    flush(n if 'n' in locals() else 1)
    return out


def _strip_running_lines(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Drop running headers/footers: lines repeated on at least half the pages of a 4+ page PDF.

    Otherwise every page contributes a near-identical chunk ("Article  https://doi.org/...")
    that crowds the relevant passages out of the search results. Page numbers are
    ignored when comparing, so "Page 3 of 12" style footers are caught too.
    """
    if len(pages) < 4:
        return pages
    def norm(line):
        return re.sub(r"\d+", "#", " ".join(line.split()).casefold())
    counts = {}
    for _, text in pages:
        for key in {norm(line) for line in text.splitlines() if line.strip()}:
            counts[key] = counts.get(key, 0) + 1
    # Only short lines: running headers and footers are short; body text that merely repeats is kept.
    running = {key for key, n in counts.items() if n >= max(3, len(pages) / 2) and len(key) <= 160}
    return [(number, "\n".join(line for line in text.splitlines() if norm(line) not in running))
            for number, text in pages]


def load_file(path: str | Path, *, max_rows: int = 25) -> list[Document]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type {suffix}: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    if suffix in (".txt", ".md"):
        return [_doc(path.read_text(encoding="utf-8-sig"), path)]
    if suffix == ".pdf":
        try:
            import pymupdf
        except ImportError as exc:
            raise ImportError("PDF ingestion requires: pip install 'raglab-experiments[documents]'") from exc
        with pymupdf.open(path) as pdf:
            pages = [(idx + 1, page.get_text(sort=True)) for idx, page in enumerate(pdf)]
        pages = _strip_running_lines(pages)
        return [_doc(text, path, page=number) for number, text in pages if text.strip()]
    if suffix == ".docx":
        from docx import Document as WordDocument
        document = WordDocument(path)
        out = [_doc(p.text, path, paragraph=i+1) for i,p in enumerate(document.paragraphs) if p.text.strip()]
        for i, table in enumerate(document.tables):
            rows = [[cell.text for cell in row.cells] for row in table.rows]
            for part in _rows_as_docs(rows,path,max_rows=max_rows):
                part.metadata["table"] = i+1
                out.append(part)
        return out
    if suffix == ".doc":
        import shutil, subprocess
        if not shutil.which("antiword"):
            raise RuntimeError("Legacy .doc requires the 'antiword' system utility, or convert the file to .docx first")
        result = subprocess.run(["antiword", str(path)], check=True, capture_output=True, text=True, timeout=60)
        return [_doc(result.stdout, path)]
    if suffix in (".csv", ".tsv"):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return _rows_as_docs(csv.reader(handle, delimiter="\t" if suffix == ".tsv" else ","), path, max_rows=max_rows)
    if suffix == ".xlsx":
        from openpyxl import load_workbook
        workbook = load_workbook(path, read_only=True, data_only=True)
        try:
            return [item for sheet in workbook.worksheets for item in _rows_as_docs(sheet.values,path,sheet=sheet.title,max_rows=max_rows)]
        finally:
            workbook.close()
    if suffix == ".xls":
        import xlrd
        book = xlrd.open_workbook(path)
        return [item for sheet in book.sheets() for item in _rows_as_docs((sheet.row_values(i) for i in range(sheet.nrows)),path,sheet=sheet.name,max_rows=max_rows)]
    if suffix in (".json", ".jsonl"):
        def stringify(obj):
            return json.dumps(obj,ensure_ascii=False,default=str)
        if suffix == ".jsonl":
            return [_doc(stringify(json.loads(line)),path,record=i) for i,line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(),1) if line.strip()]
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(data,list):
            return [_doc(stringify(item),path,record=i) for i,item in enumerate(data,1)]
        return [_doc(stringify(data),path)]
    if suffix in (".html", ".htm"):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(path.read_text(encoding="utf-8-sig"),"html.parser")
        for node in soup(["script","style","noscript"]): node.decompose()
        return [_doc(soup.get_text(separator="\n",strip=True),path)]
    raise AssertionError(suffix)
