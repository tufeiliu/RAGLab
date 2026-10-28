"""Download FinanceBench annotations and turn source PDFs into a RAGLab-ready corpus.

Never indexes the gold answer, justification, or annotated evidence.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
DATA_URL = 'https://raw.githubusercontent.com/patronus-ai/financebench/main/data/financebench_open_source.jsonl'


def fetch(url: str, destination: Path, *, timeout=60) -> None:
    parsed = urlparse(url)
    if parsed.scheme != 'https' or not parsed.hostname:
        raise ValueError('Only HTTPS URLs are accepted')
    req = Request(url, headers={'User-Agent': 'RAGLab-financebench-research/1.0 (public dataset)'})
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(req, timeout=timeout) as response:
        destination.write_bytes(response.read())


def load_rows(path: Path):
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    for row in rows:
        for key in ('financebench_id', 'doc_name', 'question', 'answer', 'evidence'):
            if key not in row:
                raise ValueError(f'Missing FinanceBench field {key}')
    return rows


def select_rows(rows, max_questions: int, max_documents: int):
    """Choose a deterministic doc-limited subset, keeping all its questions up to cap."""
    chosen = []
    document_names = set()
    for row in sorted(rows, key=lambda r: (str(r['doc_name']), str(r['financebench_id']))):
        doc = str(row['doc_name'])
        if doc not in document_names and len(document_names) >= max_documents:
            continue
        if len(chosen) >= max_questions:
            break
        chosen.append(row)
        document_names.add(doc)
    return chosen


def pdf_to_text(pdf_path: Path) -> str:
    import fitz
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text('text', sort=True)
        pages.append(f'\n\n[PDF PAGE {i + 1}]\n{text}')
    return ''.join(pages).strip()


def prepare(dataset: Path, out_dir: Path, *, max_questions: int, max_documents: int,
            download_pdfs: bool, pause_seconds: float):
    if not dataset.is_file():
        print('Downloading public FinanceBench annotations...')
        fetch(DATA_URL, dataset)
    rows = select_rows(load_rows(dataset), max_questions, max_documents)
    if not rows:
        raise ValueError('No questions selected')
    pdf_dir = out_dir / 'pdfs'
    text_dir = out_dir / 'documents'
    text_dir.mkdir(parents=True, exist_ok=True)
    by_doc = {str(r['doc_name']): r for r in rows}
    ready = set()
    for doc_name, row in sorted(by_doc.items()):
        if not re.fullmatch(r'[A-Za-z0-9_\-.]+', doc_name):
            raise ValueError(f'Unsafe document name {doc_name!r}')
        pdf_path = pdf_dir / f'{doc_name}.pdf'
        if not pdf_path.exists() and download_pdfs:
            urls = [
                f'https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs/{doc_name}.pdf',
                row.get('doc_link') or '',
            ]
            for url in urls:
                try:
                    if url:
                        fetch(url, pdf_path)
                        # Validate downloaded bytes before trusting success.
                        if not pdf_path.read_bytes().startswith(b'%PDF-'):
                            raise ValueError('File is not a PDF')
                        break
                except Exception as exc:
                    pdf_path.unlink(missing_ok=True)
                    print(f'Could not obtain {doc_name} from {url}: {exc}')
            time.sleep(pause_seconds)
        if not pdf_path.exists():
            print(f'MISSING SOURCE: {pdf_path} (place the corresponding FinanceBench PDF here)')
            continue
        txt_path = text_dir / f'{doc_name}.txt'
        try:
            if not txt_path.exists():
                txt_path.write_text(pdf_to_text(pdf_path), encoding='utf-8')
            if len(txt_path.read_text(encoding='utf-8').strip()) < 80:
                print(f'Insufficient extractable text in {pdf_path}; OCR may be needed')
                continue
            ready.add(doc_name)
        except Exception as exc:
            print(f'PDF extraction failed for {pdf_path}: {exc}')
    selected = [r for r in rows if r['doc_name'] in ready]
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / 'questions.jsonl').open('w', encoding='utf-8') as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    # RAGLab's test format; gold evidence is intentionally NOT treated as a keyword list.
    with (out_dir / 'tests.jsonl').open('w', encoding='utf-8') as f:
        for row in selected:
            f.write(json.dumps({'question': row['question'], 'reference_answer': row['answer'],
                'keywords': [], 'category': row.get('question_reasoning') or 'financial_qa'}, ensure_ascii=False) + '\n')
    print(f'Prepared {len(selected)} questions from {len(ready)} filings.')
    if not selected:
        raise RuntimeError('No usable PDF filings. Check downloads or add PDFs to data/pdfs manually.')
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--max-questions', type=int, default=20)
    p.add_argument('--max-documents', type=int, default=4)
    p.add_argument('--dataset', type=Path, default=ROOT / 'data' / 'raw' / 'financebench.jsonl')
    p.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    p.add_argument('--no-download-pdfs', action='store_true')
    p.add_argument('--pause-seconds', type=float, default=1.0)
    args = p.parse_args()
    if args.max_questions < 1 or args.max_documents < 1:
        p.error('Limits must be positive')
    prepare(args.dataset, args.data_dir, max_questions=args.max_questions,
            max_documents=args.max_documents, download_pdfs=not args.no_download_pdfs,
            pause_seconds=args.pause_seconds)


if __name__ == '__main__':
    main()
