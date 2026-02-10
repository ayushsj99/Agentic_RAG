import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from backend.ingestion_logger import log, info, success, warn, timed_step

TEXT_CHUNK_SIZE = 800
TEXT_CHUNK_OVERLAP = 120

TABLE_CHUNK_SIZE = 400
TABLE_CHUNK_OVERLAP = 50

EXCEL_ROWS_PER_CHUNK = 25

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=TEXT_CHUNK_SIZE, 
    chunk_overlap=TEXT_CHUNK_OVERLAP, 
    separators=["\n\n", "\n", ". ", " ", ""]
)

table_splitter = RecursiveCharacterTextSplitter(
    chunk_size=TABLE_CHUNK_SIZE,
    chunk_overlap=TABLE_CHUNK_OVERLAP,
    separators=["\n\n", "\n", ",", " ", ""]
)


def split_excel_rows(text: str, rows_per_chunk: int = EXCEL_ROWS_PER_CHUNK) -> list[dict]:
    lines = [l for l in text.strip().split('\n') if l.strip()]
    if len(lines) < 2:
        return [{"text": text, "row_start": 1, "row_end": len(lines)}]
    
    header = lines[0]
    data_rows = lines[1:]
    
    if not data_rows:
        return [{"text": text, "row_start": 1, "row_end": 1}]
    
    chunks = []
    for i in range(0, len(data_rows), rows_per_chunk):
        batch = data_rows[i:i + rows_per_chunk]
        chunk_text = header + '\n' + '\n'.join(batch)
        chunks.append({
            "text": chunk_text,
            "row_start": i + 2,
            "row_end": i + len(batch) + 1,
            "total_rows": len(data_rows)
        })
    
    return chunks


def extract_section(text: str) -> str:
    patterns = [
        r'^#{1,4}\s+(.+)$',
        r'^\*\*([^*]+)\*\*\s*$',
        r'^(\d+\.?\s+[A-Z][^.]{5,60})$',
    ]
    for line in text.split('\n')[:5]:
        line = line.strip()
        for p in patterns:
            m = re.match(p, line, re.MULTILINE)
            if m:
                return m.group(1).strip()[:80]
    return ""


def split_docs(documents: list[Document]) -> list[Document]:
    log("SPLIT", f"Splitting {len(documents)} sections  |  Text: {TEXT_CHUNK_SIZE}/{TEXT_CHUNK_OVERLAP}  |  Table: {TABLE_CHUNK_SIZE}/{TABLE_CHUNK_OVERLAP}  |  Excel: {EXCEL_ROWS_PER_CHUNK} rows/chunk")
    chunked_docs = []
    chunk_index = 0
    empty_skipped = 0
    split_stats = {"text": 0, "table": 0, "slide": 0, "excel": 0}

    with timed_step("CHUNKING"):
        for doc in documents:
            content_type = doc.metadata.get("content_type", "text")
            file_type = doc.metadata.get("file_type", "")
            source = doc.metadata.get("file_name", "")
            section = extract_section(doc.page_content)

            if file_type in (".xlsx", ".csv") or content_type == "table":
                excel_chunks = split_excel_rows(doc.page_content, EXCEL_ROWS_PER_CHUNK)
                for ec in excel_chunks:
                    if not ec["text"].strip():
                        empty_skipped += 1
                        continue
                    chunked_docs.append(Document(
                        page_content=ec["text"],
                        metadata={
                            **doc.metadata,
                            "source": source,
                            "section": section,
                            "chunk_index": chunk_index,
                            "chunk_type": "table",
                            "row_start": ec.get("row_start", 1),
                            "row_end": ec.get("row_end", 1),
                            "total_rows": ec.get("total_rows", 0),
                        }
                    ))
                    split_stats["excel"] += 1
                    chunk_index += 1
            elif content_type == "slide" or file_type == ".pptx":
                if doc.page_content.strip():
                    chunked_docs.append(Document(
                        page_content=doc.page_content,
                        metadata={
                            **doc.metadata,
                            "source": source,
                            "section": section,
                            "chunk_index": chunk_index,
                            "chunk_type": "slide"
                        }
                    ))
                    split_stats["slide"] += 1
                    chunk_index += 1
            else:
                splits = text_splitter.split_text(doc.page_content)
                for chunk in splits:
                    if not chunk.strip():
                        empty_skipped += 1
                        continue
                    chunked_docs.append(Document(
                        page_content=chunk,
                        metadata={
                            **doc.metadata,
                            "source": source,
                            "section": section,
                            "chunk_index": chunk_index,
                            "chunk_type": content_type
                        }
                    ))
                    split_stats["text"] += 1
                    chunk_index += 1

    if empty_skipped:
        warn(f"Skipped {empty_skipped} empty chunks")

    info(f"Chunk breakdown -> text: {split_stats['text']}  |  excel: {split_stats['excel']}  |  slide: {split_stats['slide']}")
    
    sections_found = sum(1 for c in chunked_docs if c.metadata.get("section"))
    info(f"Sections extracted: {sections_found}/{len(chunked_docs)}")

    if chunked_docs:
        lengths = [len(c.page_content) for c in chunked_docs]
        info(f"Chunk lengths -> min: {min(lengths)}  |  max: {max(lengths)}  |  avg: {sum(lengths) // len(lengths)}")

    success(f"Created {len(chunked_docs)} chunks from {len(documents)} sections")
    return chunked_docs




