from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from backend.ingestion_logger import log, info, success, warn, timed_step

TEXT_CHUNK_SIZE = 800
TEXT_CHUNK_OVERLAP = 120

TABLE_CHUNK_SIZE = 400
TABLE_CHUNK_OVERLAP = 50

text_splitter = RecursiveCharacterTextSplitter(chunk_size=TEXT_CHUNK_SIZE, 
                                               chunk_overlap=TEXT_CHUNK_OVERLAP, 
                                               separators=["\n\n", "\n",'.', " ", ""])

table_splitter = RecursiveCharacterTextSplitter(chunk_size=TABLE_CHUNK_SIZE,
                                               chunk_overlap=TABLE_CHUNK_OVERLAP,
                                                  separators=["\n\n", "\n",',', " ", ""])


def split_docs(documents: list[Document]) -> list[Document]:
    log("SPLIT", f"Splitting {len(documents)} sections  |  Text: {TEXT_CHUNK_SIZE}/{TEXT_CHUNK_OVERLAP}  |  Table: {TABLE_CHUNK_SIZE}/{TABLE_CHUNK_OVERLAP}")
    chunked_docs = []
    chunk_id = 0
    empty_skipped = 0
    split_stats = {"text": 0, "table": 0, "slide": 0}

    with timed_step("CHUNKING"):
        for doc in documents:
            content_type = doc.metadata.get("content_type", "text")
            file_type = doc.metadata.get("file_type", "")

            if content_type == "table":
                splits = table_splitter.split_text(doc.page_content)
                splitter_type = "table"
            elif file_type == '.pptx':
                splits = [doc.page_content]
                splitter_type = "slide"
            else:
                splits = text_splitter.split_text(doc.page_content)
                splitter_type = "text"

            for chunk in splits:
                if not chunk.strip():
                    empty_skipped += 1
                    continue

                chunked_docs.append(Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_id": chunk_id,
                        "chunk_type": "slide" if file_type == ".pptx" else content_type
                    }
                ))
                split_stats[splitter_type] = split_stats.get(splitter_type, 0) + 1
                chunk_id += 1

    if empty_skipped:
        warn(f"Skipped {empty_skipped} empty chunks")

    info(f"Chunk breakdown -> text: {split_stats['text']}  |  table: {split_stats['table']}  |  slide: {split_stats['slide']}")

    if chunked_docs:
        lengths = [len(c.page_content) for c in chunked_docs]
        info(f"Chunk lengths -> min: {min(lengths)}  |  max: {max(lengths)}  |  avg: {sum(lengths) // len(lengths)}")

    success(f"Created {len(chunked_docs)} chunks from {len(documents)} sections")
    return chunked_docs




# test

# if __name__ == "__main__":
#     from doc_loader import load_docs
#     files = ['data.txt','final-project-proposal.pptx','sample.pdf','sample.docx','sample.xlsx']

#     for file in files:
#         print(f"\nTesting splitting of {file}...")
#         try:
#             docs = load_docs(f"backend/data/{file}")
#             if docs:
#                 chunked_docs = split_docs(docs)
#                 print(f"Total chunks created: {len(chunked_docs)}")
#                 print(f"First chunk metadata: {chunked_docs[0].metadata}")
#                 print(f"First chunk content preview: {chunked_docs[0].page_content[:500]}")
#             else:
#                 print("No documents loaded to split.")
#         except Exception as e:
#             print(f"Error during splitting: {e}")