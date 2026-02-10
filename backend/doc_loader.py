from langchain_docling.loader import DoclingLoader
from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    TextLoader,
)
from backend.ingestion_logger import log, info, success, warn, error, timed_step
loaders = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".xlsx": UnstructuredExcelLoader,
}
supported_formats = [".pdf", ".docx", ".txt", ".pptx", ".xlsx"]


def normalize_text(text: str) -> str:
    if not text:
        return ""
    return (
        text.replace("\u00a0", " ") 
            .replace("\r", "\n")
            .replace("\n\n\n", "\n\n")
            .strip()
    )
    

def enrich_docling_metadata(doc):
    dl_meta = doc.metadata.get("dl_meta", {})
    items = dl_meta.get("doc_items", [])

    if items:
        doc.metadata["content_type"] = items[0].get("label")   # table / text / list_item
        prov = items[0].get("prov", [])
        if prov:
            doc.metadata["page_no"] = prov[0].get("page_no")


def load_docs(PATH: str):
    path = Path(PATH)
    if not path.exists():
        raise FileNotFoundError(f"Path {PATH} does not exist.")
    suff = path.suffix.lower()
    if suff not in supported_formats:
        error(f"Unsupported file format: {suff}. Supported: {supported_formats}")
        raise ValueError(f"Unsupported file format: {suff}. Supported formats are: {supported_formats}")

    log("LOAD", f"File: {path.name}  |  Format: {suff}  |  Size: {path.stat().st_size / 1024:.1f} KB")

    with timed_step("DOCLING_LOAD"):
        try:
            loader = DoclingLoader(file_path=str(path))
            documents = loader.load()
            loader_used = "docling"
            info(f"Docling parsed {len(documents)} raw sections")
        except Exception as e:
            warn(f"Docling failed: {e}")
            log("FALLBACK", "Trying standard loader...")
            try:
                if suff == ".txt":
                    loader = TextLoader(
                        str(path),
                        encoding="utf-8",
                        autodetect_encoding=True
                    )
                else:
                    loader = loaders[suff](str(path))

                documents = loader.load()
                loader_used = "fallback"
                info(f"Fallback loader parsed {len(documents)} raw sections")
            except Exception as e:
                error(f"Fallback loader also failed: {e}")
                return []

    skipped = 0
    for doc in documents:
        doc.page_content = normalize_text(doc.page_content)

        if len(doc.page_content) < 30:
            skipped += 1
            continue

        enrich_docling_metadata(doc)

        doc.metadata.update({
                "file_name": path.name,
                "file_type": path.suffix,
                "file_path": str(path),
                "loader": loader_used,
        })

    if skipped:
        warn(f"Skipped {skipped} sections (content < 30 chars)")

    content_types = {}
    for doc in documents:
        ct = doc.metadata.get("content_type", "text")
        content_types[ct] = content_types.get(ct, 0) + 1
    info(f"Content breakdown: {content_types}")

    success(f"Loaded {len(documents)} sections from {path.name} (loader={loader_used})")
    return documents


# test

# if __name__ == "__main__":

#     files = ['data.txt','final-project-proposal.pptx','sample.pdf','sample.docx','sample.xlsx']
    
#     for file in files:
#         print(f"\nTesting loading of {file}...")
#         try:
#             docs = load_docs(f"backend/data/{file}")
#             if docs:
#                 print(f"First document metadata: {docs[0].metadata}")
#                 print(f"First document content preview: {docs[0].page_content[:500]}")
#             else:
#                 print("No documents loaded.")
#         except Exception as e:
#             print(f"Error during testing: {e}") 
        

        
        