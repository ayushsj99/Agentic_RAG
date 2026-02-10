from pathlib import Path
from datetime import datetime, timezone
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    TextLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
)
from backend.ingestion_logger import log, info, success, warn, error, timed_step


LOADERS = {
    ".pdf":  PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt":  TextLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".csv":  CSVLoader,
    ".md":   UnstructuredMarkdownLoader,
}

try:
    from langchain_docling.loader import DoclingLoader
    _DOCLING = True
except ImportError:
    _DOCLING = False

SUPPORTED_FORMATS = list(LOADERS.keys())


def _clean(text: str) -> str:
    if not text:
        return ""
    return text.replace("\u00a0", " ").replace("\r", "\n").replace("\n\n\n", "\n\n").strip()


def _content_type(doc: Document, file_type: str) -> str:
    """Infer content type from metadata or file extension."""
    # Docling gives us a label via dl_meta
    dl_items = doc.metadata.get("dl_meta", {}).get("doc_items", [])
    if dl_items:
        return dl_items[0].get("label", "text")
    if file_type in (".xlsx", ".csv"):
        return "table"
    if file_type == ".pptx":
        return "slide"
    return "text"


def _page_no(doc: Document) -> int:
    """Extract page number from any loader's metadata."""
    # Docling
    dl_items = doc.metadata.get("dl_meta", {}).get("doc_items", [])
    if dl_items:
        prov = dl_items[0].get("prov", [])
        if prov:
            return prov[0].get("page_no", -1)
    # PyPDF / Unstructured
    return doc.metadata.get("page", doc.metadata.get("page_number", -1))


def _metadata(doc: Document, path: Path, loader: str) -> dict:
    """Uniform metadata dict — same keys for every format and loader."""
    suff = path.suffix.lower()
    return {
        "file_name":    path.name,
        "file_type":    suff,
        "file_path":    str(path),
        "loader":       loader,
        "content_type": _content_type(doc, suff),
        "page_no":      _page_no(doc),
        "ingested_at":  datetime.now(timezone.utc).isoformat(),
    }


def load_docs(PATH: str) -> list[Document]:
    path = Path(PATH)
    if not path.exists():
        raise FileNotFoundError(f"Path {PATH} does not exist.")

    suff = path.suffix.lower()
    if suff not in SUPPORTED_FORMATS:
        error(f"Unsupported format: {suff}")
        raise ValueError(f"Unsupported format: {suff}. Supported: {SUPPORTED_FORMATS}")

    log("LOAD", f"{path.name}  |  {suff}  |  {path.stat().st_size / 1024:.1f} KB")

    # 1. Try LangChain Community loader
    docs, loader_used = [], ""
    with timed_step("PRIMARY_LOAD"):
        try:
            cls = LOADERS[suff]
            ldr = cls(str(path), encoding="utf-8", autodetect_encoding=True) if suff == ".txt" else cls(str(path))
            docs = ldr.load()
            loader_used = "langchain"
            info(f"Loaded {len(docs)} sections via LangChain")
        except Exception as e:
            warn(f"LangChain loader failed: {e}")

    # 2. Fallback to Docling
    if not docs and _DOCLING:
        with timed_step("DOCLING_FALLBACK"):
            try:
                docs = DoclingLoader(file_path=str(path)).load()
                loader_used = "docling"
                info(f"Loaded {len(docs)} sections via Docling")
            except Exception as e:
                error(f"Docling also failed: {e}")
                return []
    elif not docs:
        error("No loader succeeded.")
        return []

    # 3. Clean text + apply uniform metadata
    cleaned = []
    for doc in docs:
        doc.page_content = _clean(doc.page_content)
        if len(doc.page_content) < 30:
            continue
        doc.metadata = _metadata(doc, path, loader_used)
        cleaned.append(doc)

    # Stats
    types = {}
    for d in cleaned:
        t = d.metadata["content_type"]
        types[t] = types.get(t, 0) + 1
    info(f"Content breakdown: {types}")
    success(f"{len(cleaned)} sections from {path.name} (loader={loader_used})")
    return cleaned