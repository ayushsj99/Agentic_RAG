from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

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
    chunked_docs = []
    chunk_id = 0
    for doc in documents:
        content_type = doc.metadata.get("content_type", "text")
        file_type = doc.metadata.get("file_type", "")
        
        if content_type == "table":
            splits = table_splitter.split_text(doc.page_content)
            
        elif file_type =='.pptx':
            splits = [doc.page_content]
            
        else:
            splits = text_splitter.split_text(doc.page_content)
            
        for chunk in splits:
            if not chunk.strip():
                continue
            
            chunked_docs.append(Document(
                page_content=chunk,
                metadata={
                    **doc.metadata,
                    "chunk_id": chunk_id,
                    "chunk_type": "slide" if file_type == ".pptx" else content_type

                }
            ))
            chunk_id += 1
            
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