from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from typing import List
from langchain.schema import Document


def filter_to_minimal_docs(docs: List[Document])-> List[Document]:
    """
    Given a list of Documnet objects, return a new list of Document objects 
    containing only 'source' in metadata and the original page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append
        (
            Document
            (
                page_content = doc.page_content,
                metadata = ('source', src)
            )
        )
    return minimal_docs

def chunk_data(data: List[Document], chunk_size: int = 1000, chunk_overlap: int = 100)-> List[Document]:
    """
    Splits a list of documents into smaller chunks.
    
    Args:
        data: A list of Document objects to be split.
        chunk_size: The maximum size of each chunk (in characters).
        chunk_overlap: The number of characters to overlap between chunks.
        
    Returns:
        A list of chunked Document objects.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return text_splitter.split_documents(data)