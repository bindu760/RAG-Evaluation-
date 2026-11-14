
#PDF Document Loader

from langchain_community.document_loaders import PyPDFLoader


def load_pdf(file_path: str):
    """
    Load a PDF document and extract its text page by page.
    """

    loader = PyPDFLoader(file_path)  #PyPDFLoader this is a class that loads PDF documents and extracts text from them.

    documents = loader.load()

    return documents
