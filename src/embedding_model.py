
from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_model():
    """
    Load a multilingual sentence transformer
    embedding model.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    return embeddings
