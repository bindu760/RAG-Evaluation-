
from embedding_model import get_embedding_model

# Load embedding model
embeddings = get_embedding_model()

# Sample text
text = "Artificial Intelligence is transforming the world."

# Convert text into vector
vector = embeddings.embed_query(text)

# Display results
print("Original Text:", text)
print("Vector Length:", len(vector))
print("First 10 Vector Values:", vector[:10])
