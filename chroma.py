import chromadb
from sklearn.metrics.pairwise import cosine_similarity
from embeddings import extract_features
import numpy as np

# Initialize ChromaDB client and collection
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="blockto")


def store_image_embedding(image_path):
    # Extract features
    embedding = extract_features(image_path).tolist()


    collection.add(
        embeddings=[embedding],
        documents=[image_path],  
        ids=[image_path]  
    )

# Define your compare_embedding_with_collection function here
def compare_embedding_with_collection(new_embedding, collection):
    existing_embeddings = collection.peek()["embeddings"]

    # Convert the new_embedding to a numpy array if it's not already
    new_embedding = np.array(new_embedding)

    # List to store similarities
    similarities = []

    # Iterate over existing embeddings and calculate cosine similarity
    for i, existing_embedding in enumerate(existing_embeddings):
        existing_embedding_np = np.array(existing_embedding)
        similarity = cosine_similarity(new_embedding.reshape(1, -1), existing_embedding_np.reshape(1, -1))
        similarities.append((collection.peek()["documents"][i], similarity[0][0]))

    return similarities