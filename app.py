import streamlit as st

import numpy as np
import chroma as chroma
from chroma import compare_embedding_with_collection,store_image_embedding
from embeddings import extract_features

from chroma import collection


# Streamlit interface
st.title('Image Embedding and Comparison')

uploaded_file = st.file_uploader("Choose an image...", type="jpeg")

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    with open("temp_image.jpeg", "wb") as f:
        f.write(bytes_data)

    # Store and compare image embedding
    store_image_embedding("temp_image.jpeg")
    new_embedding = collection.peek()["embeddings"][-1]  # Get the new embedding
    similarities = compare_embedding_with_collection(new_embedding, collection)
    st.write("Similarities with existing images in the collection:")
    st.write(similarities)
