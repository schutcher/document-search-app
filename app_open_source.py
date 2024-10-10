import streamlit as st
import pandas as pd
import numpy as np
import faiss
import os
from split_to_chunks_open_source import clean_text, validate_input
from sentence_transformers import SentenceTransformer

# Initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the DataFrame and FAISS index
df = pd.read_pickle('combined_chunks_with_embeddings.pkl')
index = faiss.read_index('combined_faiss_index.index')

def get_query_embedding(query):
    try:
        # Clean and validate the query
        query = clean_text(query)
        if not validate_input(query):
            st.error("Invalid query input.")
            return None

        embedding = model.encode(query)
        embedding = np.array(embedding, dtype='float32')
        return embedding
    except Exception as e:
        st.error(f"Error generating embedding for the query: {e}")
        return None

def search(query, k=5, selected_documents=None):
    if selected_documents is not None and len(selected_documents) == 0:
        # No documents selected, return empty DataFrame
        return pd.DataFrame()
    
    query_embedding = get_query_embedding(query)
    if query_embedding is None:
        return pd.DataFrame()
    
    query_embedding = query_embedding.reshape(1, -1)
    distances, indices = index.search(query_embedding, k=min(5000, index.ntotal))
    
    # Map indices to original DataFrame indices
    results = df.iloc[indices[0]].copy()
    results['distance'] = distances[0]
    
    # Filter results based on selected documents
    if selected_documents:
        results = results[results['document_name'].isin(selected_documents)]
    
    # Check if any results remain after filtering
    if results.empty:
        return results  # Return empty DataFrame if no results after filtering
    
    # Now select top k results after filtering
    results = results.nsmallest(k, 'distance')
    
    # Convert distances to similarity scores
    results['similarity'] = np.exp(-results['distance'])
    
    return results

# Get unique document names
document_list = df['document_name'].unique().tolist()

# Sidebar for document selection
st.sidebar.title('Document Selection')

doc_search = st.sidebar.text_input('Search documents:')
if doc_search:
    filtered_documents = [doc for doc in document_list if doc_search.lower() in doc.lower()]
else:
    filtered_documents = document_list

# Check if any documents are found after filtering
if filtered_documents:
    selected_documents = st.sidebar.multiselect(
        'Select documents to include in the search:',
        options=filtered_documents,
        default=filtered_documents
    )
else:
    st.sidebar.write("No documents found.")
    selected_documents = []  # Set to empty list to handle in search function

# Add a number input widget for the number of results
k = st.sidebar.slider('Number of results to return:', min_value=1, max_value=50, value=5)

# Main app interface
st.title('Multi-Document Interactive Search with Embeddings')

query = st.text_input('Enter your search query:', placeholder='e.g., "What are the safety regulations for dam construction?"')

if query:
    with st.spinner('Searching...'):
        results = search(query, k=k, selected_documents=selected_documents)
    num_results = len(results)
    if num_results > 0:
        st.write(f'Top {num_results} result{"s" if num_results != 1 else ""}:')
        for idx, row in results.iterrows():
            st.markdown(f"**Document:** {row['document_name']}")
            st.markdown(f"**Page Number:** {row['page_number']}")
            st.markdown(f"**Similarity Score:** {row['similarity']:.4f}")
            if row['url']:
                st.markdown(f"**Web Address:** [{row['url']}]({row['url']})")
            else:
                st.markdown("**Web Address:** Not available")
            st.write(row['chunk'])
            st.write('---')
    else:
        st.write("No results found. Try adjusting your query or document selection.")
