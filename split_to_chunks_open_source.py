import os
import fitz  # PyMuPDF for PDF text extraction
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import csv

nltk.download('punkt', quiet=True)

# Initialize Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_path):
    texts = []
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            texts.append({
                'page_number': page_num + 1,
                'text': page_text
            })
    return texts

# Function to count tokens (words)
def num_tokens_from_string(string):
    return len(word_tokenize(string))

# Function to clean text
def clean_text(text):
    # Remove control characters and non-UTF-8 characters
    text = text.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
    return text

# Function to validate input
def validate_input(text):
    if not isinstance(text, str):
        return False
    if not text.strip():
        return False
    num_tokens = num_tokens_from_string(text)
    if num_tokens > 512:
        return False
    return True

# Function to get embedding
def get_embedding(text):
    text = text.replace("\n", " ")  # Remove newlines for consistent tokenization
    embedding = model.encode(text)
    return embedding

if __name__ == "__main__":

    # Load processed documents metadata
    metadata_file = 'processed_documents.pkl'
    if os.path.exists(metadata_file):
        processed_docs_df = pd.read_pickle(metadata_file)
    else:
        processed_docs_df = pd.DataFrame(columns=['document_name', 'last_modified'])

    # Load Document URLs
    url_mapping = {}
    with open('document_urls.csv', mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        print(f"CSV Headers: {reader.fieldnames}")
        required_keys = {'document_name', 'url'}
        if not required_keys.issubset(reader.fieldnames):
            raise ValueError(f"CSV file is missing required columns: {required_keys - set(reader.fieldnames)}")
        for row in reader:
            url_mapping[row['document_name']] = row['url']

    # Step 1: Identify New or Updated Documents
    documents_dir = 'documents'  # Replace with your actual directory containing PDFs

    # Get all document paths
    document_paths = [os.path.join(documents_dir, filename) for filename in os.listdir(documents_dir) if filename.endswith('.pdf')]

    # List to hold new documents to process
    new_documents = []

    for doc_path in document_paths:
        doc_name = os.path.basename(doc_path)
        last_modified = os.path.getmtime(doc_path)

        # Check if the document is already processed
        if doc_name in processed_docs_df['document_name'].values:
            # Get the recorded last modified time
            recorded_time = processed_docs_df.loc[processed_docs_df['document_name'] == doc_name, 'last_modified'].values[0]
            # If the file has been modified since last processing, mark it for reprocessing
            if last_modified > recorded_time:
                new_documents.append({'path': doc_path, 'name': doc_name, 'last_modified': last_modified})
        else:
            # New document
            new_documents.append({'path': doc_path, 'name': doc_name, 'last_modified': last_modified})

    if not new_documents:
        print("No new documents to process.")
        exit()

    # Load existing data and embeddings
    data_file = 'combined_chunks_with_embeddings.pkl'
    embeddings_file = 'embeddings.npy'
    if os.path.exists(data_file):
        df = pd.read_pickle(data_file)
        embeddings_matrix = np.load(embeddings_file)
    else:
        df = pd.DataFrame()
        embeddings_matrix = np.array([], dtype='float32').reshape(0, 384)  # Adjust dimension as needed

    # Remove entries for updated documents from df and embeddings_matrix
    updated_doc_names = [doc['name'] for doc in new_documents]

    if not df.empty and len(updated_doc_names) > 0:
        # Get indices to keep (not in updated documents)
        indices_to_keep = df.index[~df['document_name'].isin(updated_doc_names)].tolist()
        df = df.loc[indices_to_keep].reset_index(drop=True)
        embeddings_matrix = embeddings_matrix[indices_to_keep]

    # Step 2: Process New Documents
    all_texts = []

    for doc in new_documents:
        doc_path = doc['path']
        document_name = doc['name']
        print(f"Processing {document_name}...")
        pages = extract_text_from_pdf(doc_path)
        for page in pages:
            page_number = page['page_number']
            text = page['text']
            all_texts.append({
                'document_name': document_name,
                'page_number': page_number,
                'text': text,
                'url': url_mapping.get(document_name, '')  # Get URL or empty string if not found
            })

    # Step 3: Split Text into Chunks
    chunks = []
    chunk_id = df['chunk_id'].max() + 1 if not df.empty else 0  # Continue chunk_id from existing data
    max_tokens = 500  # Adjust as needed

    for doc in all_texts:
        document_name = doc['document_name']
        page_number = doc['page_number']
        text = doc['text']
        url = doc['url']

        sentences = sent_tokenize(text)
        current_chunk = ''
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = num_tokens_from_string(sentence)
            if current_tokens + sentence_tokens <= max_tokens:
                current_chunk += ' ' + sentence
                current_tokens += sentence_tokens
            else:
                # Save the chunk with metadata
                chunks.append({
                    'chunk_id': chunk_id,
                    'document_name': document_name,
                    'page_number': page_number,
                    'chunk': current_chunk.strip(),
                    'url': url
                })
                chunk_id += 1
                current_chunk = sentence
                current_tokens = sentence_tokens

        # Add the last chunk
        if current_chunk:
            chunks.append({
                'chunk_id': chunk_id,
                'document_name': document_name,
                'page_number': page_number,
                'chunk': current_chunk.strip(),
                'url': url
            })
            chunk_id += 1

    print(f"Total new chunks: {len(chunks)}")

    # Step 4: Generate Embeddings for New Chunks
    new_embeddings = []
    processed_chunks = []
    skipped_chunks = []

    for idx, item in enumerate(tqdm(chunks)):
        chunk_text = item['chunk']
        chunk_text = clean_text(chunk_text)
        if not validate_input(chunk_text):
            print(f"Invalid input at chunk ID {item['chunk_id']}. Skipping.")
            skipped_chunks.append(item)
            continue
        try:
            embedding = get_embedding(chunk_text)
            new_embeddings.append(embedding)
            processed_chunks.append(item)
        except Exception as e:
            print(f"Error processing chunk ID {item['chunk_id']}: {e}")
            skipped_chunks.append(item)

    print(f"Total new chunks processed: {len(processed_chunks)}")
    print(f"Total new chunks skipped: {len(skipped_chunks)}")

    # Step 5: Update DataFrame and Embeddings
    new_df = pd.DataFrame(processed_chunks)
    df = pd.concat([df, new_df], ignore_index=True)

    # Save updated DataFrame
    df.to_pickle(data_file)

    # Append new embeddings to embeddings_matrix
    new_embeddings_matrix = np.array(new_embeddings, dtype='float32')
    embeddings_matrix = np.vstack([embeddings_matrix, new_embeddings_matrix])

    # Save updated embeddings
    np.save(embeddings_file, embeddings_matrix)

    # Save skipped chunks for analysis
    with open('skipped_chunks.pkl', 'wb') as f:
        pickle.dump(skipped_chunks, f)

    # Step 6: Update FAISS Index
    embedding_dim = embeddings_matrix.shape[1] if embeddings_matrix.shape[0] > 0 else 384  # Adjust dimension

    # Load or initialize FAISS index
    index_file = 'combined_faiss_index.index'
    if os.path.exists(index_file):
        index = faiss.read_index(index_file)
    else:
        index = faiss.IndexFlatL2(embedding_dim)

    # Add new embeddings to the index
    index.add(new_embeddings_matrix)

    # Save updated index
    faiss.write_index(index, index_file)

    # Step 7: Update processed documents metadata
    # Ensure processed_docs_df has the correct columns and data types
    if processed_docs_df.empty:
        processed_docs_df = pd.DataFrame(columns=['document_name', 'last_modified'])

    for doc in new_documents:
        # Remove old entry if it exists
        processed_docs_df = processed_docs_df[processed_docs_df['document_name'] != doc['name']]

        # Create a DataFrame for the new entry
        new_entry = pd.DataFrame([{
            'document_name': doc['name'],
            'last_modified': doc['last_modified']
        }])

        # Exclude empty or all-NA entries before concatenation
        if not new_entry.empty and not new_entry.isna().all().all():
            if not processed_docs_df.empty:
                processed_docs_df = pd.concat([processed_docs_df, new_entry], ignore_index=True)
            else:
                processed_docs_df = new_entry.copy()
        else:
            print(f"Skipping empty or all-NA new_entry for document {doc['name']}")

    # Save updated metadata
    processed_docs_df.to_pickle(metadata_file)

    print("Processing complete.")
