import os
import numpy as np
import faiss
import argparse
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import json

# Function to load and parse JSON file
def load_questions_and_data(json_file_path):
    """
    Loads a JSON file and structures its content by pairing each question with its corresponding data.

    Parameters:
    - json_file_path (str): The path to the JSON file.

    Returns:
    - list of tuples: A list where each tuple contains a question and its corresponding data.
    """
    with open(json_file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)

    question_data_pairs = []
    for entry in json_data:
        data_entry = entry.get('paragraph_id')
        questions = entry.get('questions', [])
        for question in questions:
            question_data_pairs.append((question, data_entry))

    return question_data_pairs


def load_faiss_index_and_mapping(faiss_index_path, mapping_csv_path):
    """
    Loads the FAISS index and the mapping DataFrame.

    Parameters:
    - faiss_index_path (str): Path to the saved FAISS index.
    - mapping_csv_path (str): Path to the saved mapping CSV.

    Returns:
    - faiss.Index: The loaded FAISS index.
    - pd.DataFrame: The loaded mapping DataFrame.
    """
    # Load FAISS index
    index = faiss.read_index(faiss_index_path)
    print(f"Loaded FAISS index with {index.ntotal} vectors.")

    # Load mapping DataFrame
    mapping_df = pd.read_csv(mapping_csv_path, encoding='utf-8-sig')
    print(f"Loaded mapping DataFrame with {len(mapping_df)} entries.")
    return index, mapping_df


def search_faiss(query, index, mapping_df, fine_tuned_model,k =5):
    """
    Performs a similarity search for the given query using FAISS.

    Parameters:
    - query (str): The search query string.
    - index (faiss.Index): The FAISS index.
    - mapping_df (pd.DataFrame): The mapping DataFrame.
    - k (int): Number of nearest neighbors to retrieve.

    Returns:
    - List[Dict]: A list of search results with metadata.
    """
    # Generate embedding for the query
    query_embedding = fine_tuned_model.encode(query)

    # Perform the search
    distances, indices = index.search(np.array([query_embedding], dtype='float32'), k)

    results = []
    for rank, idx in enumerate(indices[0], start=1):
        if idx < len(mapping_df):
            result = {
                'rank': rank,
                'answer_chunk': mapping_df.iloc[idx]['text_chunk'],
                'original_index': mapping_df.iloc[idx]['file_name'],
                'distance': distances[0][rank-1],
                'chunk_id': idx
            }
            results.append(result)
        else:
            # Handle out-of-bounds indices
            print(f"Index {idx} is out of bounds for the mapping DataFrame.")

    return results

def main():
    # Argument parsing for flexibility
    parser = argparse.ArgumentParser(description="FAISS Search with OpenAI Embeddings")
    parser.add_argument('--query', type=str, default="Ai là người đã cho đúc những đồng tiền thưởng đầu tiên trong triều Nguyễn?", help='Search query string')
    parser.add_argument('--k', type=int, default=10, help='Number of nearest neighbors to retrieve')
    parser.add_argument('--faiss_index', type=str, default='database_building/chunked_faiss_index.bin', help='Path to FAISS index file')
    parser.add_argument('--mapping_csv', type=str, default='database_building/chunked_text_data.csv', help='Path to mapping CSV file')
    args = parser.parse_args()

    file_path = 'independent_10k_test.json'  # Replace with your JSON file path
    question_data_pairs = load_questions_and_data(file_path)

# Accessing the structured data:
    print(len(question_data_pairs))
    # Load the FAISS index and mapping
    print("Loading FAISS index and mapping DataFrame...")
    index, mapping_df = load_faiss_index_and_mapping(args.faiss_index, args.mapping_csv)
    # fine_tuned_model = SentenceTransformer('fine-tuned-sentence-transformer')
    fine_tuned_model = SentenceTransformer('fine-tuned-sentence-transformer')
    # fine_tuned_model = SentenceTransformer('BAAI/bge-m3')
    print("THIS IS INDEX", index)
    # Perform the search
    num_correct = 0
    cnt = 0
    pair_for_testing = question_data_pairs[:]
    for question, data in tqdm(pair_for_testing):
        cnt += 1
        query = question
        label = data
        k = args.k
        check = False
        search_results = search_faiss(query, index, mapping_df, fine_tuned_model, k=10)
        for result in search_results:
            if result['chunk_id'] == label:
                check = True
                break
        if check:
            num_correct += 1
        
    print("PERCENTAGE from", len(pair_for_testing), num_correct/len(pair_for_testing))
        # # Display the results
        # for res in search_results:
        #     print(f"--- Rank {res['rank']} ---")
        #     print(f"Answer Chunk: {res['answer_chunk']}")
        #     print(f"Original QA Pair Index: {res['original_index']}")
        #     print(f"Distance: {res['distance']:.4f}")
        #     print("-------------------------\n")

if __name__ == "__main__":
    main()
