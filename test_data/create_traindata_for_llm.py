import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import argparse
import torch
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import numpy as np
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from openai import OpenAI


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
    parser = argparse.ArgumentParser(description="FAISS Search with Fine-Tuned PhoBERT")
    parser.add_argument('--output_json', type=str, default='llm_data/training_data-3k-5k.json')
    parser.add_argument('--output_txt', type=str, default='llm_data/query_results.txt')  # New output file for text format
    parser.add_argument('--faiss_index', type=str, default='database_building/chunked_faiss_index.bin', help='Path to FAISS index file')
    parser.add_argument('--mapping_csv', type=str, default='database_building/chunked_text_data.csv', help='Path to mapping CSV file')
    args = parser.parse_args()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the fine-tuned PhoBERT model and tokenizer
    print("Loading the fine-tuned BGEM3 model")
    model = SentenceTransformer('fine-tuned-sentence-transformer')
    model.to(device)
    model.eval()

    # Load FAISS index and mapping
    index, mapping_df = load_faiss_index_and_mapping(args.faiss_index, args.mapping_csv)

    # Load the queries from CSV
    question_data_pairs = load_questions_and_data("independent_10k_test.json")[3000:5000]
    queries_df = pd.DataFrame(question_data_pairs, columns=['query', 'data'])
    print(f"Loaded {len(queries_df)} queries from CSV.")

    results_list = []
    query_count = 0
    correct_count = 0 
    with open(args.output_txt, "w", encoding="utf-8") as txt_file:
        for query,data in zip(queries_df['query'],queries_df['data']):
            label = data
            query_count += 1
            print(f"\nProcessing query: '{query}'...\n")
            txt_file.write(f"\nProcessing query: '{query}'...\n")

            # Perform FAISS search
            search_results = search_faiss(query, index, mapping_df, model, k=10)
            reranked_results = search_results

            for result in search_results:
                if result['chunk_id'] == label:
                    correct_count += 1
                    print(f"Found correct chunk ID: {result['chunk_id']}")
                    break
            print("CURRENT PERCENTAGE", correct_count/query_count*100)
            # Add metadata to reranked results
            for res in reranked_results:
                text_chunk = res['answer_chunk']
                for _, data_rec in mapping_df.iterrows():
                    if text_chunk == data_rec['text_chunk']:
                        res['original_text'] = data_rec['file_name']

            # Load file-to-link mapping
            file_path = "database_building/article_links.csv"
            df_links = pd.read_csv(file_path)
            file_to_link = dict(zip(df_links['Filename'], df_links['URL']))

            # Write search results to text file
            txt_file.write("\nSearch Results:\n")
            for id, res in enumerate(search_results):
                txt_file.write(f"{id}. {res['answer_chunk']}\n")

            # Write reranked results to text file
            answer_content = ""
            txt_file.write("\nReranked Results:\n")
            for id, res in enumerate(reranked_results):
                filename = res['original_text']
                original_link = file_to_link.get(filename, "Link not found")

                entry = f"{id}. Content: {res['answer_chunk']}\n   Location: {filename}\n   Link: {original_link}\n\n"
                txt_file.write(entry)
                answer_content += entry

            # Generate response using GPT
            MODEL = "gpt-4o-mini"
            key = os.getenv('OPENAI_API_KEY')
            client = OpenAI(api_key=key)

            completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": f"""You are an assistant that answers questions based solely on the provided documents. 
                     Do not use any external knowledge or information not contained within the provided data. 
                     If the answer is not found in the documents, respond with 'Không có thông tin trong cơ sở dữ liệu.'
                     Also include link and location of text evidence in the answer.
                     Example:
                     câu hỏi: Đại tướng Võ Nguyên Giáp sinh ra ở đâu?
                     nội dung cung cấp: 0. Content Võ Nguyên Giáp sinh ra ở Quảng Bình. Location: abc.txt, Link: xyz
                     câu trả lời: Đại tướng Võ Nguyên Giáp sinh ra ở Quảng Bình. Location: abc.txt, Link: xyz
                     câu hỏi: Vị vua nào được dân gian mô tả với "gương mặt sắt đen sì"?
                     nội dung cung cấp: 
                     0. Mai Hắc Đế là vị vua anh minh. Location abc.txt, Link xyz, 
                     1. Nhà Trần được thành lập năm ... Location abc.txt, Link xyz.
                     câu trả lời: Không tìm được thông tin liên quan """},
                    {"role": "user", "content": f"""câu hỏi: {query}. 
                     nội dung cung cấp: {answer_content}
                     câu trả lời:"""},
                ]
            )
            response_content = completion.choices[0].message.content

            print("============THIS IS THE RESPONSE CONTENT", response_content)

            # Store the result in the required format
            results_list.append({
                "instruction": query,         # The original query/question goes here
                "input": answer_content,      # The retrieved answer content (relevant text chunks)
                "output": response_content    # The final AI-generated response
            })

            # Write GPT response to text file
            txt_file.write("\nGenerated Response:\n")
            txt_file.write(response_content + "\n\n")

    # Save results to JSON
    with open(args.output_json, "w", encoding="utf-8-sig") as json_file:
        json.dump(results_list, json_file, indent=4, ensure_ascii=False)

    print(f"Results saved to {args.output_json} and {args.output_txt}")

if __name__ == "__main__":
    main()
