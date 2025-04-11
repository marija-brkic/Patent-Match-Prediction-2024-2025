"""
create embeddings for all patent texts with sentence-transformers, and save the embeddings as a numpy array
"""

import argparse
# import faiss
# from faiss import write_index, read_index
from sentence_transformers import SentenceTransformer, models

import pandas as pd
import numpy as np

import json
import os


def load_json_data(file_path):
    with open(file_path, "r") as file:
        contents = json.load(file)
    return contents


def combine_content(doc, content_type):
    """
    Combine specific content types from a patent document into a single string
    
    Parameters:
    doc (dict): Patent document
    content_type (str): Type of combination - 'TA', 'claims', or 'TAC'
    
    Returns:
    str: Combined text
    """
    combined_text = ""
    
    if content_type in ['TA', 'TAC']:
        # Add title if available
        if 'title' in doc['Content']:
            combined_text += doc['Content']['title'] + " "
        
        # Add abstract if available
        if 'pa01' in doc['Content']:
            combined_text += doc['Content']['pa01'] + " "
    
    if content_type in ['claims', 'TAC']:
        # Add all claims
        for key in doc['Content']:
            if key.startswith('c-en-'):
                combined_text += doc['Content'][key] + " "
    
    return combined_text.strip()


def main():
    parser = argparse.ArgumentParser(description='Create embeddings for all patent texts with sentence-transformers, and save the embeddings as a numpy array')
    parser.add_argument('--model', '-m', type=str, default='sentence-transformers/all-MiniLM-L6-v2', help='The model to use for embeddings')    # all models that support sentence-transformers architecture can be used : https://huggingface.co/models?library=sentence-transformers&sort=downloads
    parser.add_argument('--pooling', '-p', type=str, default='mean', choices=['mean', 'max', 'cls'], help='The pooling strategy to use for embeddings') # see more pooling strategy options here: https://github.com/UKPLab/sentence-transformers/blob/0ab62663b5b1425f7df05aad34636f7eb6e3a07c/sentence_transformers/models/Pooling.py#L9
    parser.add_argument('--input_file', '-i', type=str, help='The input file to create embeddings for', 
                        default='/bigstorage/DATASETS_JSON/Content_JSONs/Citing_2020_Cleaned_Content_12k/Citing_Train_Test/citing_TRAIN.json')
    parser.add_argument('--output_dir', '-o', type=str, default='embeddings_precalculated', help='The output file to save the embeddings to')
    parser.add_argument('--content_types', '-c', type=str, default='TA,claims,TAC', help='Comma-separated list of content types to create embeddings for (TA: Title+Abstract, claims: All Claims, TAC: Title+Abstract+Claims)')
    args = parser.parse_args()


    # Load the input json file using the new function
    data = load_json_data(args.input_file)
    print(f"Loaded {len(data)} documents from {args.input_file}")
    
    # Parse content types to process
    content_types = args.content_types.split(',')
    
    # Load the model
    base_model = models.Transformer(args.model, max_seq_length=512)
    pooling_model = models.Pooling(base_model.get_word_embedding_dimension(), pooling_mode=args.pooling)
    model = SentenceTransformer(modules=[base_model, pooling_model]).to('cuda')
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Process each content type
    for content_type in content_types:
        print(f"Processing {content_type} embeddings...")
        
        # Combine content for each document based on the content type
        combined_contents = []
        app_ids = []
        
        for doc in data:
            combined_text = combine_content(doc, content_type)
            if combined_text:  # Only include non-empty texts
                combined_contents.append(combined_text)
                app_ids.append(doc['Application_Number'] + doc['Application_Category'])
        
        print(f"Created {len(combined_contents)} combined documents for {content_type}")
        
        # Encode the combined contents
        corpus_embeddings = model.encode(combined_contents, show_progress_bar=True, batch_size=128)
        print(f"Encoded {len(corpus_embeddings)} documents for {content_type}")
        
        # Save the embeddings
        output_file = f'{args.output_dir}/embeddings_{args.model.split("/")[-1]}_{args.pooling}_{content_type}.npy'
        np.save(output_file, corpus_embeddings)
        
        # Also save the app_ids for reference
        app_ids_file = f'{args.output_dir}/app_ids_{args.model.split("/")[-1]}_{args.pooling}_{content_type}.json'
        with open(app_ids_file, 'w') as f:
            json.dump(app_ids, f)
        
        print(f"Saved {content_type} embeddings to {output_file}")


if __name__ == '__main__':
    main()








