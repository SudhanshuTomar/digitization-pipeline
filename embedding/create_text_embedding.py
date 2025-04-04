# -*- coding: utf-8 -*-
import dataiku
from dataiku import pandasutils as pdu
from dataiku.scenario import Scenario
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from tqdm.auto import tqdm
import openai
import json
import time

class EmbeddingGenerator:
    def __init__(self, embedding_model, azure_openai_endpoint=None, azure_openai_key=None):
        #self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedding_model = embedding_model
        self.azure_openai_endpoint = azure_openai_endpoint
        self.azure_openai_key = azure_openai_key
        self.model = None
        self.client = dataiku.api_client()
        self.EMBEDDING_MODEL_ID = "custom:iliad-plugin-conn-prod:roughly-jazzy-mermaid"
        self._initialize_model()
    
    def _initialize_model(self):
        if self.embedding_model == 'BAAI/bge-base-en':
            model_id = "BAAI/bge-base-en"
#             self.model = SentenceTransformer(model_id).to(self.device)
#             self.model.eval()
        elif self.embedding_model == 'text-embedding-ada-002':
            openai.api_key = self.azure_openai_key
            self.client = openai.AzureOpenAI(
                api_key=self.azure_openai_key,
                api_version="2025-02-01-preview",
                azure_endpoint=self.azure_openai_endpoint,
            )
        elif self.embedding_model == 'roughly-jazzy-mermaid':
            project = self.client.get_default_project()
            emb_model = project.get_llm(self.EMBEDDING_MODEL_ID)
            self.model = emb_model
    
    def compute_huggingface_embeddings(self, texts, batch_size=32):
        embeddings_list = []
        num_batches = len(texts) // batch_size + (1 if len(texts) % batch_size > 0 else 0)
        for i in tqdm(range(num_batches), desc="Generating embeddings"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(texts))
            batch_texts = [text.lower() for text in texts[start_idx:end_idx]]
            batch_embeddings = self.model.encode(
                batch_texts, normalize_embeddings=True, show_progress_bar=False
            )
            embeddings_list.extend(batch_embeddings.tolist())
        return embeddings_list
    
    def compute_openai_embeddings(self, texts):
        embeddings_list = []
        for text in texts:
            try:
                response = self.client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=text.strip()
                )
                extracted_embeddings = response.data[0].embedding
                embeddings_list.append(extracted_embeddings)
                time.sleep(0.5)
            except Exception as e:
                print(f"Error processing text: {text[:50]}... -> {str(e)}")
                embeddings_list.append(None)
        return embeddings_list
    
    def compute_iliad_embeddings(self, texts):
        embeddings_list = []
        for text in texts:
            try:
                emb_query = self.model.new_embeddings()
                emb_query.add_text(text)
                emb_resp = emb_query.execute()
#                 print(emb_resp.get_embeddings())
                extracted_embeddings = emb_resp.get_embeddings()
                embeddings_list.append(extracted_embeddings[0])
            except Exception as e:
                print(f"Error processing text: {text[:50]}... -> {str(e)}")
                embeddings_list.append(None)
        return embeddings_list
    
    def generate_embeddings(self, df, text_column):
        texts = df[text_column].astype(str).tolist()
        if self.embedding_model == 'BAAI/bge-base-en':
            df['embeddings'] = self.compute_huggingface_embeddings(texts)
        elif self.embedding_model == 'text-embedding-ada-002':
            df['embeddings'] = self.compute_openai_embeddings(texts)
        elif self.embedding_model == 'roughly-jazzy-mermaid':
            df['embeddings'] = self.compute_iliad_embeddings(texts)
        return df


class EmbeddingProcessor:
    def __init__(self, dataset_name, output_dataset_name, text_column, embedding_model, azure_openai_endpoint=None, azure_openai_key=None):
        self.dataset_name = dataset_name
        self.output_dataset_name = output_dataset_name
        self.text_column = text_column
        self.generator = EmbeddingGenerator(embedding_model, azure_openai_endpoint, azure_openai_key)
    
    def process(self):
        dataset = dataiku.Dataset(self.dataset_name)
        df = dataset.get_dataframe()
        df = self.generator.generate_embeddings(df, self.text_column)
        print(f"output datadf")
        output_dataset = dataiku.Dataset(self.output_dataset_name)
        output_dataset.write_with_schema(df)
        print(f"Embeddings written to {self.output_dataset_name}")

