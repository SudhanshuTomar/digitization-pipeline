# -*- coding: utf-8 -*-
import dataiku
import pandas as pd
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings
import json
import ast

from typing import List

class DSSLLMEmbeddingWrapper:
    def __init__(self, dsllm):
        self.dsllm = dsllm

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Compute embeddings for a list of documents.
        
        Args:
            texts: List of text documents.
        
        Returns:
            A list of embeddings, one per document.
        """
        embeddings = []
        for text in texts:
            # Create and run an embedding query for each document
            emb_query = self.dsllm.new_embeddings()
            emb_query.add_text(text)
            emb_resp = emb_query.execute()
            # Assume get_embeddings() returns a list of embeddings,
            # where each embedding is a list of floats.
            # Since we're processing one text at a time, we take the first element.
            embedding = emb_resp.get_embeddings()[0]
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Compute an embedding for a single query text.
        
        Args:
            text: The text to embed.
        
        Returns:
            The embedding vector as a list of floats.
        """
        emb_query = self.dsllm.new_embeddings()
        emb_query.add_text(text)
        emb_resp = emb_query.execute()
        # Return the first (and only) embedding from the result.
        return emb_resp.get_embeddings()[0]


class ChunkingUtilities:
    @staticmethod
    def format_table_content(table_content):
        """
        Converts a nested list (table) into a readable string format.
        """
        if not table_content or table_content == "[]":
            return ""

        formatted_text = "\n\n### Table Data:\n"
        try:
            # Try to parse the table_content as JSON.
            tables = json.loads(table_content)
        except json.JSONDecodeError:
            try:
                # Fallback to literal_eval if JSON parsing fails.
                tables = ast.literal_eval(table_content)
            except Exception as e:
                print(f"Error processing table data: {e}")
                return f"\n\n### Raw Table Data:\n{table_content}"

        for table in tables:
            for row in table:
                if len(row) >= 3:
                    formatted_text += f"**{row[1]}**: {row[2]}\n"
                else:
                    # For rows with fewer than 3 elements, join available elements.
                    formatted_text += " | ".join(row) + "\n"
        return formatted_text.strip()


    @staticmethod
    def create_paragraph(row):
        """
        Combines extracted text and formatted table data into a meaningful paragraph.
        """
        paragraph = ""
        # Extract text (if available)
        if pd.notna(row.get('text_content')) and row['text_content'].strip():
            paragraph += f"### Extracted Text:\n{row['text_content'].strip()}\n\n"
        
        # Extract table data (if available)
        if pd.notna(row.get('table_content')):
            table_text = ChunkingUtilities.format_table_content(row['table_content'])
            if table_text:
                paragraph += f"{table_text}\n\n"
        
        return paragraph.strip()

    @staticmethod
    def get_image_links(file_name, image_folder):
        """
        Retrieves image links for a given file from the managed folder.
        """
        image_links = []
        try:
            folder_files = image_folder.list_paths_in_partition()
            for file_path in folder_files:
                if file_name in file_path:
                    image_links.append(file_path)  # Store relative path
        except Exception as e:
            print(f"Error retrieving image links for file {file_name}: {e}")
        return image_links

    @staticmethod
    def get_image_names(row):
        """
        Retrieves image names for a given file from the 'images_extracted' column.
        """
        if pd.notna(row.get('images_extracted')) and row['images_extracted'].strip():
            return row['images_extracted'].split(", ")  # Convert comma-separated string into a list
        return []


class TextChunker:
    def __init__(self, splitting_method="RecursiveCharacterTextSplitter", chunk_size=1000, chunk_overlap=200):
        self.splitting_method = splitting_method
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize the appropriate text splitter based on the method.
        if self.splitting_method == "SemanticChunker":
            client = dataiku.api_client()
            project = client.get_default_project()
            EMBEDDING_MODEL_ID = "custom:iliad-plugin-conn-prod:roughly-jazzy-mermaid"
            emb_model = project.get_llm(EMBEDDING_MODEL_ID)
            embedding_model = DSSLLMEmbeddingWrapper(emb_model)
            self.text_splitter = SemanticChunker(embedding_model)
            
        elif self.splitting_method == "RecursiveCharacterTextSplitter":
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", ".", "?", "!", ";"]
            )
            
        elif self.splitting_method == "SimpleTextSplitter":
            self.text_splitter = CharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separator="\n"
            )
        else:
            raise ValueError(f"Invalid splitting method: {self.splitting_method}")

    def split_into_chunks(self, text, file_name, image_links):
        """
        Splits the provided text into chunks and attaches metadata including file name,
        image links, chunk ID, and chunk order.
        """
        chunks = self.text_splitter.split_text(text)
        chunked_records = []
        for i, chunk in enumerate(chunks):
            metadata = {
                "file_name": file_name,
                "image_links": ", ".join(image_links),  # Stored as a comma-separated string
                "chunk_id": f"{file_name}_chunk_{i+1}",
                "chunk_order": i + 1
            }
            chunked_records.append({
                "chunk_text": chunk,
                "metadata": json.dumps(metadata)
            })
        return chunked_records


class DatasetChunkProcessor:
    def __init__(self, input_dataset_name, output_dataset_name, image_folder=None,
                 splitting_method="RecursiveCharacterTextSplitter", chunk_size=1000, chunk_overlap=200):
        self.input_dataset = dataiku.Dataset(input_dataset_name)
        self.output_dataset = dataiku.Dataset(output_dataset_name)
        self.image_folder = image_folder
        self.chunker = TextChunker(splitting_method, chunk_size, chunk_overlap)

    def process_dataset(self):
        """
        Processes each row in the input dataset:
          1. Creates a meaningful paragraph by combining text and table data.
          2. Retrieves image names (or links) for the file.
          3. Splits the paragraph into chunks with metadata.
        The resulting chunked records are written to the output dataset.
        """
        df = self.input_dataset.get_dataframe()
        chunked_data = []
        for _, row in df.iterrows():
            file_name = row.get('file_name', '')
            paragraph = ChunkingUtilities.create_paragraph(row)
            if not paragraph:
                print(f"Skipping empty paragraph for file: {file_name}")
                continue

            # Option to get image links from the image folder if desired:
            # image_links = ChunkingUtilities.get_image_links(file_name, self.image_folder) if self.image_folder else []
            # Alternatively, get image names stored in the row.
            image_links = ChunkingUtilities.get_image_names(row)
            
            chunks = self.chunker.split_into_chunks(paragraph, file_name, image_links)
            chunked_data.extend(chunks)
        
        if chunked_data:
            df_chunked = pd.DataFrame(chunked_data)
            self.output_dataset.write_with_schema(df_chunked)
        else:
            print("No valid data to process.")
