# Libraries
import dataiku
import pandas as pd
import numpy as np
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.vectorstores import FAISS, Chroma
from langchain.docstore.document import Document
from langchain.schema import Document as LangChainDocument
from langchain.embeddings.base import Embeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import pickle
import os
from typing import List
import re
from sklearn.metrics.pairwise import cosine_similarity
from IliadLLMWrapper import LLMWrapper
from IliadEmbeddingWrapper import DSSLLMEmbeddingWrapper
from typing import List



class ModelDefination:
    def __init__(self, embedding_model, llm, vector_store_type="FAISS", azure_openai_key="Key"):
        # Store model parameters
        self.embedding_model_name = embedding_model
        self.llm_model_name = llm
        self.vector_store_type = vector_store_type.upper()  # Normalize to uppercase
        self.embedding_model = None
        self.llm = None
        self.azure_openai_key = azure_openai_key
        self.client = dataiku.api_client()
        self.project = self.client.get_default_project()

        # Define paths for vector stores
        self.faiss_index_path = "./faiss_index"
        self.chromadb_index_path = "./chromadb_index"

        # Initialize models
        self._initialize_embedding_model()
        self._initialize_llm()

        if self.embedding_model is None:
            raise ValueError("Embedding model could not be initialized")
        if self.llm is None:
            raise ValueError("LLM could not be initialized")

    def _initialize_embedding_model(self):
        try:
            if self.embedding_model_name == "text-embedding-ada-002":
                client = dataiku.api_client()
                connection = client.get_connection("text-embedding-ada-002")
                connection_params = connection.get_info()["params"]

                available_deployments = connection_params.get("availableDeployments", [])
                if not available_deployments:
                    raise ValueError("No deployments found for embedding model.")

                self.embedding_deployment_name = available_deployments[0]["name"]
                model_name = available_deployments[0]["underlyingModelName"]
                azure_openai_endpoint = f"https://{connection_params['resourceName']}.openai.azure.com/"

                self.embedding_model = AzureOpenAIEmbeddings(
                    azure_endpoint=azure_openai_endpoint,
                    api_key=connection_params.get("apiKey"),
                    deployment=self.embedding_deployment_name,
                    model=model_name,
                    chunk_size=1000
                )
                print(f"Initialized embedding model: {model_name}")

            elif "custom:iliad-plugin-conn-prod" in self.embedding_model_name:
                emb_model = self.project.get_llm(self.embedding_model_name)
                self.embedding_model = DSSLLMEmbeddingWrapper(emb_model)
                print(f"Initialized custom embedding model: {self.embedding_model_name}")

        except Exception as e:
            print(f"Error initializing embedding model: {str(e)}")
            import traceback
            traceback.print_exc()

    def _initialize_llm(self):
        try:
            if self.llm_model_name == "gpt-35-turbo-16k":
                connection = self.client.get_connection("gpt-35-turbo-16k-2")
                connection_params = connection.get_info()["params"]

                available_deployments_llm = connection_params.get("availableDeployments", [])
                if not available_deployments_llm:
                    raise ValueError("No deployments found for LLM.")

                llm_deployment_name = available_deployments_llm[0]["name"]
                llm_model_name = available_deployments_llm[0]["underlyingModelName"]
                azure_llm_endpoint = f"https://{connection_params['resourceName']}.openai.azure.com/"

                self.llm = AzureChatOpenAI(
                    azure_endpoint=azure_llm_endpoint,
                    api_key=connection_params.get("apiKey"),
                    deployment_name=llm_deployment_name,
                    model_name=llm_model_name,
                    temperature=0.1,
                    api_version="2024-02-01"
                )
                print(f"Initialized LLM: {llm_model_name}")

            elif "custom:iliad-plugin-conn-prod" in self.llm_model_name:
                llm_model = self.project.get_llm(self.llm_model_name)
                self.llm = LLMWrapper(llm_model)
                print(f"Initialized custom LLM: {self.llm_model_name}")

        except Exception as e:
            print(f"Failed to initialize LLM: {str(e)}")
            raise Exception(f"Error initializing LLM: {str(e)}")

    def document_preparation(self, df):
        """
        Prepare documents and create vector store index.
        Supports both FAISS and ChromaDB.
        """
        try:
            if not isinstance(df, pd.DataFrame):
                raise ValueError("The provided 'df' is not a pandas DataFrame")

            documents = [
                LangChainDocument(
                    page_content=row["chunk_text"],
                    metadata={"id": str(index), "metadata": row["metadata"]}
                )
                for index, row in df.iterrows()
            ]

            # Fix here: Check for any case version of "FAISS"
            if self.vector_store_type.upper() == "FAISS":
                store_path = self.faiss_index_path
                os.makedirs(store_path, exist_ok=True)

                if "embeddings" in df.columns:
                    embeddings = np.array([eval(embed) if isinstance(embed, str) else embed for embed in df["embeddings"]])
                    vectorstore = FAISS.from_embeddings(
                        text_embeddings=zip(df["chunk_text"], embeddings),
                        embedding=self.embedding_model,
                        metadatas=[{"id": str(index), "metadata": row["metadata"]} for index, row in df.iterrows()]
                    )
                else:
                    vectorstore = FAISS.from_documents(documents, embedding=self.embedding_model)

                vectorstore.save_local(store_path)
                print("FAISS index saved successfully.")
                return True

            # Fix here: Check for any case version of "CHROMADB"
            elif self.vector_store_type.upper() == "CHROMADB":
                store_path = self.chromadb_index_path
                os.makedirs(store_path, exist_ok=True)

                vectorstore = Chroma.from_documents(documents, embedding=self.embedding_model, persist_directory=store_path)
                vectorstore.persist()
                print("ChromaDB index saved successfully.")
                return True

            else:
                raise ValueError(f"Unsupported vector store type: {self.vector_store_type}")

        except Exception as e:
            print(f"Failed to create index in Vector Store: {str(e)}")
            return False

    def _compute_semantic_similarity(self, query_vector, document_vectors):
        """
        Compute semantic similarity between query vector and document vectors.
        """
        if not document_vectors:
            return []

        query_vector = np.array(query_vector).reshape(1, -1)
        document_vectors = np.array(document_vectors)
        return cosine_similarity(query_vector, document_vectors).flatten()

    def _preprocess_query(self, query):
        """
        Preprocess the query to extract key terms and concepts.
        """
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 'when', 'where', 'how', 'is', 'are', 'was', 'were'}
        query = re.sub(r'[^\w\s]', ' ', query.lower())
        words = query.split()
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        entities = re.findall(r'\b[A-Z][a-zA-Z]+\b', query)
        return {"original": query, "key_terms": key_terms, "entities": entities}

    def _rerank_documents(self, query, documents, embedded_query=None):
        """
        Rerank documents based on semantic and lexical relevance
        """
        if not documents:
            return []
        
        # Get query components
        query_info = self._preprocess_query(query)
        key_terms = query_info['key_terms']
        
        # Get embeddings for reranking if not provided
        if embedded_query is None and hasattr(self.embedding_model, 'embed_query'):
            embedded_query = self.embedding_model.embed_query(query)
        
        # Extract document embeddings
        doc_embeddings = []
        for doc in documents:
            if hasattr(doc, 'metadata') and 'embedding' in doc.metadata:
                doc_embeddings.append(doc.metadata['embedding'])
            else:
                # If no embedding in metadata, create one
                if hasattr(self.embedding_model, 'embed_documents'):
                    doc_embedding = self.embedding_model.embed_documents([doc.page_content])[0]
                    doc_embeddings.append(doc_embedding)
        
        # Compute semantic similarity if we have embeddings
        if embedded_query is not None and doc_embeddings:
            semantic_scores = self._compute_semantic_similarity(embedded_query, doc_embeddings)
        else:
            semantic_scores = [0] * len(documents)
        
        # Compute lexical similarity (term frequency)
        lexical_scores = []
        for doc in documents:
            content = doc.page_content.lower()
            term_matches = sum(1 for term in key_terms if term in content)
            lexical_scores.append(term_matches / max(1, len(key_terms)))
        
        # Combine scores (0.7 semantic, 0.3 lexical)
        combined_scores = [0.7 * sem + 0.3 * lex for sem, lex in zip(semantic_scores, lexical_scores)]
        
        # Create result tuples with score and document
        results = [(score, doc) for score, doc in zip(combined_scores, documents)]
        
        # Sort by score (descending)
        results.sort(reverse=True, key=lambda x: x[0])
        
        # Return only the documents with their scores
        return [(doc, score) for score, doc in results]

    def retrieve_relevant_chunks(self, df, query, top_k=10):
        """
        Retrieve relevant chunks based on query similarity with improved relevance
        """
        try:
            print(f"Retrieving documents for query: {query[:50]}...")

            # Prepare documents and create index if needed
            try:
                index_created = self.document_preparation(df)
                if not index_created:
                    print(f"Warning: Failed to create document index, trying to use existing index...")
            except Exception as prep_error:
                print(f"Error during document preparation: {prep_error}")
                # Continue anyway, in case the index already exists

            # Check if index exists
            if not os.path.exists(self.faiss_index_path):
                print(f"Vector store index not found at {self.faiss_index_path}. Creating new index...")
                try:
                    index_created = self.document_preparation(df)
                    if not index_created:
                        raise FileNotFoundError(f"Failed to create vector store index at {self.faiss_index_path}")
                except Exception as e:
                    print(f"Error creating index: {e}")
                    return []

            # Load the vector store
            try:
                vectorstore = FAISS.load_local(self.faiss_index_path, self.embedding_model, allow_dangerous_deserialization=True)
                print(f"Successfully loaded FAISS vector store")
            except Exception as load_error:
                print(f"Error loading vector store: {load_error}")
                return []

            # Create retriever
            try:
                # We get more than top_k for reranking
                retriever = vectorstore.as_retriever(
                    search_type="similarity", 
                    search_kwargs={"k": top_k * 2}  # Get more for reranking
                )
                print("Successfully created retriever")
            except Exception as retriever_error:
                print(f"Error creating retriever: {retriever_error}")
                return []

            # Make sure the query is embedded 
            embedded_query = None
            try:
                if hasattr(self.embedding_model, 'embed_query'):
                    embedded_query = self.embedding_model.embed_query(query)
                elif hasattr(self.embedding_model, 'encode'):
                    embedded_query = self.embedding_model.encode(query)
            except Exception as embed_error:
                print(f"Warning: Failed to explicitly embed query: {embed_error}")

            # Retrieve relevant documents
            try:
                print("Attempting to retrieve documents...")
                # Try using the newer invoke method
                if embedded_query is not None:
                    print("Using pre-embedded query")
                    # For FAISS specific embedding search
                    if hasattr(vectorstore, 'similarity_search_by_vector'):
                        initial_docs = vectorstore.similarity_search_by_vector(embedded_query, k=top_k * 2)
                    else:
                        initial_docs = retriever.invoke(query)
                else:
                    initial_docs = retriever.invoke(query)

                print(f"Successfully retrieved {len(initial_docs)} documents for reranking")
            except Exception as invoke_error:
                print(f"Failed to use invoke method: {invoke_error}")
                # Fall back to the older method
                try:
                    if embedded_query is not None and hasattr(vectorstore, 'similarity_search_by_vector'):
                        initial_docs = vectorstore.similarity_search_by_vector(embedded_query, k=top_k * 2)
                    else:
                        initial_docs = retriever.get_relevant_documents(query)
                    print(f"Successfully retrieved {len(initial_docs)} documents using fallback method")
                except Exception as retrieve_error:
                    print(f"Failed to retrieve documents: {retrieve_error}")
                    return []

            # Rerank the documents for better relevance
            reranked_docs = self._rerank_documents(query, initial_docs, embedded_query)
            
            # Take top k after reranking
            top_docs = reranked_docs[:top_k]

            # Format the results
            results = []
            for doc, score in top_docs:
                result = {
                    "chunk_text": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score
                }
                results.append(result)

            return results

        except Exception as e:
            print(f"Failed to retrieve data from Vector Store: {str(e)}")
            return []

    def generate_llm_response(self, user_query, df, llm=None, return_context=True):
        """
        Generate LLM response based on retrieved chunks with improved relevance
        """
        print(f"Generating LLM response for query: {user_query[:50]}...")

        # Use the passed LLM if provided, otherwise use the instance's LLM
        llm_to_use = llm if llm is not None else self.llm

        if llm_to_use is None:
            return ("Error: No LLM instance available", []) if return_context else "Error: No LLM instance available"

        try:
            # Extract key terms from the user query for better retrieval
            query_info = self._preprocess_query(user_query)
            key_terms = query_info['key_terms']
            
            print(f"Identified key terms: {key_terms}")
            
            # Create an augmented query with the key terms emphasized
            augmented_query = user_query
            if key_terms:
                augmented_query = f"{user_query} [KEY TERMS: {', '.join(key_terms)}]"
            
            # Retrieve relevant chunks using the augmented query
            relevant_chunks = self.retrieve_relevant_chunks(df, augmented_query, top_k=10)

            if not relevant_chunks:
                return (
                    "No relevant information found in the provided documents.", 
                    []
                ) if return_context else "No relevant information found in the provided documents."

            print("Retrieved Relevant Chunks:", len(relevant_chunks))
            for i, chunk in enumerate(relevant_chunks, 1):
                print(f"\nResult {i}:")
                print(f"Chunk Text: {chunk['chunk_text'][:100]}...")  # Print just the beginning
                print(f"Metadata: {chunk['metadata']}")
                if chunk["score"] is not None:
                    print(f"Relevance Score: {chunk['score']:.4f}")

            # Prepare formatted context for prompt
            formatted_chunks = ""
            for i, chunk in enumerate(relevant_chunks, 1):
                formatted_chunks += f"CHUNK {i}:\n{chunk['chunk_text']}\n"
                formatted_chunks += f"Source: {chunk['metadata'].get('source', 'Unknown')}\n\n"


            # Create the prompt with improved instruction
            prompt = f"""
                You are an AI assistant that provides precise answers from provided document chunks. Follow these steps:

                1. Carefully read all the document chunks provided.
                2. Determine what specific detail the user is asking for in their query.
                3. Find the most relevant information in the chunks that answers this query.
                4. Extract and return a precise answer, citing the specific chunk used (e.g., "According to Chunk X...").
                5. If no relevant information is found, state that the data is insufficient.

                User Query:
                {user_query}

                Document Chunks:
                {formatted_chunks}

                Answer:
                """

            print("Sending prompt to LLM...")
            # Try different methods to call the LLM based on what's available
            try:
                if hasattr(llm_to_use, 'invoke'):
                    print("Using invoke method")
                    response = llm_to_use.invoke(prompt)
                elif hasattr(llm_to_use, 'predict'):
                    print("Using predict method")
                    response = llm_to_use.predict(prompt)
                elif hasattr(llm_to_use, 'generate'):
                    print("Using generate method")
                    response_obj = llm_to_use.generate(prompt=prompt)
                    response = response_obj
                else:
                    print("Trying to call the LLM object directly")
                    response = llm_to_use(prompt)

                print("Successfully received LLM response")

                # If the response is a DSSLLMCompletionResponse, extract its text
                if hasattr(response, "text") and response.text is not None:
                    final_response = response.text
                # If response is a tuple or list, handle unpacking issues:
                elif isinstance(response, (tuple, list)):
                    try:
                        # If more than two values are returned, simply take the first one
                        final_response = response[0]
                    except Exception as e:
                        print(f"Error unpacking response tuple: {str(e)}")
                        final_response = str(response)
                else:
                    final_response = response

                # Prepare context dictionary for return
                context_dict = {
                    "relevant_chunks": relevant_chunks,
                    "formatted_context": formatted_chunks,
                    "query": user_query,
                    "key_terms": key_terms
                }

                # Return based on return_context flag
                if return_context:
                    return final_response, context_dict
                else:
                    return final_response

            except ValueError as ve:
                # Specifically catch ValueError related to unpacking
                print(f"ValueError encountered: {ve}")
                error_msg = f"Error generating response: {ve}"
                return (error_msg, []) if return_context else error_msg

        except Exception as e:
            print(f"Error in generate_llm_response: {str(e)}")
            error_msg = f"Error generating response: {str(e)}"
            return (error_msg, []) if return_context else error_msg