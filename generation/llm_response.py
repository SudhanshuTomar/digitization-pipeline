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
from retriever import ModelDefination
import pickle
import os
from typing import List
import re
from sklearn.metrics.pairwise import cosine_similarity


            
class VectorStoreGeneration:
    def __init__(self, input_dataset_name, output_dataset_name, user_query, embedding_model, llm, vector_store_type, top_k, use_compression, azure_openai_key):
        self.input_dataset_name = input_dataset_name
        self.output_dataset_name = output_dataset_name
        self.user_query = user_query
        
        # Define the embedding model and LLM first
        self.embedding_model_name = embedding_model
        self.llm_model_name = llm
        self.vector_store_type = vector_store_type
        
        # Create the model definition object with the model names
        self.ModelDef = ModelDefination(
            embedding_model=self.embedding_model_name, 
            llm=self.llm_model_name, 
            vector_store_type=vector_store_type,
            azure_openai_key=azure_openai_key,
        )
        
        self.top_k = top_k
        # We don't use compression as per the requirement
        self.use_compression = False
    
    def assess_context_relevance(self, user_query, contexts, llm):
        """
        Assess if the retrieved contexts are relevant to the user query.
        Returns a relevance score and determination if the contexts are sufficient.
        """
        try:
            relevance_prompt = f"""
            You are an AI evaluator assessing the relevance of retrieved contexts to a user query.
            
            User Query: {user_query}
            
            Retrieved Contexts:
            {contexts}
            
            Task:
            1) Rate the relevance of these contexts to the query on a scale of 0-10 
               (0 = completely irrelevant, 10 = perfectly relevant)
            2) Determine if the contexts contain sufficient information to answer the query (Yes/No)
            3) Explain your reasoning in 1-2 sentences
            
            Format your response as:
            Relevance Score: [0-10]
            Sufficient Information: [Yes/No]
            Reasoning: [Your explanation]
            """
            
            # Use the LLM to assess relevance
            if hasattr(llm, 'invoke'):
                response = llm.invoke(relevance_prompt)
            elif hasattr(llm, 'predict'):
                response = llm.predict(relevance_prompt)
            elif hasattr(llm, 'generate'):
                response_obj = llm.generate(prompt=relevance_prompt)
                response = response_obj.text if hasattr(response_obj, "text") else response_obj
            else:
                response = llm(relevance_prompt)
                
            # Handle tuple/list returns
            if isinstance(response, (tuple, list)):
                response = response[0]
                
            # Parse the response to extract the relevance data
            score_match = re.search(r"Relevance Score:\s*(\d+(?:\.\d+)?)", response)
            sufficiency_match = re.search(r"Sufficient Information:\s*(Yes|No)", response, re.IGNORECASE)
            reasoning_match = re.search(r"Reasoning:\s*(.*?)(?:\n|$)", response, re.DOTALL)
            
            relevance_score = float(score_match.group(1)) if score_match else 0
            is_sufficient = sufficiency_match and sufficiency_match.group(1).lower() == "yes"
            reasoning = reasoning_match.group(1).strip() if reasoning_match else "No explanation provided"
            
            return {
                "score": relevance_score,
                "is_sufficient": is_sufficient,
                "reasoning": reasoning
            }
        except Exception as e:
            print(f"Error assessing context relevance: {str(e)}")
            return {"score": 0, "is_sufficient": False, "reasoning": f"Error: {str(e)}"}

    def generate_ground_truth(self, user_query, contexts, llm, relevance_assessment=None):
        """
        Generate ground truth based on the retrieved contexts, accounting for relevance.
        """
        try:
            # Check if we have a relevance assessment
            if relevance_assessment and not relevance_assessment["is_sufficient"]:
                return f"[INSUFFICIENT CONTEXT] The retrieved information is insufficient to generate a reliable ground truth. Relevance score: {relevance_assessment['score']}/10. {relevance_assessment['reasoning']}"
            
            # Create specialized prompt for ground truth extraction
            prompt = f"""
            You are an objective evaluator extracting factual information from provided contexts.
            Based ONLY on the information in the contexts below, provide the most accurate, 
            concise answer to the question. If the information is not present in the contexts,
            state "Information not available in provided contexts."
            
            Question: {user_query}
            
            Contexts:
            {contexts}
            
            Ground Truth Answer (based solely on the above contexts):
            """
            
            # Use the LLM to generate the ground truth
            if hasattr(llm, 'invoke'):
                response = llm.invoke(prompt)
            elif hasattr(llm, 'predict'):
                response = llm.predict(prompt)
            elif hasattr(llm, 'generate'):
                response_obj = llm.generate(prompt=prompt)
                response = response_obj.text if hasattr(response_obj, "text") else response_obj
            else:
                response = llm(prompt)
                
            # Handle tuple/list returns
            if isinstance(response, (tuple, list)):
                response = response[0]
                
            return response
        except Exception as e:
            print(f"Error generating ground truth: {str(e)}")
            return "Failed to generate ground truth"
    
    def process(self):
        try:
            dataset = dataiku.Dataset(self.input_dataset_name)
            df = dataset.get_dataframe()

            # Generate LLM response and get context
            llm_response, context_dict = self.ModelDef.generate_llm_response(
                self.user_query, 
                df, 
                self.ModelDef.llm
            )

            # Extract the formatted context from the context dictionary
            formatted_context = ""
            if isinstance(context_dict, dict) and "formatted_context" in context_dict:
                formatted_context = context_dict["formatted_context"]
            elif isinstance(context_dict, dict) and "relevant_chunks" in context_dict:
                # Create formatted context from relevant chunks if needed
                chunks = context_dict["relevant_chunks"]
                for i, chunk in enumerate(chunks, 1):
                    formatted_context += f"CHUNK {i}:\n{chunk['chunk_text']}\n"
                    if 'metadata' in chunk and isinstance(chunk['metadata'], dict):
                        formatted_context += f"Source: {chunk['metadata'].get('source', 'Unknown')}\n\n"
            
            # Assess if the retrieved contexts are relevant to the query
            relevance_assessment = self.assess_context_relevance(
                self.user_query,
                formatted_context,
                self.ModelDef.llm
            )
            
            # Generate ground truth, considering relevance assessment
            ground_truth = self.generate_ground_truth(
                self.user_query,
                formatted_context,
                self.ModelDef.llm,
                relevance_assessment
            )

            # Construct the final result with additional metadata
            result = {
                "question": self.user_query,
                "ground_truth": ground_truth,
                "answer": llm_response,
                "contexts": formatted_context,
                "context_relevance": relevance_assessment["score"],
                "context_sufficient": relevance_assessment["is_sufficient"],
                "relevance_reasoning": relevance_assessment["reasoning"]
            }
            return result

        except Exception as e:
            print(f"Failed to generate LLM response: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "question": self.user_query,
                "ground_truth": "",
                "answer": f"Error: {str(e)}",
                "contexts": ""
            }