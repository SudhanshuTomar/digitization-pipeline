# Import necessary libraries
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from typing import List, Optional, Dict, Any
import dataiku
import os
import uuid
import shutil
from pydantic import BaseModel
from vector_store_creation import VectorStoreGeneration

# Configure Dataiku connection
host = "DATAIKU_INSTANCE_URL"
apiKey = "DATAIKU_PROJECT/PERSONAL_KEY"
os.environ["DKU_CURRENT_PROJECT_KEY"] = "GENAIPOC" 
dataiku.set_remote_dss(host, apiKey, no_check_certificate=True)

# Dataset and model configuration parameters
input_dataset_name = "data_embedded"
output_dataset_name = "llm_response"
processed_files_dataset = "processed_files_registry"  # Dataset to track processed files
vector_store_type = 'faiss'
EMBEDDING_MODEL = "custom:iliad-plugin-conn-prod:text-embedding-ada-002"
LLM_MODEL = "custom:iliad-plugin-conn-prod:gpt-4o"

# Temp storage for uploaded files
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(title="RAG Pipeline API", 
              description="API for querying and updating RAG knowledge base",
              version="1.0.0")

# Pydantic model for query requests
class QueryRequest(BaseModel):
    query: str

# Pydantic model for file processing status
class ProcessingStatus(BaseModel):
    filename: str
    status: str
    document_id: Optional[str] = None
    error: Optional[str] = None

def get_processed_files_registry():
    """Get the dataset containing the registry of processed files"""
    try:
        return dataiku.Dataset(processed_files_dataset)
    except Exception:
        # If the dataset doesn't exist, create it (implementation depends on your Dataiku setup)
        # This is a placeholder - you'll need to implement dataset creation logic
        raise HTTPException(status_code=500, detail=f"Processed files registry '{processed_files_dataset}' not found")

def is_file_processed(filename: str, file_hash: str) -> bool:
    """Check if a file has already been processed based on filename and hash"""
    registry_ds = get_processed_files_registry()
    
    # Read registry as pandas dataframe
    registry_df = registry_ds.get_dataframe()
    
    # Check if file exists in registry with matching hash
    return ((registry_df['filename'] == filename) & 
            (registry_df['file_hash'] == file_hash)).any()

def calculate_file_hash(file_path: str) -> str:
    """Calculate a hash for the file to detect changes"""
    import hashlib
    
    hash_md5 = hashlib5.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def update_registry(filename: str, file_hash: str, document_id: str):
    """Add a processed file to the registry"""
    registry_ds = get_processed_files_registry()
    
    # Create a new row for the registry
    import pandas as pd
    new_entry = pd.DataFrame({
        'filename': [filename],
        'file_hash': [file_hash],
        'document_id': [document_id],
        'processed_date': [pd.Timestamp.now()]
    })
    
    # Append to the registry dataset
    registry_ds.write_with_schema(new_entry)

def process_file(file_path: str, original_filename: str):
    """
    Process a single file through the RAG pipeline
    
    Args:
        file_path: Path to the file to process
        original_filename: Original name of the file
        
    Returns:
        document_id: ID of the document in the knowledge base
    """
    try:
        # Calculate file hash to identify this version of the file
        file_hash = calculate_file_hash(file_path)
        
        # Check if this file version has already been processed
        if is_file_processed(original_filename, file_hash):
            return None, "File already processed"
        
        # 1. Extract text from the file
        # Here you'd call your text extraction function
        # This is a placeholder - implement your actual text extraction logic
        # extracted_text = extract_text(file_path)
        
        # 2. Chunk the extracted text
        # chunks = chunk_text(extracted_text)
        
        # 3. Create embeddings
        # embeddings = create_embeddings(chunks)
        
        # 4. Update knowledge base
        # document_id = update_knowledge_base(original_filename, chunks, embeddings)
        
        # For demonstration, we'll use a temporary ID
        document_id = str(uuid.uuid4())
        
        # Update registry with processed file
        update_registry(original_filename, file_hash, document_id)
        
        return document_id, None
        
    except Exception as e:
        return None, str(e)

async def process_files_background(file_paths: List[str], original_filenames: List[str], 
                                  result_list: List[ProcessingStatus]):
    """Background task to process multiple files"""
    for file_path, original_filename in zip(file_paths, original_filenames):
        try:
            document_id, error = process_file(file_path, original_filename)
            
            status = ProcessingStatus(
                filename=original_filename,
                status="processed" if document_id else "skipped",
                document_id=document_id,
                error=error
            )
            
        except Exception as e:
            status = ProcessingStatus(
                filename=original_filename,
                status="error",
                error=str(e)
            )
            
        result_list.append(status)
        
        # Clean up the temporary file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass

@app.post("/query/", response_model=Dict[str, Any])
async def query_endpoint(request: QueryRequest):
    """Endpoint to query the RAG system"""
    response = generate(request.query)
    return response

@app.post("/process-documents/", response_model=Dict[str, Any])
async def process_documents(background_tasks: BackgroundTasks, 
                           files: List[UploadFile] = File(...)):
    """
    Endpoint to process new documents and add them to the knowledge base
    
    This endpoint accepts files, saves them temporarily, and processes them
    through the RAG pipeline in the background.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Prepare for background processing
    file_paths = []
    original_filenames = []
    result_list = []
    
    # Save uploaded files temporarily
    for file in files:
        # Generate unique filename
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        # Save the file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        file_paths.append(file_path)
        original_filenames.append(file.filename)
    
    # Start background processing
    background_tasks.add_task(
        process_files_background, 
        file_paths, 
        original_filenames, 
        result_list
    )
    
    # Return immediate response
    return {
        "status": "processing",
        "message": f"Processing {len(files)} files in the background",
        "job_id": str(uuid.uuid4())  # You could use a real job tracking system here
    }

@app.get("/health/", response_model=Dict[str, Any])
async def health_check_endpoint():
    """Health check endpoint"""
    return health_check()

def generate(user_query: str) -> Dict[str, Any]:
    """
    Generate a response using Retrieval-Augmented Generation (RAG)
    
    Args:
        user_query (str): The user's input query
    
    Returns:
        dict: Response containing the generated answer
    """
    try:
        # Create vector store generation instance
        vector_store = VectorStoreGeneration(
            input_dataset_name=input_dataset_name,
            output_dataset_name=output_dataset_name,
            user_query=user_query,
            embedding_model=EMBEDDING_MODEL,
            llm=LLM_MODEL,
            vector_store_type='FAISS',
            top_k=5,
            use_compression=False,
            azure_openai_key="None"
        )
        
        # Process and get response
        response = vector_store.process()
        
        return {
            "status": "success",
            "user_query": user_query,
            "response": response.split(',')[1],
            "input_dataset": input_dataset_name,
            "output_dataset": output_dataset_name
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "user_query": user_query
        }
    
def health_check() -> Dict[str, Any]:
    """
    Health check endpoint returns basic service status and configuration details.
    
    Returns:
        dict: A dictionary containing the service health status and configuration details.
    """
    try:
        # Additional checks can be added here if necessary
        return {
            "status": "healthy",
            "message": "Service is up and running.",
            "input_dataset": input_dataset_name,
            "output_dataset": output_dataset_name,
            "vector_store_type": vector_store_type,
            "embedding_model": EMBEDDING_MODEL,
            "llm_model": LLM_MODEL
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

# For running the FastAPI app directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)