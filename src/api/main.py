"""FastAPI backend for AggieConnect."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from rag.pipeline import AggieConnectAssistant, RAGPipeline
from config import API_HOST, API_PORT, DEBUG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AggieConnect API",
    description="LLM-powered assistant for UC Davis students",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global assistant instance
assistant = None

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    k: Optional[int] = 5

class QueryResponse(BaseModel):
    question: str
    response: str
    sources: int
    context: Optional[List[Dict[str, Any]]] = None

class HealthResponse(BaseModel):
    status: str
    message: str
    version: str

class StatsResponse(BaseModel):
    total_vectors: int
    dimension: int
    total_documents: int
    type_counts: Dict[str, int]

@app.on_event("startup")
async def startup_event():
    """Initialize the assistant on startup."""
    global assistant
    try:
        logger.info("Initializing AggieConnect Assistant...")
        assistant = AggieConnectAssistant()
        logger.info("Assistant initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize assistant: {str(e)}")
        assistant = None

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with health check."""
    return HealthResponse(
        status="healthy" if assistant else "unhealthy",
        message="AggieConnect API is running" if assistant else "Assistant not initialized",
        version="1.0.0"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if assistant else "unhealthy",
        message="Service is operational" if assistant else "Service unavailable",
        version="1.0.0"
    )

@app.post("/query", response_model=QueryResponse)
async def query_assistant(request: QueryRequest):
    """Query the assistant with a question."""
    if not assistant:
        raise HTTPException(status_code=503, detail="Assistant not initialized")
    
    try:
        result = assistant.rag_pipeline.query(request.question, k=request.k)
        
        return QueryResponse(
            question=request.question,
            response=result['response'],
            sources=result['num_sources'],
            context=result.get('retrieved_context', [])
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/help-topics", response_model=List[str])
async def get_help_topics():
    """Get list of help topics."""
    if not assistant:
        raise HTTPException(status_code=503, detail="Assistant not initialized")
    
    return assistant.get_help_topics()

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get vector store statistics."""
    if not assistant:
        raise HTTPException(status_code=503, detail="Assistant not initialized")
    
    try:
        stats = assistant.rag_pipeline.vector_store.get_stats()
        return StatsResponse(**stats)
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@app.get("/conversation-history")
async def get_conversation_history():
    """Get conversation history."""
    if not assistant:
        raise HTTPException(status_code=503, detail="Assistant not initialized")
    
    return assistant.get_conversation_history()

@app.delete("/conversation-history")
async def clear_conversation_history():
    """Clear conversation history."""
    if not assistant:
        raise HTTPException(status_code=503, detail="Assistant not initialized")
    
    assistant.clear_history()
    return {"message": "Conversation history cleared"}

@app.get("/search")
async def search_documents(query: str, k: int = 5):
    """Search documents without generating a response."""
    if not assistant:
        raise HTTPException(status_code=503, detail="Assistant not initialized")
    
    try:
        results = assistant.rag_pipeline.retrieve_context(query, k=k)
        return {
            "query": query,
            "results": results,
            "num_results": len(results)
        }
    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in search: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=DEBUG
    )
