import os
from pypdf import PdfReader
import requests
import tempfile
import logging
import hashlib
import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Pinecone v3 imports
from pinecone import Pinecone, ServerlessSpec

# FastAPI imports
from fastapi import FastAPI, Request, Header, HTTPException, BackgroundTasks, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Environment Variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hackrx-bot")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Pinecone v3
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    logger.info("Pinecone initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Pinecone: {e}")
    raise

# Pydantic Models
class HackRxRequest(BaseModel):
    documents: str  # PDF URL
    questions: List[str]
    chunk_size: Optional[int] = 1000
    chunk_overlap: Optional[int] = 200
    top_k: Optional[int] = 10
    temperature: Optional[float] = 0.1
    use_hybrid_search: Optional[bool] = True
    
    @field_validator('questions')
    def validate_questions(cls, v):
        if not v or len(v) == 0:
            raise ValueError("At least one question is required")
        return v
    
    @field_validator('chunk_size')
    def validate_chunk_size(cls, v):
        if v < 100 or v > 4000:
            raise ValueError("Chunk size must be between 100 and 4000")
        return v

class HackRxResponse(BaseModel):
    answers: List[Dict[str, Any]]
    document_id: str
    processing_time: float
    total_chunks: int
    metadata: Dict[str, Any]

class DocumentMetadata(BaseModel):
    document_id: str
    url: str
    total_pages: int
    total_chunks: int
    processed_at: datetime
    file_size: int

# FastAPI App
app = FastAPI(
    title="Advanced PDF Search API",
    description="Advanced PDF search using Gemini AI and Pinecone vector database",
    version="2.0.0"
)

security = HTTPBearer()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for caching
embedding_model = None
document_cache = {}

class AdvancedPDFProcessor:
    def __init__(self):
        self.embedding_model = self._load_embedding_model()
        self.text_splitter = None
        
    def _load_embedding_model(self):
        """Load embedding model with caching"""
        global embedding_model
        if embedding_model is None:
            try:
                embedding_model = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                logger.info("Embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
        return embedding_model
    
    def download_pdf(self, url: str) -> tuple[str, int]:
        """Download PDF with better error handling and size tracking"""
        try:
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '')
            if 'pdf' not in content_type.lower():
                logger.warning(f"Content type might not be PDF: {content_type}")
            
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            file_size = 0
            
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp.write(chunk)
                    file_size += len(chunk)
            
            temp.close()
            logger.info(f"Downloaded PDF: {file_size} bytes")
            return temp.name, file_size
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download PDF: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to download PDF: {str(e)}")
    
    def extract_text_advanced(self, pdf_path: str) -> tuple[str, int, List[Dict]]:
        """Extract text with metadata and better handling"""
        try:
            reader = PdfReader(pdf_path)
            pages_data = []
            full_text = ""
            
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                
                # Extract additional metadata
                page_info = {
                    'page_number': page_num + 1,
                    'text': page_text,
                    'char_count': len(page_text),
                    'word_count': len(page_text.split()),
                }
                
                pages_data.append(page_info)
                full_text += f"\n\n--- Page {page_num + 1} ---\n\n{page_text}"
            
            logger.info(f"Extracted text from {len(pages_data)} pages")
            return full_text, len(pages_data), pages_data
            
        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to process PDF: {str(e)}")
    
    def chunk_text_advanced(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict]:
        """Advanced text chunking with metadata"""
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""],
                length_function=len,
            )
            
            chunks = splitter.split_text(text)
            
            # Add metadata to chunks
            chunk_data = []
            for i, chunk in enumerate(chunks):
                chunk_info = {
                    'id': f"chunk-{i}",
                    'text': chunk,
                    'char_count': len(chunk),
                    'word_count': len(chunk.split()),
                    'chunk_index': i
                }
                chunk_data.append(chunk_info)
            
            logger.info(f"Created {len(chunk_data)} chunks")
            return chunk_data
            
        except Exception as e:
            logger.error(f"Failed to chunk text: {e}")
            raise HTTPException(status_code=500, detail="Failed to process text chunks")
    
    def generate_document_id(self, url: str) -> str:
        """Generate unique document ID"""
        return hashlib.md5(url.encode()).hexdigest()
    
    def setup_pinecone_index(self, document_id: str) -> str:
        """Setup Pinecone index with proper configuration"""
        try:
            index_name = PINECONE_INDEX_NAME
            
            # Check if index exists
            existing_indexes = [index.name for index in pc.list_indexes()]
            
            if index_name not in existing_indexes:
                pc.create_index(
                    name=index_name,
                    dimension=384,  # Dimension for all-MiniLM-L6-v2
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                logger.info(f"Created new Pinecone index: {index_name}")
            else:
                logger.info(f"Using existing Pinecone index: {index_name}")
                
            return index_name
            
        except Exception as e:
            logger.error(f"Failed to setup Pinecone index: {e}")
            raise HTTPException(status_code=500, detail="Failed to setup vector database")
    
    def upsert_to_pinecone(self, chunks_data: List[Dict], document_id: str) -> PineconeVectorStore:
        """Upsert chunks to Pinecone with batch processing"""
        try:
            index_name = self.setup_pinecone_index(document_id)
            
            # Extract texts for embedding
            texts = [chunk['text'] for chunk in chunks_data]
            
            # Create vector store
            vectorstore = PineconeVectorStore.from_texts(
                texts=texts,
                embedding=self.embedding_model,
                index_name=index_name,
                namespace=document_id,
                metadatas=[{
                    'document_id': document_id,
                    'chunk_id': chunk['id'],
                    'char_count': chunk['char_count'],
                    'word_count': chunk['word_count'],
                    'chunk_index': chunk['chunk_index']
                } for chunk in chunks_data]
            )
            
            logger.info(f"Upserted {len(chunks_data)} chunks to Pinecone")
            return vectorstore
            
        except Exception as e:
            logger.error(f"Failed to upsert to Pinecone: {e}")
            raise HTTPException(status_code=500, detail="Failed to store in vector database")
    
    def setup_advanced_rag_chain(self, vectorstore: PineconeVectorStore, top_k: int = 10, temperature: float = 0.1):
        """Setup advanced RAG chain with custom prompt and memory"""
        try:
            # Custom prompt template
            prompt_template = """
            You are an expert document analyst. Use the following pieces of context to answer the question at the end.
            If you don't know the answer based on the context provided, just say that you don't know, don't try to make up an answer.
            
            Context:
            {context}
            
            Question: {input}
            
            Instructions:
            1. Provide a comprehensive and accurate answer based on the context
            2. Include relevant details and specifics from the document
            3. If applicable, mention page numbers or sections
            4. Structure your response clearly
            5. If the information is incomplete, acknowledge the limitations
            
            Answer:
            """
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "input"]
            )
            
            # Setup retriever with advanced search
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": top_k,
                    "include_metadata": True
                }
            )
            
            # Initialize Gemini with custom parameters
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=GOOGLE_API_KEY,
                temperature=temperature,
                max_tokens=2048,
                convert_system_message_to_human=True
            )
            
            # Create RAG chain using LCEL
            combine_docs_chain = create_stuff_documents_chain(llm, PROMPT)
            qa_chain = create_retrieval_chain(retriever, combine_docs_chain)
            
            logger.info("RAG chain setup completed")
            return qa_chain
            
        except Exception as e:
            logger.error(f"Failed to setup RAG chain: {e}")
            raise HTTPException(status_code=500, detail="Failed to setup AI chain")

# Initialize processor
processor = AdvancedPDFProcessor()

@app.post("/hackrx/run", response_model=HackRxResponse)
async def handle_hackrx_advanced(
    req: HackRxRequest, 
    credentials: HTTPAuthorizationCredentials = Depends(security),
    background_tasks: BackgroundTasks = None
):
    """Advanced PDF processing endpoint with comprehensive error handling"""
    start_time = asyncio.get_event_loop().time()
    
    try:
        # Generate document ID
        document_id = processor.generate_document_id(req.documents)
        logger.info(f"Processing document ID: {document_id}")
        
        # Check cache first
        if document_id in document_cache:
            logger.info("Using cached document processing")
            vectorstore = document_cache[document_id]['vectorstore']
            total_chunks = document_cache[document_id]['total_chunks']
        else:
            # Download and process PDF
            pdf_path, file_size = processor.download_pdf(req.documents)
            
            try:
                # Extract text
                full_text, total_pages, pages_data = processor.extract_text_advanced(pdf_path)
                
                # Chunk text
                chunks_data = processor.chunk_text_advanced(
                    full_text, 
                    req.chunk_size, 
                    req.chunk_overlap
                )
                
                # Store in vector database
                vectorstore = processor.upsert_to_pinecone(chunks_data, document_id)
                
                # Cache the results
                document_cache[document_id] = {
                    'vectorstore': vectorstore,
                    'total_chunks': len(chunks_data),
                    'metadata': {
                        'total_pages': total_pages,
                        'file_size': file_size,
                        'processed_at': datetime.now()
                    }
                }
                
                total_chunks = len(chunks_data)
                
            finally:
                # Cleanup temporary file
                if os.path.exists(pdf_path):
                    os.unlink(pdf_path)
        
        # Setup RAG chain
        qa_chain = processor.setup_advanced_rag_chain(
            vectorstore, 
            req.top_k, 
            req.temperature
        )
        
        # Process questions
        answers = []
        for i, question in enumerate(req.questions):
            try:
                logger.info(f"Processing question {i+1}/{len(req.questions)}")
                
                result = qa_chain.invoke({"input": question})
                
                answer_data = {
                    "question": question,
                    "answer": result.get("answer", ""),
                    "confidence": "high",  # You could implement confidence scoring
                    "sources": [
                        {
                            "chunk_id": doc.metadata.get("chunk_id", "unknown"),
                            "relevance_score": 1.0,  # You could get actual scores
                            "text_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                        }
                        for doc in result.get("context", [])
                    ]
                }
                
                answers.append(answer_data)
                
            except Exception as e:
                logger.error(f"Failed to process question {i+1}: {e}")
                answers.append({
                    "question": question,
                    "answer": f"Error processing question: {str(e)}",
                    "confidence": "low",
                    "sources": []
                })
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        response = HackRxResponse(
            answers=answers,
            document_id=document_id,
            processing_time=round(processing_time, 2),
            total_chunks=total_chunks,
            metadata={
                "chunk_size": req.chunk_size,
                "chunk_overlap": req.chunk_overlap,
                "top_k": req.top_k,
                "temperature": req.temperature,
                "model_used": "gemini-1.5-flash",
                "embedding_model": "all-MiniLM-L6-v2"
            }
        )
        
        logger.info(f"Successfully processed {len(req.questions)} questions in {processing_time:.2f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }

@app.get("/cache/info")
async def cache_info():
    """Get cache information"""
    return {
        "cached_documents": len(document_cache),
        "cache_keys": list(document_cache.keys())
    }

@app.delete("/cache/clear")
async def clear_cache():
    """Clear document cache"""
    global document_cache
    document_cache.clear()
    return {"message": "Cache cleared successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)