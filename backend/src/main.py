from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import os
from typing import AsyncGenerator

from .api.chat_endpoints import router as chat_router
from .api.text_selection_endpoints import router as text_selection_router
from .api.conversation_endpoints import router as conversation_router  # New conversation endpoints
from .api.module_context_endpoints import router as module_context_router  # New module context endpoints
from .api.system_endpoints import router as system_router  # New system endpoints


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan event handler for FastAPI application
    """
    # Startup
    print("Starting up RAG Chatbot API...")
    
    # Initialize any resources here
    # For example: database connections, external service connections, etc.
    
    yield  # This is where the app runs
    
    # Shutdown
    print("Shutting down RAG Chatbot API...")
    # Clean up any resources here


# Create the main FastAPI application
app = FastAPI(
    title="RAG Chatbot for Physical AI & Humanoid Robotics Textbook",
    description="An intelligent chatbot that helps students with the Physical AI & Humanoid Robotics textbook using Retrieval-Augmented Generation",
    version="1.0.0",
    lifespan=lifespan
)

from .middleware.error_handling import ErrorHandlingMiddleware, add_security_headers, add_rate_limiting
from .middleware.logging import LoggingMiddleware

# Add custom error handling middleware first
app.add_middleware(ErrorHandlingMiddleware)

# Add logging middleware
app.add_middleware(LoggingMiddleware)

# Add CORS middleware
allowed_origins_raw = os.getenv("ALLOWED_ORIGINS", "http://localhost,http://localhost:3000")
allowed_origins = [origin.strip() for origin in allowed_origins_raw.split(",") if origin.strip()]

# Add wildcard and additional Vercel domains if not present for better resilience
# This helps handle different Vercel deployment URLs (like preview or branch URLs)
if "*" not in allowed_origins:
    # Explicitly include the user's observed Vercel domain patterns
    vercel_patterns = ["physical-ai-book.vercel.app", "physical-ai-book-five-ivory.vercel.app"]
    for pattern in vercel_patterns:
        full_origin = f"https://{pattern}"
        if full_origin not in allowed_origins:
            # We check if the user has any of these patterns in their allowed origins or if we're on Vercel
            allowed_origins.append(full_origin)
            allowed_origins.append(f"{full_origin}/") # Include trailing slash variant

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Add security headers
add_security_headers(app)

# Add rate limiting
add_rate_limiting(app)

# Include API routers
app.include_router(chat_router)
app.include_router(text_selection_router)
app.include_router(conversation_router)  # Include new conversation endpoints
app.include_router(module_context_router)  # Include new module context endpoints
app.include_router(system_router)  # Include new system endpoints

# Additional API endpoints can be added here
@app.get("/")
async def root():
    """
    Root endpoint for the API
    """
    return {
        "message": "Welcome to the RAG Chatbot API for Physical AI & Humanoid Robotics Textbook",
        "version": "1.0.0",
        "endpoints": [
            "/api/chat/",
            "/api/text-selection/",
            "/api/conversation/",
            "/api/module-context/",
            "/api/system/"
        ]
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "service": "rag-chatbot-api",
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }


if __name__ == "__main__":
    # This allows running the app directly for development
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("src.main:app", host="0.0.0.0", port=port, reload=True)