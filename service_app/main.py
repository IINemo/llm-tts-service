"""
LLM Test-Time Scaling Service - OpenAI-compatible API
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from service_app.api.routes import chat, models
from service_app.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

log = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="""
    LLM Test-Time Scaling Service with OpenAI-compatible API.

    This service exposes TTS strategies (DeepConf, Tree-of-Thoughts, Best-of-N, etc.) through
    an OpenAI-compatible interface. You can use the OpenAI Python SDK or
    any OpenAI-compatible client to interact with this service.

    ## Features
    - Drop-in replacement for OpenAI's Chat Completions API
    - Supports DeepConf strategy (offline and online modes)
    - Supports Tree-of-Thoughts (ToT) strategy with beam search
    - Compatible with OpenAI Python SDK
    - Additional TTS-specific parameters for advanced control

    ## Quick Start

    ```python
    from openai import OpenAI

    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="dummy"  # Not required yet
    )

    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "Solve: 2+2=?"}],
        # TTS parameters
        extra_body={
            "tts_strategy": "deepconf",
            "tts_mode": "offline",
            "tts_budget": 8
        }
    )

    print(response.choices[0].message.content)
    ```

    ## Environment Variables
    - `OPENROUTER_API_KEY`: OpenRouter API key (required)
    - `OPENAI_API_KEY`: Direct OpenAI API key (optional)
    """,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allow_origins,
    allow_credentials=settings.allow_credentials,
    allow_methods=settings.allow_methods,
    allow_headers=settings.allow_headers,
)

# Include routers
app.include_router(chat.router, tags=["Chat Completions"])
app.include_router(models.router, tags=["Models"])


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint."""
    return {
        "message": "LLM Test-Time Scaling Service",
        "version": settings.api_version,
        "docs": "/docs",
        "openai_compatible": True,
    }


@app.get("/health", tags=["Health"])
@app.get("/v1/health", tags=["Health"])
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": settings.api_version,
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    log.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "type": "internal_error",
                "code": "internal_server_error",
            }
        },
    )


if __name__ == "__main__":
    import uvicorn

    log.info(f"Starting {settings.api_title} v{settings.api_version}")
    log.info(f"Server running on http://{settings.host}:{settings.port}")
    log.info(f"OpenAPI docs: http://{settings.host}:{settings.port}/docs")

    uvicorn.run(
        "service_app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,  # Enable auto-reload for development
        log_level="info",
    )
