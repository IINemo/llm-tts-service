"""
LLM Test-Time Scaling Service - OpenAI-compatible API
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from service_app.api.routes import chat, models
from service_app.core.config import settings
from service_app.core.logging_config import setup_logging

# Configure logging — each run gets logs/<date>/<time>/service.log
_log_dir = setup_logging()

log = logging.getLogger(__name__)
log.info(f"Logging to {_log_dir}/service.log")

# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="""
    LLM Test-Time Scaling Service with OpenAI-compatible API.

    This service exposes TTS strategies (Self-Consistency, Offline Best-of-N,
    Online Best-of-N, Beam Search) through an OpenAI-compatible interface.
    You can use the OpenAI Python SDK or any OpenAI-compatible client.

    ## Features
    - Drop-in replacement for OpenAI's Chat Completions API
    - Self-consistency strategy via OpenAI/OpenRouter APIs
    - Offline Best-of-N, Online Best-of-N, Beam Search via local vLLM backend
    - Multiple scorers: entropy, perplexity, sequence_prob, PRM
    - Compatible with OpenAI Python SDK
    - Additional TTS-specific parameters for advanced control

    ## Quick Start

    ```python
    from openai import OpenAI

    client = OpenAI(
        base_url="http://localhost:8001/v1",
        api_key="your-openrouter-key"
    )

    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Reason step by step, put answer in \\\\boxed{}."},
            {"role": "user", "content": "What is 15 * 7?"}
        ],
        extra_body={
            "tts_strategy": "self_consistency",
            "num_paths": 5
        }
    )

    print(response.choices[0].message.content)
    ```

    ## Environment Variables
    - `OPENROUTER_API_KEY`: OpenRouter API key (required for self_consistency)
    - `OPENAI_API_KEY`: Direct OpenAI API key (optional)
    - `VLLM_MODEL_PATH`: Local model for vLLM strategies (optional)
    - `PRM_MODEL_PATH`: PRM scorer model path (optional)
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
        reload_excludes=["logs/*"],
        log_level="info",
    )
