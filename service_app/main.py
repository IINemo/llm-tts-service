"""
LLM Test-Time Scaling Service - OpenAI-compatible API
"""

import logging
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from service_app.api.routes import chat, debugger, models
from service_app.core.config import settings
from service_app.core.logging_config import setup_logging

# Allow running this file directly from inside the `service_app` directory:
# This is useful for development and testing
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

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
app.include_router(debugger.router, tags=["Visual Debugger"])

# Serve static assets for the visual debugger demo.
static_dir = Path(__file__).resolve().parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

home_html_path = Path(__file__).resolve().parent / "static" / "home" / "index.html"


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint."""
    if home_html_path.exists():
        return FileResponse(home_html_path)

    return JSONResponse(
        content={
            "message": "LLM Test-Time Scaling Service",
            "version": settings.api_version,
            "docs": "/docs",
            "openai_compatible": True,
        }
    )


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
