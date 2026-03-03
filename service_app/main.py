"""
LLM Test-Time Scaling Service - OpenAI-compatible API
"""

import json
import logging
import sys
from html import escape
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
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

# Configure logging - each run gets logs/<date>/<time>/service.log
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
deploy_doc_path = static_dir / "DEPLOY.md"


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
            "deploy": "/deploy",
            "openai_compatible": True,
        }
    )


@app.get("/deploy", tags=["Docs"])
@app.get("/deploy/", tags=["Docs"])
async def deploy_guide():
    """Render deployment guide in browser for standalone demo usage."""
    if deploy_doc_path.exists():
        markdown_content = deploy_doc_path.read_text(encoding="utf-8")
        markdown_json = json.dumps(markdown_content)
        escaped_markdown = escape(markdown_content)

        # Render markdown in-browser (like docs pages) and fall back to raw text.
        page = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Deployment Guide</title>
    <style>
      :root {{
        --bg: #fff9ef;
        --ink: #10243a;
        --muted: #5f6d7c;
        --line: #d2dae3;
        --card: #ffffff;
      }}
      * {{
        box-sizing: border-box;
      }}
      body {{
        margin: 0;
        background: var(--bg);
        color: var(--ink);
        font-family: "Space Grotesk", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }}
      .shell {{
        width: min(980px, 92vw);
        margin: 1.5rem auto 2rem;
      }}
      .panel {{
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 1.2rem 1.2rem 1.4rem;
      }}
      .topbar {{
        display: flex;
        gap: 0.7rem;
        flex-wrap: wrap;
        margin-bottom: 1rem;
      }}
      .btn {{
        display: inline-block;
        text-decoration: none;
        color: var(--ink);
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 0.5rem 0.75rem;
        font-size: 0.93rem;
      }}
      .btn:hover {{
        border-color: #98acc2;
      }}
      #deploy-content {{
        line-height: 1.62;
      }}
      #deploy-content h1,
      #deploy-content h2,
      #deploy-content h3 {{
        line-height: 1.25;
        margin-top: 1.25rem;
      }}
      #deploy-content pre {{
        background: #0f1720;
        color: #f2f7ff;
        border-radius: 10px;
        padding: 0.9rem;
        overflow-x: auto;
      }}
      #deploy-content code {{
        font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
      }}
      #deploy-content :not(pre) > code {{
        background: #eef3f8;
        border: 1px solid #d8e1eb;
        border-radius: 6px;
        padding: 0.08rem 0.35rem;
      }}
      #deploy-content blockquote {{
        border-left: 3px solid #d2dae3;
        margin: 0.9rem 0;
        padding-left: 0.8rem;
        color: var(--muted);
      }}
      #deploy-content table {{
        border-collapse: collapse;
        width: 100%;
      }}
      #deploy-content th,
      #deploy-content td {{
        border: 1px solid #dce5ef;
        padding: 0.42rem 0.52rem;
        text-align: left;
      }}
      #deploy-content hr {{
        border: 0;
        border-top: 1px solid #dce5ef;
        margin: 1.1rem 0;
      }}
    </style>
  </head>
  <body>
    <main class="shell">
      <section class="panel">
        <div class="topbar">
          <a class="btn" href="/">Home</a>
          <a class="btn" href="/docs">API Docs</a>
          <a class="btn" href="/debugger">Debugger</a>
        </div>
        <article id="deploy-content"><pre><code>{escaped_markdown}</code></pre></article>
      </section>
    </main>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script>
      (function renderDeployMarkdown() {{
        var source = {markdown_json};
        var root = document.getElementById("deploy-content");
        if (window.marked) {{
          root.innerHTML = marked.parse(source);
        }}
      }})();
    </script>
  </body>
</html>
"""
        return HTMLResponse(content=page)

    return JSONResponse(
        status_code=404,
        content={
            "error": {
                "message": "Deployment guide not found",
                "type": "not_found_error",
                "code": "deployment_guide_not_found",
            }
        },
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
