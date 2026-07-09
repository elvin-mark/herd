import os
import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from herd.api.state import manager
from herd.core.metrics import collector
from herd.core.utils import close_http_clients
from herd.api.exceptions import HerdError

from herd.api.routers.models import router as models_router
from herd.api.routers.chat import router as chat_router
from herd.api.routers.db import router as db_router

logger = logging.getLogger("herd.server")
cleanup_task = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Configure logging for the gateway server process dynamically on startup
    from rich.logging import RichHandler

    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    rich_handler = RichHandler(rich_tracebacks=True, show_path=True)
    root_logger.addHandler(rich_handler)
    root_logger.setLevel(logging.INFO)

    # Silence httpx info logs to prevent polling request spam
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Startup: start process manager idle cleanup background loop
    global cleanup_task
    cleanup_task = asyncio.create_task(manager.cleanup_loop())
    logger.info("Herd Gateway Server started. Idle cleanup task scheduled.")
    yield
    # Shutdown
    if cleanup_task:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
    # Stop all models concurrently
    running_models = list(manager.running_models.keys())
    tasks = []
    for model_path in running_models:
        info = manager.running_models.get(model_path)
        if info:
            tasks.append(manager.stop_model(info["model_name"]))
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    close_http_clients()
    logger.info("Herd Gateway Server stopped. All child processes terminated.")


app = FastAPI(title="Herd API Gateway", lifespan=lifespan)


@app.exception_handler(HerdError)
async def herd_error_handler(request: Request, exc: HerdError):
    logger.error(f"API Error: {exc.message} (status: {exc.status_code})")
    return JSONResponse(status_code=exc.status_code, content={"error": exc.message})


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled Exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": f"Internal Gateway Error: {str(exc)}"},
    )


# Mount assets folder to serve the logo image
assets_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "assets")
)
if os.path.exists(assets_dir):
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")


@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}


@app.get("/")
@app.get("/dashboard")
async def get_dashboard():
    """Serves the Herd Web Control Center dashboard."""
    templates_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "templates")
    )
    html_path = os.path.join(templates_dir, "dashboard.html")
    css_path = os.path.join(templates_dir, "dashboard.css")
    js_path = os.path.join(templates_dir, "dashboard.js")
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        with open(js_path, "r", encoding="utf-8") as f:
            js_content = f.read()

        content = html_content.replace(
            "<!-- INJECT_STYLE -->", f"<style>\n{css_content}\n</style>"
        ).replace("<!-- INJECT_SCRIPT -->", f"<script>\n{js_content}\n</script>")
        return Response(content=content, media_type="text/html")
    except Exception as e:
        return Response(
            content=f"Error loading dashboard: {e}",
            status_code=500,
            media_type="text/plain",
        )


@app.get("/metrics")
async def get_prometheus_metrics_endpoint():
    """Prometheus metrics scraping endpoint."""
    active = []
    async with manager.lock:
        for path, info in manager.running_models.items():
            name = info["model_name"]
            pid = info["process"].pid
            resources = manager.get_process_resources(pid)
            active.append(
                {
                    "model": name,
                    "port": info["port"],
                    "cpu_percent": resources["cpu_percent"],
                    "memory_bytes": resources["memory_bytes"],
                }
            )
    prometheus_data = collector.get_prometheus_metrics(active)
    return Response(
        content=prometheus_data,
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


# Mount APIRouters
app.include_router(models_router)
app.include_router(chat_router)
app.include_router(db_router)
