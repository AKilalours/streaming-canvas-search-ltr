from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, FastAPI
from fastapi.responses import HTMLResponse
from starlette.staticfiles import StaticFiles


def mount_demo(app: FastAPI) -> None:
    """
    Serves a Netflix-like demo UI.

    - Static assets at: /demo/ui/{index.html, app.js, styles.css}
    - Convenience page at: /demo (serves the same index.html)
    """
    ui_dir = Path(__file__).resolve().parent / "demo_ui"
    ui_dir.mkdir(parents=True, exist_ok=True)

    # Serve /demo/ui/ (directory index) properly
    app.mount("/demo/ui", StaticFiles(directory=str(ui_dir), html=True), name="demo-ui")

    router = APIRouter()

    @router.get("/demo", response_class=HTMLResponse, response_model=None)
    def demo_page() -> str:
        index = ui_dir / "index.html"
        if not index.exists():
            return "<html><body><h2>UI missing</h2><p>Create src/app/demo_ui/index.html</p></body></html>"
        return index.read_text(encoding="utf-8")

    app.include_router(router)
