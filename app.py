"""
app.py
======
LazyFin -- Dash application entry point.

Run
---
    python app.py
    # then open http://127.0.0.1:8050

Production (single-worker required -- cache is in-process)
---
    gunicorn "app:server" --bind 0.0.0.0:8050 --workers 1 --timeout 120
"""

from __future__ import annotations

import logging
import os
import sys

# Ensure project root is on sys.path so aleph_toolkit resolves as a package.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import dash
import dash_bootstrap_components as dbc

from lazyfin.cache import PortfolioCache
from layout import create_layout
from callbacks import register_callbacks

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt = "%H:%M:%S",
)
_LOG = logging.getLogger("lazyfin.app")

# ---------------------------------------------------------------------------
# Shared in-process cache (single instance for the process lifetime)
# ---------------------------------------------------------------------------
CACHE: PortfolioCache = PortfolioCache(ttl_hours=8.0)
_LOG.info("PortfolioCache initialised (TTL = 8.0 h).")

# ---------------------------------------------------------------------------
# Dash application
# ---------------------------------------------------------------------------
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap",
    ],
    suppress_callback_exceptions=True,
    title="LazyFin",
    update_title=None,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
    ],
)

app.layout = create_layout()
register_callbacks(app, CACHE)

_LOG.info("Layout and callbacks registered.")

# WSGI entry point for gunicorn / uWSGI
server = app.server

# ---------------------------------------------------------------------------
# Dev server
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _LOG.info("Starting dev server at http://127.0.0.1:8050")
    app.run(
        debug                = True,
        host                 = "127.0.0.1",
        port                 = 8050,
        dev_tools_hot_reload = True,
    )
