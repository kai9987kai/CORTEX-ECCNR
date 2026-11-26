"""Launch the CORTEX-ECCNR Interactive Scientific GUI (Desktop pywebview window or Browser)."""
from __future__ import annotations

import argparse
import socket
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn

from eccnr.gui_server import create_app


def _find_free_port(preferred: int = 8765) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", preferred)) != 0:
            return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch CORTEX-ECCNR Interactive GUI Observatory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--desktop", action="store_true", help="Open native desktop window via pywebview")
    parser.add_argument("--headless", action="store_true", help="Start server without opening browser/window")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    port = _find_free_port(args.port)
    url = f"http://{args.host}:{port}"
    app = create_app(project_root=args.project_root)

    if args.desktop and not args.headless:
        def _run_server() -> None:
            uvicorn.run(app, host=args.host, port=port, log_level="warning")

        t = threading.Thread(target=_run_server, daemon=True)
        t.start()
        time.sleep(0.8)
        try:
            import webview
            print(f"Opening native desktop CORTEX-ECCNR Observatory at {url} ...")
            webview.create_window("CORTEX-ECCNR — Evolutionary Connectome Observatory", url, width=1420, height=920)
            webview.start()
            return
        except Exception as exc:
            print(f"Desktop webview unavailable ({exc}); opening browser at {url} ...")
            webbrowser.open(url)
            while t.is_alive():
                time.sleep(1.0)
            return

    if not args.headless:
        threading.Timer(0.9, lambda: webbrowser.open(url)).start()
    print(f"CORTEX-ECCNR Observatory running at {url}")
    uvicorn.run(app, host=args.host, port=port, log_level="info")


if __name__ == "__main__":
    main()
