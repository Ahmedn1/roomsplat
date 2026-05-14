"""
Serve the static viewer directory over HTTP and open a browser tab.
"""

import http.server
import os
import threading
import webbrowser
from pathlib import Path

from rich.console import Console

console = Console()


class _SilentHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        pass  # suppress request logs

    def end_headers(self) -> None:
        # Allow SharedArrayBuffer (needed for wasm-based sort in future)
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()


def serve(viewer_dir: Path, port: int = 8080) -> None:
    """
    Start an HTTP server for viewer_dir and open the browser.
    Blocks until the user presses Ctrl+C.
    """
    os.chdir(viewer_dir)

    def _open_browser():
        import time
        time.sleep(0.5)  # give server a moment to bind
        webbrowser.open(f"http://localhost:{port}/index.html")

    # Find a free port if the requested one is taken
    import socket
    for candidate in range(port, port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", candidate)) != 0:
                port = candidate
                break

    threading.Thread(target=_open_browser, daemon=True).start()

    with http.server.HTTPServer(("", port), _SilentHandler) as httpd:
        console.print(
            f"[bold cyan]Viewer running at[/] http://localhost:{port}/index.html\n"
            "Press [bold]Ctrl+C[/] to stop."
        )
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            console.print("\n[dim]Viewer stopped.[/]")
