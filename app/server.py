"""Local HTTP server for the IR chat app.

    python app/server.py            # then open http://127.0.0.1:8765

Standard library only, so the app adds no dependency to requirements.txt. It
binds to the loopback interface: this is a single-user analysis tool for the
assignment, not a service, and the corpus and index are held in one process.
"""

import argparse
import json
import mimetypes
import pathlib
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from app.engine import ChatEngine
    from app.session import Session
else:
    from .engine import ChatEngine
    from .session import Session

STATIC = pathlib.Path(__file__).resolve().parent / "static"
MAX_BODY = 64 * 1024


class ChatServer(ThreadingHTTPServer):
    """Holds the one shared engine, plus the lock that serialises chat turns.

    Turns are serialised because a turn may build an index on demand, which
    mutates the session. Everything else is read-only.
    """

    def __init__(self, address, handler, engine: ChatEngine) -> None:
        super().__init__(address, handler)
        self.engine = engine
        self.turn_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    server_version = "IRChat/1.0"

    def log_message(self, *args):  # quieter than the default access log
        if self.path.startswith("/api/"):
            sys.stderr.write(f"  {self.command} {self.path}\n")

    @property
    def engine(self) -> ChatEngine:
        return self.server.engine

    # ------------------------------------------------------------- routing

    def do_GET(self) -> None:
        route = urlparse(self.path)
        if route.path in ("/", "/index.html"):
            return self._send_static("index.html")
        if route.path == "/api/boot":
            with self.server.turn_lock:
                return self._send_json(
                    {
                        "state": self.engine.state(),
                        "blocks": self.engine.handle(":help")["blocks"],
                    }
                )
        if route.path == "/api/doc":
            ids = parse_qs(route.query).get("id", [])
            if not ids or not ids[0].isdigit():
                return self._send_json({"error": "bad id"}, 400)
            doc = self.engine.s.document(int(ids[0]))
            if doc is None:
                return self._send_json({"error": "not found"}, 404)
            return self._send_json(
                {
                    "doc_id": doc.doc_id,
                    "category": doc.category,
                    "filename": doc.filename,
                    "text": doc.text,
                }
            )
        if route.path.startswith("/static/"):
            return self._send_static(route.path[len("/static/") :])
        self._send_json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/chat":
            return self._send_json({"error": "not found"}, 404)
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY:
            return self._send_json({"error": "bad request body"}, 400)
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return self._send_json({"error": "malformed JSON"}, 400)
        message = payload.get("message")
        if not isinstance(message, str):
            return self._send_json({"error": "message must be a string"}, 400)
        with self.server.turn_lock:
            reply = self.engine.handle(message)
        self._send_json(reply)

    # ------------------------------------------------------------ plumbing

    def _send_static(self, relative: str) -> None:
        # resolve() then containment check: blocks ../ traversal out of static/.
        target = (STATIC / relative).resolve()
        if not target.is_file() or STATIC.resolve() not in target.parents:
            return self._send_json({"error": "not found"}, 404)
        body = target.read_bytes()
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="IR chat app")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    print("Loading corpus and building the default index...")
    session = Session()
    session.index()
    print(
        f"  {len(session.docs)} documents, "
        f"{len(session.index().terms):,} terms in '{session.active}'"
    )

    url = f"http://127.0.0.1:{args.port}/"
    httpd = ChatServer(("127.0.0.1", args.port), Handler, ChatEngine(session))
    print(f"Chat app ready at {url}   (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
