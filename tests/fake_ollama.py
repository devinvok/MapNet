"""A stand-in Ollama server for tests: speaks /api/tags, /api/chat (streaming),
/api/generate (JSON section cards, questions, judge) and /api/embed."""
import hashlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    calls: dict = {}

    def log_message(self, *a):
        pass

    def _send(self, obj):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps(obj).encode())

    def do_GET(self):
        self._send({"models": [{"name": "qwen2.5:3b"}, {"name": "qwen2.5:7b"},
                               {"name": "nomic-embed-text:latest"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Handler.calls[self.path] = Handler.calls.get(self.path, 0) + 1
        if self.path == "/api/embed":
            vecs = []
            for t in body["input"]:
                v = [0.0] * 16
                for w in re.findall(r"\w+", t.lower()):
                    v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 16] += 1
                vecs.append(v)
            return self._send({"embeddings": vecs})
        if self.path == "/api/chat":
            ctx = body["messages"][1]["content"]
            first = ctx.split("---", 3)[-1][:120].replace("\n", " ")
            if body["model"] == "qwen2.5:7b":  # the "bigger reader" quotes more of the passages
                first = ctx.split("Question:")[0][-2000:].replace("\n", " ")
            self.send_response(200)
            self.end_headers()
            for piece in [first, " (source: x)"]:
                self.wfile.write((json.dumps({"message": {"content": piece}, "done": False}) + "\n").encode())
            self.wfile.write((json.dumps({"message": {"content": ""}, "done": True,
                                          "prompt_eval_count": len(ctx) // 4, "eval_count": 30}) + "\n").encode())
            return
        prompt = body.get("prompt", "")
        if body.get("format") == "json" and "navigation index" in prompt:
            n = prompt.count("\n[")
            cards = [{"n": i + 1, "about": "section summary in plain words",
                      "asks": ["how do I do this", "why does this happen"]} for i in range(n)]
            return self._send({"response": json.dumps({"cards": cards})})
        if body.get("format") == "json":
            return self._send({"response": json.dumps({"question": "What is in this file?",
                                                       "answer": "something"})})
        return self._send({"response": "yes"})


def start():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}"
