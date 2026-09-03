"""Small local HTTP service for running MedGRPO in its own conda environment."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from agent_baselines.controller import ClipCandidate, VideoRequest
from agent_baselines.medgrpo_inspector import MedGRPOInspector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--medgrpo-root", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    inspector = MedGRPOInspector(args.model, medgrpo_root=args.medgrpo_root, device=args.device, quantized=True)
    inspector._load_backend()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                data = json.loads(self.rfile.read(length))
                r = data["request"]
                c = data["candidate"]
                request = VideoRequest(**r)
                candidate = ClipCandidate(**c)
                result = inspector.inspect(request, candidate)
                body = json.dumps({"text": result.text, "metadata": result.metadata}, ensure_ascii=False).encode()
                self.send_response(200)
            except Exception as error:
                body = json.dumps({"error": f"{type(error).__name__}: {error}"}).encode()
                self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_: object) -> None:
            return

    print(f"MedGRPO server listening on http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
