from http.server import BaseHTTPRequestHandler, HTTPServer
import json

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/show-image":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"not found")
            return
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length).decode('utf-8') if length else ''
        try:
            data = json.loads(body) if body else {}
        except Exception:
            data = {"_raw": body}
        print(f"[projector] /show-image payload: {data}")
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"code": 0, "message": "ok"}).encode('utf-8'))

if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", 9000), Handler)
    print("Projector mock listening on http://127.0.0.1:9000 ...")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
