"""Servidor Groq falso para simular falhas. Uso: groq_falso.py <porta> <sequencia>
sequência de respostas (repete a última): ok, 429, 429long, 401, 500, hang, 413. Ex.: "ok,429long"
Cada resposta ok devolve 1 palavra no meio do trecho recebido (duração inferida do tamanho do FLAC não é possível; usa 1,0 s)."""
import json, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

seq = sys.argv[2].split(",")
n = {"i": 0}
log = []


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_POST(self):
        tam = int(self.headers.get("content-length", 0))
        self.rfile.read(tam)
        modo = seq[min(n["i"], len(seq) - 1)]
        n["i"] += 1
        print(f"req {n['i']} modo={modo} bytes={tam} auth={'sim' if self.headers.get('Authorization') else 'nao'}", flush=True)

        def resp(code, corpo, hdr=None):
            b = json.dumps(corpo).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            for k, v in (hdr or {}).items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers(); self.wfile.write(b)

        if modo == "ok":
            resp(200, {"text": "fake", "words": [{"word": "fake", "start": 3.0, "end": 3.4}],
                       "segments": [{"start": 3.0, "end": 3.4, "text": " fake", "avg_logprob": -0.1, "no_speech_prob": 0.0, "compression_ratio": 1.0}]})
        elif modo == "429":
            resp(429, {"error": {"message": "rate limit"}}, {"retry-after": "1"})
        elif modo == "429long":
            resp(429, {"error": {"message": "daily limit"}}, {"retry-after": "3000"})
        elif modo == "401":
            resp(401, {"error": {"message": "Invalid API Key"}})
        elif modo == "500":
            resp(503, {"error": {"message": "over capacity"}})
        elif modo == "413":
            resp(413, {"error": {"message": "file too large"}})
        elif modo == "hang":
            time.sleep(30)
            resp(200, {"text": "", "words": [], "segments": []})


ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
