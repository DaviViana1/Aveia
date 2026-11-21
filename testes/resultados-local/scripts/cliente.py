"""Cliente de medicao: aovivo (tempo real) ou classico. Uso: cliente.py aovivo|classico"""
import json, sys, time, urllib.request, uuid, math
from pathlib import Path
RAIZ = str(Path(__file__).resolve().parents[3])  # raiz do projeto
sys.path.insert(0, RAIZ)
from testes.wer import wer, normalizar
B = "http://127.0.0.1:8799"
AR = RAIZ + "/testes/audio-referencia"
OUT = RAIZ + "/testes/resultados-local/bruto"
DUR = 57.7
def req(path, data=None, method=None, ctype="application/json"):
    h = {"Content-Type": ctype} if data is not None else {}
    r = urllib.request.Request(B + path, data=data, method=method, headers=h)
    with urllib.request.urlopen(r, timeout=60) as f:
        return json.loads(f.read())
def rj(path, obj): return req(path, json.dumps(obj).encode())
dados = open(AR + r"\audio.webm", "rb").read()
ref_txt = open(AR + r"\transcricao-referencia.txt", encoding="utf-8").read()
ref_json = " ".join(f["texto"] for f in json.load(open(AR + r"\referencia.json", encoding="utf-8")))
def resumo(falas, extra):
    texto = " ".join(f["texto"] for f in falas)
    return dict(palavras=len(normalizar(texto).split()), wer_txt=round(wer(ref_txt, texto) * 100, 2),
                wer_json=round(wer(ref_json, texto) * 100, 2), falas=len(falas),
                locutores=sorted({f["locutor"] for f in falas}), texto=texto, falas_det=falas, **extra)
modo = sys.argv[1]
if modo == "aovivo":
    c = rj("/api/aovivo", {"ext": "webm", "idioma": "pt", "locutores": 0})
    assert c.get("ao_vivo"), c
    sid = c["sessao"]
    n = math.ceil(DUR / 2); tam = math.ceil(len(dados) / n)
    pedacos = [dados[i * tam:(i + 1) * tam] for i in range(n)]
    pedacos = [p for p in pedacos if p]
    t0 = time.time(); primeiro = None; consultas = []; seq = 0
    proximo_envio = 2.0; proximo_poll = 1.0
    while seq < len(pedacos):
        agora = time.time() - t0
        if agora >= proximo_envio:
            req(f"/api/aovivo/{sid}/pedaco?seq={seq}", pedacos[seq], ctype="application/octet-stream"); seq += 1
            proximo_envio = 2.0 * (seq + 1)
        if agora >= proximo_poll:
            r = req(f"/api/aovivo/{sid}"); proximo_poll += 1.0
            if r["falas"] and primeiro is None:
                primeiro = time.time() - t0
                print(f"PRIMEIRO TEXTO aos {primeiro:.1f}s: {r['falas'][0]['texto'][:60]!r}", flush=True)
            consultas.append((round(agora, 1), len(r["falas"]), r["status"]))
        time.sleep(0.05)
    t_parar = time.time(); dur_grav = t_parar - t0
    print(f"parou aos {dur_grav:.1f}s; finalizando", flush=True)
    j = rj(f"/api/aovivo/{sid}/finalizar", {"bytes": len(dados), "pedacos": len(pedacos), "duracao": DUR})
    job = j["job"]
    # polling do job; tambem consulta o texto ao vivo ate o fim se ainda nao houve
    while True:
        r = req(f"/api/jobs/{job}")
        if r["status"] != "processando": break
        time.sleep(0.5)
    t_pronto = time.time() - t_parar
    print(f"PRONTO em {t_pronto:.1f}s do parar; status {r['status']}", flush=True)
    res = resumo(r.get("falas") or [], dict(via="aovivo", primeiro_texto_s=round(primeiro, 1) if primeiro else None,
        parar_ao_pronto_s=round(t_pronto, 1), gravacao_simulada_s=round(dur_grav, 1), pedacos=len(pedacos), status=r["status"],
        erro=r.get("erro"), avisos=r.get("avisos"), lacunas=r.get("lacunas"), motor=r.get("motor"), consultas=consultas))
    json.dump(res, open(OUT + r"\aovivo_ocioso__aovivo__ref58-webm.json", "w", encoding="utf-8"), ensure_ascii=False)
else:
    bnd = uuid.uuid4().hex
    corpo = (f'--{bnd}\r\nContent-Disposition: form-data; name="audio"; filename="audio.webm"\r\nContent-Type: audio/webm\r\n\r\n').encode() + dados + \
        (f'\r\n--{bnd}\r\nContent-Disposition: form-data; name="idioma"\r\n\r\npt\r\n--{bnd}\r\nContent-Disposition: form-data; name="locutores"\r\n\r\n0\r\n--{bnd}--\r\n').encode()
    t0 = time.time()
    j = req("/api/transcrever", corpo, ctype=f"multipart/form-data; boundary={bnd}")
    while True:
        r = req(f"/api/jobs/{j['job']}")
        if r["status"] != "processando": break
        time.sleep(0.5)
    dt = time.time() - t0
    print(f"CLASSICO pronto em {dt:.1f}s; status {r['status']}", flush=True)
    res = resumo(r.get("falas") or [], dict(via="classico", envio_ao_pronto_s=round(dt, 1), status=r["status"], erro=r.get("erro"),
        avisos=r.get("avisos"), lacunas=r.get("lacunas"), motor=r.get("motor")))
    json.dump(res, open(OUT + r"\aovivo_ocioso__classico__ref58-webm.json", "w", encoding="utf-8"), ensure_ascii=False)
print({k: v for k, v in res.items() if k not in ("texto", "falas_det", "consultas")})
