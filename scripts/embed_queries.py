#!/usr/bin/env python3
"""問いの埋め込みを取ってキャッシュする（data/emb_cache/<model>.jsonl）。標準ライブラリだけ。

  python3 -I scripts/embed_queries.py bge        # llama-embed（bge-m3）。直列。load が高いと待つ
  python3 -I scripts/embed_queries.py sarashina  # sarashina-embed。20 回/分以下（最小間隔 3.2 秒）。429 か遅延で止まる

- 入力は data/questions.jsonl の query。長い問いは questions 作成時に先頭 500 字へ切ってある（本番のフックと同じ）。
- Sarashina には `[[sarashina:query]]` を付ける（本番の設定と同じ）。bge-m3 は接頭辞なし。
- キャッシュのキーは sha1(model + 接頭辞 + 本文)。あるものは撃たない（再実行で撃ち直さない）。
- 1 回ごとに所要時間（ms）と開始時刻を残す。遅延の集計はこれから取る。
- Sarashina で HTTP エラー、または 1 回 20 秒を超えたら、その場で止めて終了コード 3（枠の取り合いの兆し）。
"""
import hashlib, json, os, sys, time, urllib.request, urllib.error
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(os.path.dirname(HERE), "data")
CFG = {
    "bge": dict(url="http://127.0.0.1:8081/v1/embeddings", model="bge-m3", prefix="", min_interval=0.0),
    "sarashina": dict(url="http://127.0.0.1:8082/v1/embeddings", model="sarashina3-embedding", prefix="[[sarashina:query]]", min_interval=3.2),
}
SLOW_STOP = 20.0


def key(model, prefix, text):
    return hashlib.sha1((model + "\x00" + prefix + "\x00" + text).encode("utf-8")).hexdigest()


def load_cache(name):
    p = os.path.join(D, "emb_cache", f"{name}.jsonl"); c = {}
    if os.path.exists(p):
        for l in open(p, encoding="utf-8"):
            r = json.loads(l); c[r["key"]] = r
    return c


def load1():
    return float(open("/proc/loadavg").read().split()[0])


def call(cfg, text):
    body = json.dumps({"model": cfg["model"], "input": cfg["prefix"] + text}).encode()
    req = urllib.request.Request(cfg["url"], data=body, headers={"Content-Type": "application/json"})
    t = time.monotonic()
    with urllib.request.urlopen(req, timeout=90) as r:
        d = json.load(r)
    return d["data"][0]["embedding"], (time.monotonic() - t) * 1000


def main():
    name = sys.argv[1]; cfg = CFG[name]
    os.makedirs(os.path.join(D, "emb_cache"), exist_ok=True)
    qs = [json.loads(l) for l in open(os.path.join(D, "questions.jsonl"), encoding="utf-8")]
    cache = load_cache(name); path = os.path.join(D, "emb_cache", f"{name}.jsonl")
    todo = [q for q in qs if key(cfg["model"], cfg["prefix"], q["query"]) not in cache]
    print(f"{name}: {len(qs)} 問、キャッシュ済み {len(qs)-len(todo)}、これから {len(todo)}", flush=True)
    last = 0.0
    with open(path, "a", encoding="utf-8") as f:
        for n, q in enumerate(todo, 1):
            wait = cfg["min_interval"] - (time.monotonic() - last)
            if wait > 0: time.sleep(wait)
            while name == "bge" and load1() > 5.0:
                time.sleep(5)
            last = time.monotonic()
            try:
                v, ms = call(cfg, q["query"])
            except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
                print(f"STOP {name} {q['qid']}: {e!r}", flush=True); sys.exit(3)
            f.write(json.dumps({"key": key(cfg["model"], cfg["prefix"], q["query"]), "qid": q["qid"], "ms": round(ms, 1), "at": time.time(),
                                "load1": load1(), "dim": len(v), "vec": v}) + "\n"); f.flush()
            if name == "sarashina" and ms > SLOW_STOP * 1000:
                print(f"STOP sarashina slow {ms:.0f}ms at {q['qid']}", flush=True); sys.exit(3)
            if n % 20 == 0: print(f"  {n}/{len(todo)} last {ms:.0f}ms load1 {load1():.2f}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
