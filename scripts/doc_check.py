#!/usr/bin/env python3
"""保存済みの記憶ベクトルが「いま同じ文を埋め込んだもの」と合うかの抜き取り検査と、文書側の埋め込み遅延の測定。

標本 N 件（既定 24。fact 16＋observation 8、seed 固定）の文を、本番 retain と同じ形（日付・実体名つき。
~/opt/hindsight-docker/standby/common.py の augment と同じ式）に組み立て、2 モデルで埋め込んで、
保存済みのベクトルとのコサインと所要時間を data/doc_check.json に書く。Sarashina は 20 回/分以下（3.2 秒間隔）。
実体名は SELECT のみ（read-only）で読む。
"""
import json, os, random, subprocess, sys, time, urllib.request
from datetime import datetime
import numpy as np
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
from common import *

N = int(sys.argv[1]) if len(sys.argv) > 1 else 24


def psql(sql):
    r = subprocess.run(["docker", "exec", "-i", "-e", "PGOPTIONS=-c default_transaction_read_only=on", "postgresql", "psql", "-U", "postgres",
                        "-d", "hindsight", "-X", "-A", "-t", "-v", "ON_ERROR_STOP=1", "-c", sql], capture_output=True, text=True, check=True)
    return r.stdout.splitlines()


def readable(dt): return f"{dt.strftime('%B')} {dt.strftime('%Y')}"


def order_entities(names, signals):
    if not names: return []
    if not signals: return sorted(names)
    def key(n):
        p = signals.find(n); return (p if p >= 0 else 10**9, n)
    return sorted(names, key=key)


def call(url, model, text):
    req = urllib.request.Request(url, data=json.dumps({"model": model, "input": text}).encode(), headers={"Content-Type": "application/json"})
    t = time.monotonic()
    with urllib.request.urlopen(req, timeout=120) as r: d = json.load(r)
    return np.array(d["data"][0]["embedding"], dtype=np.float32), (time.monotonic() - t) * 1000


def main():
    units = load_units(); dv = load_doc_vecs(); rnd = random.Random(20261102)
    facts = [i for i, u in enumerate(units) if u["fact_type"] != "observation"]; obs = [i for i, u in enumerate(units) if u["fact_type"] == "observation"]
    pick = rnd.sample(facts, N * 2 // 3) + rnd.sample(obs, N - N * 2 // 3)
    ids = "','".join(units[i]["id"] for i in pick)
    sig = {}
    for l in psql(f"select json_build_object('id', m.id, 'sig', m.text_signals, 'os', m.occurred_start, 'oe', m.occurred_end, 'ma', m.mentioned_at, "
                  f"'ents', coalesce((select json_agg(e.canonical_name) from unit_entities ue join entities e on e.id=ue.entity_id where ue.unit_id=m.id), '[]'::json))::text "
                  f"from memory_units m where m.id in ('{ids}')"):
        j = json.loads(l); sig[j["id"]] = j
    def dt(x): return datetime.fromisoformat(x) if x else None
    rows = []; last = 0.0
    for i in pick:
        u = units[i]; j = sig[u["id"]]
        if u["fact_type"] == "observation": t = u["text"]
        else:
            os_, oe, ma = dt(j["os"]), dt(j["oe"]), dt(j["ma"]); fd = os_ or ma
            t = u["text"]
            if fd:
                t = f"{t} (happened from {readable(fd)} to {readable(oe)})" if (oe and oe != os_) else f"{t} (happened in {readable(fd)})"
            ents = order_entities(j["ents"], j["sig"])
            if ents: t = f"{t} [{', '.join(ents)}]"
        row = {"fact_type": u["fact_type"], "chars": len(t)}
        v, ms = call("http://127.0.0.1:8081/v1/embeddings", "bge-m3", t); v /= np.linalg.norm(v)
        row["bge_ms"] = round(ms, 1); row["bge_cos_stored"] = float(v @ dv["bge"][i])
        w = 3.2 - (time.monotonic() - last)
        if w > 0: time.sleep(w)
        last = time.monotonic()
        v, ms = call("http://127.0.0.1:8082/v1/embeddings", "sarashina3-embedding", "[[sarashina:document]]" + t); v /= np.linalg.norm(v)
        row["sar_ms"] = round(ms, 1); row["sar_cos_stored"] = float(v @ dv["sarashina"][i])
        rows.append(row); print(len(rows), row, flush=True)
    json.dump(rows, open(os.path.join(D, "doc_check.json"), "w"))


if __name__ == "__main__":
    main()
