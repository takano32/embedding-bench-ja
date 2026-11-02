#!/usr/bin/env python3
"""問い 200 問を組み立てる。入力は data/ の中の私的なファイルだけで、このスクリプトに本文は無い。

- K（既知の1件）: data/questions_k_raw.txt（`バッチ|候補番号|style|問い|答えの要点`）を、
  seed_candidates.json / seed_candidates2.json の候補番号 → 記憶 id に結びつける。
  style は d（元の文と語を共有する言い方）と i（言い換え・遠回し）。書いたのは人（本流）。
- T（実メッセージ）: 本番の受信（takano32 の inbound 文書）から乱数（seed 固定）で引く。
  本文の先頭 500 字を問いにする（本番のフックの MAX_QUERY_CHARS と同じ）。
  次を検索の対象から外す（leave-one-out）: その文書から作られた記憶、それを材料に含む observation、
  同じチャンネルの印を持ち、受信の 5 分前から 30 分後までに作られた記憶（その場の返信や続きの発言。
  本番で引くときには存在しない）。
出力: data/questions.jsonl
"""
import json, os, random, re, sys
from datetime import datetime, timedelta
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(os.path.dirname(HERE), "data")
N_T = int(sys.argv[1]) if len(sys.argv) > 1 else 84
units = [json.loads(l) for l in open(os.path.join(D, "units.jsonl"), encoding="utf-8")]
id2i = {u["id"]: i for i, u in enumerate(units)}
cands = {1: json.load(open(os.path.join(D, "seed_candidates.json"))), 2: json.load(open(os.path.join(D, "seed_candidates2.json")))}
out = []
for line in open(os.path.join(D, "questions_k_raw.txt"), encoding="utf-8"):
    line = line.strip()
    if not line: continue
    b, idx, style, q, ans = line.split("|", 4)
    _, ui = cands[int(b)][int(idx)]
    out.append({"qid": f"K{len(out)+1:03d}", "type": "K", "style": style, "query": q, "answer": ans, "seed_unit": units[ui]["id"],
                "seed_kind": units[ui]["kind"], "seed_fact_type": units[ui]["fact_type"]})
nk = len(out)
inb = [json.loads(l) for l in open(os.path.join(D, "inbound.jsonl"), encoding="utf-8")]
seen = set(); pool = []
for d in inb:
    t = (d["text"] or "").strip()
    if len(t) < 20 or t in seen: continue
    if re.fullmatch(r"[\W\d_a-zA-Z]{0,25}", t) and len(t) < 30: continue
    seen.add(t); pool.append(d)
random.seed(20261102)
pick = random.sample(pool, N_T)
by_doc = {}
for i, u in enumerate(units):
    by_doc.setdefault(u["document_id"], []).append(i)
for d in pick:
    excl = set(by_doc.get(d["id"], []))
    chat = d["id"].split(":")[1] if d["id"].startswith("discord:") else None
    t0 = datetime.fromisoformat(d["created_at"])
    if chat:
        for i, u in enumerate(units):
            if f"chat:{chat}" in u["tags"] and u["fact_type"] != "observation":
                tc = datetime.fromisoformat(u["created_at"])
                if t0 - timedelta(minutes=5) <= tc <= t0 + timedelta(minutes=30):
                    excl.add(i)
    srcs = {units[i]["id"] for i in excl}
    # この文書の記憶を1つでも材料に含む observation も外す
    for i, u in enumerate(units):
        if u["fact_type"] == "observation" and u["source_memory_ids"] and srcs and (set(u["source_memory_ids"]) & srcs):
            excl.add(i)
    out.append({"qid": f"T{len(out)-nk+1:03d}", "type": "T", "style": "m", "query": d["text"].strip()[:500], "doc_id": d["id"],
                "exclude_units": sorted(units[i]["id"] for i in excl)})
with open(os.path.join(D, "questions.jsonl"), "w", encoding="utf-8") as f:
    for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("K", nk, "T", len(out) - nk, "pool", len(pool), "excluded units per T (mean)",
      sum(len(r["exclude_units"]) for r in out if r["type"] == "T") / max(1, len(out) - nk))
