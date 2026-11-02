#!/usr/bin/env python3
"""Hindsight の共有 PostgreSQL から nilkani06 bank の記憶を読み出して data/ に置く。SELECT だけ。

- 接続は `docker exec postgresql psql`（PGOPTIONS で default_transaction_read_only=on を強制）。
- 出すもの（すべて data/、.gitignore 済み）
    units.jsonl      記憶1件ごとの本文・種別・出所の印・日付（ベクトルを除く）
    vec_sarashina.npy / vec_bge.npy   記憶と同じ順の埋め込み（float32、L2 正規化済み）
    units_ids.json   上の順の id
    inbound.jsonl    takano32 の受信文書（document の original_text）。問い作りの材料
- Sarashina のベクトル＝memory_units.embedding（本番がいま使っている列）。
- bge-m3 のベクトル＝memory_units_standby の model='bge-m3'（本番の控え）。両方持つ行だけ残す。
"""
import json, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
os.makedirs(DATA, exist_ok=True)
BANK = "nilkani06"


def psql(sql: str):
    p = subprocess.Popen(
        ["docker", "exec", "-i", "-e", "PGOPTIONS=-c default_transaction_read_only=on -c statement_timeout=300000",
         "postgresql", "psql", "-U", "postgres", "-d", "hindsight", "-X", "-A", "-t", "-v", "ON_ERROR_STOP=1", "-c", sql],
        stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
    for line in p.stdout:
        yield line.rstrip("\n")
    if p.wait() != 0:
        sys.exit("psql failed")


def parse_vec(s: str) -> np.ndarray:
    return np.fromstring(s.strip("[]"), sep=",", dtype=np.float32)


def main():
    assert "bank" not in sys.argv  # bank は固定。zerozari00 は読まない
    q = f"""
    select json_build_object('id', m.id, 'fact_type', m.fact_type, 'document_id', m.document_id, 'text', m.text,
      'context', m.context, 'occurred_start', m.occurred_start, 'mentioned_at', m.mentioned_at, 'created_at', m.created_at,
      'tags', m.tags, 'source_memory_ids', m.source_memory_ids, 'kind', m.metadata->>'kind', 'speaker', m.metadata->>'speaker')::text
      || E'\\t' || m.embedding::text || E'\\t' || s.embedding::text
    from memory_units m join memory_units_standby s on s.unit_id = m.id and s.model = 'bge-m3'
    where m.bank_id = '{BANK}' and m.embedding is not null order by m.created_at, m.id
    """
    rows, sar, bge = [], [], []
    for line in psql(q):
        j, a, b = line.split("\t")
        rows.append(json.loads(j)); sar.append(parse_vec(a)); bge.append(parse_vec(b))
    sar = np.stack(sar); bge = np.stack(bge)
    sar /= np.linalg.norm(sar, axis=1, keepdims=True); bge /= np.linalg.norm(bge, axis=1, keepdims=True)
    np.save(os.path.join(DATA, "vec_sarashina.npy"), sar); np.save(os.path.join(DATA, "vec_bge.npy"), bge)
    with open(os.path.join(DATA, "units.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump([r["id"] for r in rows], open(os.path.join(DATA, "units_ids.json"), "w"))
    print("units", len(rows), "sar", sar.shape, "bge", bge.shape)

    q2 = f"""select json_build_object('id', d.id, 'created_at', d.created_at, 'text', d.original_text,
        'speaker', d.retain_params->'metadata'->>'speaker', 'kind', d.retain_params->'metadata'->>'kind')::text
        from documents d where d.bank_id = '{BANK}' and d.retain_params->'metadata'->>'kind' = 'inbound'
        and d.retain_params->'metadata'->>'speaker' = 'takano32' order by d.created_at"""
    n = 0
    with open(os.path.join(DATA, "inbound.jsonl"), "w", encoding="utf-8") as f:
        for line in psql(q2):
            f.write(line + "\n"); n += 1
    print("inbound", n)


if __name__ == "__main__":
    main()
