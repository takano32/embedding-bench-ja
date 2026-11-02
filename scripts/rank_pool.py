#!/usr/bin/env python3
"""問いごとに全記憶のスコアを出し、判定するプール（候補）を作る。data/pools.json に書く。

スコア: bge-m3、Sarashina＝問いの埋め込みと記憶の保存済みベクトルのコサイン。lex＝字 bigram の BM25。
leave-one-out（T）の除外は -inf にする。K は seed を必ず候補に入れる。
プール（Jev が判定する）: bge 上位20、Sarashina 上位20、lex 上位10、seed。
人手プール（本流が読んで判定する）: bge 上位8、Sarashina 上位8、lex 上位5、seed（Jev プールの部分集合）。
"""
import json, sys
import numpy as np
from common import *

POOL = dict(sar=20, bge=20, lex=10); HPOOL = dict(sar=8, bge=8, lex=5)


def main():
    units = load_units(); qs = load_questions(); qv, _ = load_qvecs(qs); dv = load_doc_vecs()
    id2i = {u["id"]: i for i, u in enumerate(units)}
    bm = BM25([u["text"] for u in units])
    S = {"sar": qv["sarashina"] @ dv["sarashina"].T, "bge": qv["bge"] @ dv["bge"].T}
    pools = {}
    for qi, q in enumerate(qs):
        sc = {"sar": S["sar"][qi].copy(), "bge": S["bge"][qi].copy(), "lex": bm.scores(q["query"])}
        if q["type"] == "T":
            ex = [id2i[x] for x in q["exclude_units"]]
            for k in sc: sc[k][ex] = -np.inf
        else:
            # 自分の問い文そのものが記憶に入っている場合は無い（問いは新規）。seed は除外しない
            pass
        order = {k: np.argsort(-v, kind="stable") for k, v in sc.items()}
        rank = {k: np.empty(len(units), dtype=np.int32) for k in sc}
        for k in sc: rank[k][order[k]] = np.arange(1, len(units) + 1)
        seed = id2i[q["seed_unit"]] if q["type"] == "K" else None
        def take(spec):
            s = []
            for k, n in spec.items():
                for i in order[k][:n]:
                    if np.isfinite(sc[k][i]) and int(i) not in s: s.append(int(i))
            if seed is not None and seed not in s: s.append(seed)
            return s
        pl, hp = take(POOL), take(HPOOL)
        pools[q["qid"]] = {"pool": pl, "hpool": hp, "seed": seed,
            "cands": {str(i): {"sar": float(sc["sar"][i]), "bge": float(sc["bge"][i]), "lex": float(sc["lex"][i]),
                               "r_sar": int(rank["sar"][i]), "r_bge": int(rank["bge"][i]), "r_lex": int(rank["lex"][i])} for i in pl}}
    json.dump(pools, open(os.path.join(D, "pools.json"), "w"))
    sizes = [len(p["pool"]) for p in pools.values()]; hs = [len(p["hpool"]) for p in pools.values()]
    print("pool size mean", np.mean(sizes), "min", min(sizes), "max", max(sizes), "| hpool mean", np.mean(hs))


if __name__ == "__main__":
    main()
