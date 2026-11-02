#!/usr/bin/env python3
"""集計。入力は data/ の私的なファイル、出力は results/summary.json と results/summary.md（数だけ。本文は入らない）。

ラベル: J＝Jev（run1・run2 の p の平均 >= 0.5。K は seed を関連ありに固定）。H＝本流が読んで付けた（32 問、人手プール内）。
指標は上位 k（k<=20）が全部ラベル付きになる範囲で計算する（プール＝各モデルの上位 20）。
信頼区間は問いでのブートストラップ（10,000 回、seed 固定）。差は同じ再標本での Sarashina − bge。
"""
import json, os, sys
import numpy as np
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
from common import *

RNG = np.random.default_rng(20261102); B = 10000
KS = (1, 3, 5, 10)


def boot_ci(x, stat=np.mean):
    x = np.asarray(x, dtype=float); n = len(x)
    if n == 0: return (float("nan"),) * 3
    idx = RNG.integers(0, n, (B, n)); m = stat(x[idx], axis=1)
    return float(stat(x)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_diff(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float); n = len(a)
    idx = RNG.integers(0, n, (B, n)); d = (a[idx] - b[idx]).mean(1)
    return float((a - b).mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def kappa(a, b):
    a = np.asarray(a, bool); b = np.asarray(b, bool); po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def auc(pos, neg):
    """Mann-Whitney の AUC（同値は平均順位）。pos が neg より高い確率。"""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0: return None
    allv = np.concatenate([pos, neg]); order = allv.argsort(kind="mergesort"); ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    uniq, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        sums = np.bincount(inv, weights=ranks); ranks = (sums / cnt)[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    units = load_units(); qs = load_questions(); qv, lat = load_qvecs(qs); dv = load_doc_vecs()
    id2i = {u["id"]: i for i, u in enumerate(units)}; pools = json.load(open(os.path.join(D, "pools.json")))
    jev = {}
    for l in open(os.path.join(D, "jev_labels.jsonl"), encoding="utf-8"):
        r = json.loads(l); jev.setdefault(r["qid"], {})[r["run"]] = r["p"]
    hl = json.load(open(os.path.join(D, "human_labels.json"))) if os.path.exists(os.path.join(D, "human_labels.json")) else {}
    S = {"sar": qv["sarashina"] @ dv["sarashina"].T, "bge": qv["bge"] @ dv["bge"].T}
    out = {"n_units": len(units), "n_questions": len(qs)}
    # --- ラベルの組み立て
    labJ, pJ, labH = {}, {}, {}
    for q in qs:
        qid = q["qid"]; runs = jev.get(qid, {})
        if not runs: continue
        pm = {}
        for i in pools[qid]["pool"]:
            ps = [runs[r][str(i)] for r in runs if str(i) in runs[r]]; pm[i] = float(np.mean(ps))
        if q["type"] == "K": pm[pools[qid]["seed"]] = max(pm[pools[qid]["seed"]], 1.0)
        pJ[qid] = pm; labJ[qid] = {i for i, p in pm.items() if p >= 0.5}
    for qid, h in hl.items():
        rel = set(h["rel"])
        if qs[[q["qid"] for q in qs].index(qid)]["type"] == "K": rel.add(pools[qid]["seed"])
        labH[qid] = (set(h["judged"]), rel)
    out["jev_questions"] = len(labJ); out["human_questions"] = len(labH)
    # --- Jev の自己一致（run1 vs run2）
    a, b = [], []
    for qid, runs in jev.items():
        if 1 in runs and 2 in runs:
            for k in runs[1]:
                if k in runs[2]: a.append(runs[1][k] >= 0.5); b.append(runs[2][k] >= 0.5)
    if a: out["jev_self_agreement"] = {"pairs": len(a), "agree": float(np.mean(np.array(a) == np.array(b))), "kappa": kappa(a, b)}
    # --- Jev × 人手
    agree = {}
    for tp in ("K", "T", "all"):
        ja, ha = [], []
        for qid, (judged, rel) in labH.items():
            if tp != "all" and not qid.startswith(tp): continue
            for i in judged:
                if i == pools[qid]["seed"]: continue          # seed は両方とも関連ありに固定なので除く
                ja.append(i in labJ[qid]); ha.append(i in rel)
        if ja:
            ja = np.array(ja); ha = np.array(ha)
            tp_, fp_, fn_, tn_ = int((ja & ha).sum()), int((ja & ~ha).sum()), int((~ja & ha).sum()), int((~ja & ~ha).sum())
            agree[tp] = {"pairs": len(ja), "agree": float((ja == ha).mean()), "kappa": kappa(ja, ha), "both_rel": tp_, "jev_only_rel": fp_, "human_only_rel": fn_, "both_nonrel": tn_,
                         "human_rel_rate": float(ha.mean()), "jev_rel_rate": float(ja.mean())}
    out["jev_vs_human"] = agree
    # --- ランキング指標
    def ranks_for(qi, m):
        return np.argsort(-S[m][qi], kind="stable")
    excl = {q["qid"]: [id2i[x] for x in q.get("exclude_units", [])] for q in qs}
    def per_q_metrics(label_fn, qsel, m, depth=20):
        rows = {k: [] for k in ("top1", "hit3", "hit5", "hit10", "mrr", "p5", "p10", "n_rel_pool")}
        seedrows = {k: [] for k in ("seed1", "seed3", "seed5", "seed10", "seed20", "seed_mrr", "seed_rank")}
        for qi, q in enumerate(qs):
            if q["qid"] not in qsel: continue
            rel = label_fn(q["qid"])
            if rel is None: continue
            s = S[m][qi].copy(); s[excl[q["qid"]]] = -np.inf
            o = np.argsort(-s, kind="stable")[:depth]
            isrel = np.array([int(i) in rel for i in o])
            if isrel.sum() == 0 and not rel: continue
            rows["n_rel_pool"].append(len(rel))
            rows["top1"].append(float(isrel[0])); rows["hit3"].append(float(isrel[:3].any())); rows["hit5"].append(float(isrel[:5].any())); rows["hit10"].append(float(isrel[:10].any()))
            first = np.argmax(isrel) + 1 if isrel.any() else None
            rows["mrr"].append(1.0 / first if first else 0.0); rows["p5"].append(float(isrel[:5].mean())); rows["p10"].append(float(isrel[:10].mean()))
            if q["type"] == "K":
                sd = pools[q["qid"]]["seed"]; rk = int((s > s[sd]).sum()) + 1
                for k, kk in (("seed1", 1), ("seed3", 3), ("seed5", 5), ("seed10", 10), ("seed20", 20)): seedrows[k].append(float(rk <= kk))
                seedrows["seed_mrr"].append(1.0 / rk if rk <= 20 else 0.0); seedrows["seed_rank"].append(rk)
        return rows, seedrows
    def labfn_J(qid): return labJ.get(qid)
    def labfn_H(qid): return labH[qid][1] if qid in labH else None
    subsets = {"all": lambda q: True, "K": lambda q: q["type"] == "K", "K_direct": lambda q: q["type"] == "K" and q["style"] == "d",
               "K_indirect": lambda q: q["type"] == "K" and q["style"] == "i", "T": lambda q: q["type"] == "T"}
    res = {}
    for sname, f in subsets.items():
        qsel = {q["qid"] for q in qs if f(q) and q["qid"] in labJ}
        entry = {"n": len(qsel)}
        mm = {}
        for m in ("sar", "bge"):
            rows, srows = per_q_metrics(labfn_J, qsel, m); mm[m] = (rows, srows)
        # 同じ問いの集合（両方に値がある）で比べる
        for m in ("sar", "bge"):
            rows, srows = mm[m]
            entry[m] = {k: dict(zip(("mean", "lo", "hi"), boot_ci(v))) for k, v in rows.items() if k != "n_rel_pool" and v}
            entry[m]["n_scored"] = len(rows["top1"]); entry[m]["rel_per_q_median"] = float(np.median(rows["n_rel_pool"])) if rows["n_rel_pool"] else None
            if srows["seed1"]:
                entry[m]["seed"] = {k: dict(zip(("mean", "lo", "hi"), boot_ci(v))) for k, v in srows.items() if k != "seed_rank"}
                entry[m]["seed_rank_median"] = float(np.median(srows["seed_rank"]))
        entry["diff_sar_minus_bge"] = {k: dict(zip(("mean", "lo", "hi"), boot_diff(mm["sar"][0][k], mm["bge"][0][k]))) for k in mm["sar"][0] if k != "n_rel_pool" and mm["sar"][0][k]}
        if mm["sar"][1]["seed1"]:
            entry["diff_seed"] = {k: dict(zip(("mean", "lo", "hi"), boot_diff(mm["sar"][1][k], mm["bge"][1][k]))) for k in mm["sar"][1] if k != "seed_rank"}
        res[sname] = entry
    out["metrics_jev"] = res
    # 人手ラベル（k<=5 は人手プールで全部ラベル付き）
    hres = {}
    qselH = set(labH)
    for sname, f in (("all", lambda q: True), ("K", lambda q: q["type"] == "K"), ("T", lambda q: q["type"] == "T")):
        qsel = {q["qid"] for q in qs if f(q) and q["qid"] in qselH}
        entry = {"n": len(qsel)}; mm = {}
        for m in ("sar", "bge"):
            rows, srows = per_q_metrics(labfn_H, qsel, m, depth=5); mm[m] = rows
            entry[m] = {k: dict(zip(("mean", "lo", "hi"), boot_ci(v))) for k, v in rows.items() if k in ("top1", "hit3", "hit5", "p5") and v}
        entry["diff_sar_minus_bge"] = {k: dict(zip(("mean", "lo", "hi"), boot_diff(mm["sar"][k], mm["bge"][k]))) for k in ("top1", "hit3", "hit5", "p5") if mm["sar"][k]}
        # 同じ問いを Jev ラベルで（depth=5）
        jm = {}
        for m in ("sar", "bge"):
            rows, _ = per_q_metrics(labfn_J, qsel, m, depth=5); jm[m] = rows
            entry[m + "_with_jev_labels"] = {k: dict(zip(("mean", "lo", "hi"), boot_ci(v))) for k, v in rows.items() if k in ("top1", "hit3", "hit5", "p5") and v}
        entry["diff_with_jev_labels"] = {k: dict(zip(("mean", "lo", "hi"), boot_diff(jm["sar"][k], jm["bge"][k]))) for k in ("top1", "hit3", "hit5", "p5") if jm["sar"][k]}
        hres[sname] = entry
    out["metrics_human_subset"] = hres
    # --- 上位 k の重なり（モデル間）
    ov = []
    for qi, q in enumerate(qs):
        s1 = S["sar"][qi].copy(); s2 = S["bge"][qi].copy(); s1[excl[q["qid"]]] = -np.inf; s2[excl[q["qid"]]] = -np.inf
        ov.append(len(set(np.argsort(-s1)[:10]) & set(np.argsort(-s2)[:10])))
    out["top10_overlap_between_models"] = {"mean": float(np.mean(ov)), "min": int(min(ov)), "max": int(max(ov))}
    # --- 閾値分析
    deep = {}
    dp_path = os.path.join(D, "jev_deep.jsonl")
    if os.path.exists(dp_path):
        for l in open(dp_path, encoding="utf-8"):
            r = json.loads(l); deep[r["qid"]] = {int(k): v for k, v in r["p"].items()}
    labD = {}
    for qid, pm in deep.items():
        merged = dict(pm)
        merged.update(pJ.get(qid, {}))           # プール内は run1・run2 の平均を優先
        if qid.startswith("K") and pools[qid]["seed"] is not None: merged[pools[qid]["seed"]] = max(merged.get(pools[qid]["seed"], 0), 1.0)
        labD[qid] = {i for i, p in merged.items() if p >= 0.5}; deep[qid] = merged
    def collect(src, types):
        rows = []  # (qid, type, unit, rel, cos_sar, cos_bge, is_seed)
        for qi, q in enumerate(qs):
            qid = q["qid"]
            if q["type"] not in types: continue
            if src == "J":
                if qid not in labJ: continue
                units_, relset = pools[qid]["pool"], labJ[qid]
            elif src == "Jdeep":
                if qid not in labD: continue
                units_, relset = list(deep[qid]), labD[qid]
            else:
                if qid not in labH: continue
                units_, relset = labH[qid][0], labH[qid][1]
            for i in units_: rows.append((qid, q["type"], i, i in relset, S["sar"][qi][i], S["bge"][qi][i], i == pools[qid]["seed"]))
        return rows
    def qtile(x):
        x = np.asarray(x); return {k: float(np.percentile(x, p)) for k, p in (("min", 0), ("p5", 5), ("p25", 25), ("median", 50), ("p75", 75), ("p95", 95), ("max", 100))} if len(x) else None
    thr = {}
    grids = {"sar": [round(0.60 + 0.02 * i, 2) for i in range(0, 14)], "bge": [round(0.30 + 0.02 * i, 2) for i in range(0, 17)]}
    for src, types, name in (("Jdeep", "KT", "jev_deep_all"), ("Jdeep", "K", "jev_deep_K"), ("Jdeep", "T", "jev_deep_T"), ("J", "KT", "jev_pool_all"), ("H", "KT", "human_all")):
        rows = collect(src, types)
        if not rows: continue
        qids = sorted({r[0] for r in rows})
        block = {"pairs": len(rows), "relevant": int(sum(r[3] for r in rows)), "questions": len(qids)}
        byq = {}
        for r in rows: byq.setdefault(r[0], []).append(r)
        for m, col in (("sar", 4), ("bge", 5)):
            rel = [r[col] for r in rows if r[3]]; non = [r[col] for r in rows if not r[3]]
            block[m + "_cos_relevant"] = qtile(rel); block[m + "_cos_irrelevant"] = qtile(non); block[m + "_auc_pooled"] = auc(rel, non)
            per = []
            for qid in qids:
                p_ = [r[col] for r in byq[qid] if r[3]]; n_ = [r[col] for r in byq[qid] if not r[3]]
                a_ = auc(p_, n_)
                if a_ is not None: per.append(a_)
            block[m + "_auc_within_question"] = dict(zip(("mean", "lo", "hi"), boot_ci(per))) | {"n": len(per)}
            tab = []
            for t in grids[m]:
                loss_q = allgone_q = nrelq = seed_n = seed_pass = relp = reltot = nonrel_tot = nonrel_pass = 0; passed_nonrel = []; passed = []
                for qid in qids:
                    rr = byq[qid]; rl = [r for r in rr if r[3]]; nr = [r for r in rr if not r[3]]
                    if rl:
                        nrelq += 1; ps = [r for r in rl if r[col] >= t]; relp += len(ps); reltot += len(rl)
                        loss_q += int(len(ps) < len(rl)); allgone_q += int(not ps)
                    sd = [r for r in rr if r[6]]
                    if sd: seed_n += 1; seed_pass += int(sd[0][col] >= t)
                    pn = [r for r in nr if r[col] >= t]; passed_nonrel.append(len(pn)); nonrel_tot += len(nr); nonrel_pass += len(pn)
                    passed.append(len([r for r in rr if r[col] >= t]))
                corpus_pass = float(np.mean([(S[m][qi] >= t).sum() for qi, q in enumerate(qs) if q["qid"] in byq]))
                tab.append({"t": t, "recall_pairs": relp / reltot if reltot else None, "questions_with_a_relevant_dropped": loss_q / nrelq if nrelq else None,
                            "questions_with_all_relevant_dropped": allgone_q / nrelq if nrelq else None,
                            "seed_pass_rate": seed_pass / seed_n if seed_n else None, "irrelevant_passed_per_q": float(np.mean(passed_nonrel)),
                            "irrelevant_pass_rate": nonrel_pass / nonrel_tot if nonrel_tot else None,
                            "precision_pairs": (relp / (relp + nonrel_pass)) if (relp + nonrel_pass) else None,
                            "pool_passed_per_q": float(np.mean(passed)), "corpus_passed_per_q": corpus_pass, "n_relevant_questions": nrelq, "seed_questions": seed_n})
            block[m + "_threshold_table"] = tab
            # コサインの帯ごとの関連ありの割合
            edges = np.arange(0.50, 0.92, 0.02) if m == "sar" else np.arange(0.25, 0.80, 0.05)
            bands = []
            for lo in edges:
                hi = lo + (0.02 if m == "sar" else 0.05)
                sel = [r for r in rows if lo <= r[col] < hi]
                if len(sel) >= 30: bands.append({"lo": round(float(lo), 2), "hi": round(float(hi), 2), "n": len(sel), "relevant_rate": float(np.mean([r[3] for r in sel]))})
            block[m + "_relevance_by_band"] = bands
        thr[name] = block
    out["thresholds"] = thr
    # --- 足切りが「引いた上位」に実際どう効くか（上位60＝本番の recall が返す数に近い深さ、上位12＝Jev が使えないときの戻り先）
    def wilson(k, n, z=1.96):
        if n == 0: return None
        ph = k / n; den = 1 + z * z / n; c = ph + z * z / (2 * n); r = z * ((ph * (1 - ph) / n + z * z / (4 * n * n)) ** 0.5)
        return [float((c - r) / den), float((c + r) / den)]
    fe = {}
    for m, ts in (("sar", (0.70, 0.72, 0.74, 0.76, 0.78)), ("bge", (0.35, 0.45))):
        for t in ts:
            cnt60, below60, short12, n12, rel12_no, rel12_floor, seedin12_no, seedin12_floor, nseed, hit_no, hit_fl, nq = [], [], 0, [], [], [], 0, 0, 0, 0, 0, 0
            for qi, q in enumerate(qs):
                qid = q["qid"]
                if qid not in labD: continue
                sc = S[m][qi].copy(); sc[excl[qid]] = -np.inf
                o = np.argsort(-sc, kind="stable")
                t60, t12 = o[:60], o[:12]
                cnt60.append(int((sc[t60] >= t).sum())); below60.append(int((sc[t60] < t).sum()))
                kept = [i for i in t12 if sc[i] >= t]; n12.append(len(kept)); short12 += int(len(kept) < 12)
                nq += 1
                r_no = [int(i in labD[qid]) for i in t12]; r_fl = [int(i in labD[qid]) for i in kept]
                rel12_no.append(sum(r_no) / 12); rel12_floor.append(sum(r_fl) / len(r_fl) if r_fl else 0.0)
                hit_no += int(any(r_no)); hit_fl += int(any(r_fl))
                if q["type"] == "K":
                    nseed += 1; sd = pools[qid]["seed"]; seedin12_no += int(sd in set(map(int, t12))); seedin12_floor += int(sd in set(map(int, kept)))
            fe[f"{m}_{t:.2f}"] = {"questions": nq, "top60_passing_mean": float(np.mean(cnt60)), "top60_below_mean": float(np.mean(below60)),
                                  "questions_with_any_top60_below": float(np.mean([b > 0 for b in below60])),
                                  "top12_returned_mean": float(np.mean(n12)), "questions_top12_short": short12 / nq,
                                  "top12_precision_no_floor": float(np.mean(rel12_no)), "top12_precision_with_floor": float(np.mean(rel12_floor)),
                                  "top12_hit_no_floor": hit_no / nq, "top12_hit_with_floor": hit_fl / nq,
                                  "K_seed_in_top12_no_floor": seedin12_no / nseed if nseed else None, "K_seed_in_top12_with_floor": seedin12_floor / nseed if nseed else None,
                                  "K_seed_questions": nseed}
    out["floor_effect"] = fe
    ks = {}
    for t in (0.70, 0.74):
        n = sum(1 for q in qs if q["type"] == "K"); k = int(round(next(r["seed_pass_rate"] for r in thr["jev_deep_K"]["sar_threshold_table"] if r["t"] == t) * n))
        ks[f"{t:.2f}"] = {"passed": k, "of": n, "wilson95": wilson(k, n)}
    out["seed_pass_wilson"] = ks
    # --- 全記憶のコサイン分布（問い×全記憶。除外を除く）
    cd = {}
    for m in ("sar", "bge"):
        vals = np.concatenate([np.delete(S[m][qi], excl[q["qid"]]) for qi, q in enumerate(qs)]); cd[m] = qtile(vals)
        tops = np.array([[np.sort(np.delete(S[m][qi], excl[q["qid"]]))[::-1][k - 1] for k in (1, 5, 10)] for qi, q in enumerate(qs)])
        cd[m + "_top_k_mean"] = {"top1": float(tops[:, 0].mean()), "top5": float(tops[:, 1].mean()), "top10": float(tops[:, 2].mean()),
                                  "top1_min": float(tops[:, 0].min()), "top1_max": float(tops[:, 0].max())}
    out["cos_all_pairs"] = cd
    # --- 遅延
    def lq(x): x = np.asarray(x); return {"n": len(x), "median_ms": float(np.median(x)), "mean_ms": float(x.mean()), "p90_ms": float(np.percentile(x, 90)), "p99_ms": float(np.percentile(x, 99)), "max_ms": float(x.max()), "min_ms": float(x.min())}
    out["latency_query"] = {"bge": lq(lat["bge"]), "sarashina": lq(lat["sarashina"])}
    p = os.path.join(D, "doc_check.json")
    if os.path.exists(p):
        dc = json.load(open(p))
        out["doc_check"] = {"n": len(dc), "bge_latency": lq([r["bge_ms"] for r in dc]), "sarashina_latency": lq([r["sar_ms"] for r in dc]),
                            "bge_cos_stored": qtile([r["bge_cos_stored"] for r in dc]), "sarashina_cos_stored": qtile([r["sar_cos_stored"] for r in dc]),
                            "chars_median": float(np.median([r["chars"] for r in dc]))}
    json.dump(out, open(os.path.join(R, "summary.json"), "w"), ensure_ascii=False, indent=1)
    print("wrote results/summary.json")


if __name__ == "__main__":
    main()
