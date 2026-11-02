#!/usr/bin/env python3
"""results/summary.json から results/summary.md（集計の表）を作る。本文は入らない（数だけ）。"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(os.path.dirname(HERE), "results")
s = json.load(open(os.path.join(R, "summary.json")))
L = []
def ci(d, pct=True, nd=3):
    f = (lambda x: f"{x*100:.1f}") if pct else (lambda x: f"{x:.{nd}f}")
    return f"{f(d['mean'])} [{f(d['lo'])}, {f(d['hi'])}]"
def diff(d): return f"{d['mean']*100:+.1f} [{d['lo']*100:+.1f}, {d['hi']*100:+.1f}]"
def p(x): return "-" if x is None else f"{x:.3f}"

L += ["# 集計（本文を含まない）", "",
      f"記憶 {s['n_units']} 件（bank nilkani06）、問い {s['n_questions']} 問（K 116・T 84）。Jev が判定した問い {s['jev_questions']}、本流が判定した問い {s['human_questions']}。",
      "指標は %。角括弧はブートストラップ 95% 信頼区間。差は Sarashina − bge-m3（同じ再標本）。", ""]
L += ["## 1. 正解が上位に来るか（Jev のラベル。上位20まで全部ラベル付き）", "",
      "| 問いの集合 | 指標 | Sarashina | bge-m3 | 差 |", "|---|---|---|---|---|"]
names = {"all": "全 200 問", "K": "K 116 問", "K_direct": "K 元の文と語を共有 59", "K_indirect": "K 言い換え 57", "T": "T 実メッセージ 84"}
labels = {"top1": "正解1位率", "hit3": "上位3に正解あり", "hit5": "上位5に正解あり", "hit10": "上位10に正解あり", "mrr": "MRR（上位20）", "p5": "上位5の適合率", "p10": "上位10の適合率"}
for sub, e in s["metrics_jev"].items():
    for k, lab in labels.items():
        if k in e["sar"]:
            L.append(f"| {names[sub]}（n={e['sar']['n_scored']}） | {lab} | {ci(e['sar'][k])} | {ci(e['bge'][k])} | {diff(e['diff_sar_minus_bge'][k])} |")
L += ["", "### K の seed（問いの元にした記憶そのもの）の順位", "", "| 問いの集合 | 指標 | Sarashina | bge-m3 | 差 |", "|---|---|---|---|---|"]
sl = {"seed1": "seed が1位", "seed3": "seed が上位3", "seed5": "seed が上位5", "seed10": "seed が上位10", "seed20": "seed が上位20", "seed_mrr": "seed の MRR（上位20）"}
for sub in ("K", "K_direct", "K_indirect"):
    e = s["metrics_jev"][sub]
    for k, lab in sl.items():
        L.append(f"| {names[sub]} | {lab} | {ci(e['sar']['seed'][k])} | {ci(e['bge']['seed'][k])} | {diff(e['diff_seed'][k])} |")
L += ["", "seed が1位にならない問いが半分あるのは、同じ事実を述べる別の記憶（返信と受信、observation）が上に来るため。seed が別の関連ありに抜かれても、正解1位率の側では正解になる。", ""]
L += ["## 2. 人手のラベル（本流が読んで付けた 32 問、k<=5 だけ）", "",
      "| 集合 | 指標 | Sarashina | bge-m3 | 差 | 同じ問いを Jev ラベルで: 差 |", "|---|---|---|---|---|---|"]
for sub, e in s["metrics_human_subset"].items():
    for k in ("top1", "hit3", "hit5", "p5"):
        if k in e["sar"]:
            L.append(f"| {sub}（n={e['n']}） | {labels.get(k,k)} | {ci(e['sar'][k])} | {ci(e['bge'][k])} | {diff(e['diff_sar_minus_bge'][k])} | {diff(e['diff_with_jev_labels'][k])} |")
L += ["", "## 3. Jev と本流の一致", ""]
jh = s["jev_vs_human"]
L += ["| 集合 | 比べた組 | 一致率 | κ | 両方関連あり | Jev だけ関連あり | 本流だけ関連あり | 両方関連なし | 本流の関連あり率 | Jev の関連あり率 |", "|---|---|---|---|---|---|---|---|---|---|"]
for tp in ("K", "T", "all"):
    if tp in jh:
        a = jh[tp]; L.append(f"| {tp} | {a['pairs']} | {a['agree']*100:.1f}% | {a['kappa']:.2f} | {a['both_rel']} | {a['jev_only_rel']} | {a['human_only_rel']} | {a['both_nonrel']} | {a['human_rel_rate']*100:.1f}% | {a['jev_rel_rate']*100:.1f}% |")
sa = s["jev_self_agreement"]
L += ["", f"Jev の自己一致（候補の並びを変えた 2 回、全プール {sa['pairs']} 組）: 一致 {sa['agree']*100:.1f}%、κ {sa['kappa']:.2f}。", ""]
L += ["## 4. 遅延", "", "| | n | 中央値 ms | 平均 ms | p90 ms | p99 ms | 最大 ms |", "|---|---|---|---|---|---|---|"]
for nm, d in (("問いの埋め込み bge-m3（llama-embed、直列、CPU）", s["latency_query"]["bge"]), ("問いの埋め込み Sarashina（変換サーバー経由、20回/分）", s["latency_query"]["sarashina"])):
    L.append(f"| {nm} | {d['n']} | {d['median_ms']:.0f} | {d['mean_ms']:.0f} | {d['p90_ms']:.0f} | {d['p99_ms']:.0f} | {d['max_ms']:.0f} |")
if "doc_check" in s:
    dc = s["doc_check"]
    for nm, d in (("文書の埋め込み bge-m3（中央値 %d 字）" % dc["chars_median"], dc["bge_latency"]), ("文書の埋め込み Sarashina", dc["sarashina_latency"])):
        L.append(f"| {nm} | {d['n']} | {d['median_ms']:.0f} | {d['mean_ms']:.0f} | {d['p90_ms']:.0f} | {d['p99_ms']:.0f} | {d['max_ms']:.0f} |")
    L += ["", f"保存済みベクトルの抜き取り検査（{dc['n']} 件、本番と同じ文を撃ち直してコサイン）: Sarashina は最小 {dc['sarashina_cos_stored']['min']:.4f}、bge-m3 は最小 {dc['bge_cos_stored']['min']:.4f}（中央値 {dc['bge_cos_stored']['median']:.5f}）。"]
L += ["", "Sarashina は 429 も 20 秒超の遅延も出ず、止める条件には当たらなかった（最大 %.0f ms）。" % s["latency_query"]["sarashina"]["max_ms"], ""]
L += ["## 5. コサイン類似度の分布", ""]
c = s["cos_all_pairs"]
L += ["全問い×全記憶（除外を除く）の分位点:", "", "| | min | p5 | p25 | 中央値 | p75 | p95 | max | 1位の平均 | 5位の平均 | 10位の平均 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for m, nm in (("sar", "Sarashina"), ("bge", "bge-m3")):
    d = c[m]; t = c[m + "_top_k_mean"]
    L.append(f"| {nm} | {d['min']:.3f} | {d['p5']:.3f} | {d['p25']:.3f} | {d['median']:.3f} | {d['p75']:.3f} | {d['p95']:.3f} | {d['max']:.3f} | {t['top1']:.3f} | {t['top5']:.3f} | {t['top10']:.3f} |")
for key, title in (("jev_deep_all", "深いプール（各モデル上位100＋BM25上位10＋seed、約160件/問）、Jev ラベル、全問い"), ("jev_deep_K", "同、K のみ"), ("jev_deep_T", "同、T のみ"), ("human_all", "本流のラベル（人手プール 32 問）")):
    if key not in s["thresholds"]: continue
    b = s["thresholds"][key]
    L += ["", f"### {title}（{b['pairs']} 組、関連あり {b['relevant']}、問い {b['questions']}）", "", "| | | min | p5 | p25 | 中央値 | p75 | p95 | max | AUC（全組） | AUC（問いの中の平均） |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m, nm in (("sar", "Sarashina"), ("bge", "bge-m3")):
        for kind, lab in (("relevant", "関連あり"), ("irrelevant", "関連なし")):
            d = b[f"{m}_cos_{kind}"]
            extra = f"{b[m+'_auc_pooled']:.3f} | {b[m+'_auc_within_question']['mean']:.3f} [{b[m+'_auc_within_question']['lo']:.3f}, {b[m+'_auc_within_question']['hi']:.3f}]" if kind == "relevant" else " | "
            L.append(f"| {nm} | {lab} | {d['min']:.3f} | {d['p5']:.3f} | {d['p25']:.3f} | {d['median']:.3f} | {d['p75']:.3f} | {d['p95']:.3f} | {d['max']:.3f} | {extra} |")
L += ["", "## 6. 足切りの値ごとの取りこぼしと混入", "",
      "- 再現＝関連ありのうち閾値以上の割合（ラベルのある組の中）。関連ありを1件でも落とす問い＝その問いの関連ありのうち1件以上が閾値未満。",
      "- seed が通る割合＝K で、問いの元の記憶そのものが閾値以上の割合。",
      "- 混入＝閾値以上の関連なしの数（1問あたり、ラベル付きの候補の中）。「全記憶で通る数」は閾値以上の記憶の数で、ラベルの有無に依らない。",
      "- 深いプールは各モデルの上位100まで。その外の関連ありは数えていない（再現は上限寄り）。", ""]
sel = {"sar": (0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82), "bge": (0.34, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.54)}
for key, title in (("jev_deep_all", "Jev ラベル・全問い"), ("jev_deep_K", "Jev ラベル・K のみ"), ("jev_deep_T", "Jev ラベル・T のみ"), ("human_all", "本流のラベル（32 問・人手プールの中だけ。浅い）")):
    if key not in s["thresholds"]: continue
    b = s["thresholds"][key]
    for m, nm in (("sar", "Sarashina"), ("bge", "bge-m3")):
        L += ["", f"### {title}・{nm}", "", "| 閾値 | 再現 | 関連ありを1件でも落とす問い | 関連ありが全部落ちる問い | seed が通る | 混入（関連なしが通る数/問） | 関連なしが通る割合 | 通った中の適合率 | ラベル付きで通る数/問 | 全記憶で通る数/問 |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in b[m + "_threshold_table"]:
            if r["t"] in sel[m]:
                L.append(f"| {r['t']:.2f} | {p(r['recall_pairs'])} | {p(r['questions_with_a_relevant_dropped'])} | {p(r['questions_with_all_relevant_dropped'])} | {p(r['seed_pass_rate'])} | {r['irrelevant_passed_per_q']:.1f} | {p(r['irrelevant_pass_rate'])} | {p(r['precision_pairs'])} | {r['pool_passed_per_q']:.1f} | {r['corpus_passed_per_q']:.0f} |")
L += ["", "### コサインの帯ごとの関連ありの割合（Jev ラベル・深いプール・全問い）", ""]
for m, nm in (("sar", "Sarashina"), ("bge", "bge-m3")):
    L += ["", f"{nm}:", "", "| 帯 | n | 関連ありの割合 |", "|---|---|---|"]
    for r in s["thresholds"]["jev_deep_all"][m + "_relevance_by_band"]:
        L.append(f"| {r['lo']:.2f}〜{r['hi']:.2f} | {r['n']} | {r['relevant_rate']*100:.0f}% |")
L += ["", "### K の seed が閾値を通る割合の信頼区間（Wilson 95%）", ""]
for t, d in s["seed_pass_wilson"].items(): L.append(f"- Sarashina {t}: {d['passed']}/{d['of']}（{d['wilson95'][0]*100:.1f}〜{d['wilson95'][1]*100:.1f}%）")
L += ["", "## 7. 足切りが、引いた上位に実際にどう効くか", "",
      "全記憶をコサインで並べた上位 60（本番の recall が返す数に近い）と上位 12（Jev が使えないときの戻り先の件数）で、閾値に当たる件数と、当たったときの関連あり（Jev）の変化。", "",
      "| 設定 | 上位60のうち閾値未満の平均 | 上位60に閾値未満が1件でもある問い | 上位12が12件に満たない問い | 上位12の適合率（足切りなし→あり） | 上位12に関連ありが1件でもある問い（なし→あり） | K の seed が上位12（なし→あり） |", "|---|---|---|---|---|---|---|"]
for k, d in s["floor_effect"].items():
    L.append(f"| {k.replace('sar_','Sarashina ').replace('bge_','bge-m3 ')} | {d['top60_below_mean']:.1f} | {d['questions_with_any_top60_below']*100:.1f}% | {d['questions_top12_short']*100:.1f}% | {d['top12_precision_no_floor']:.3f} → {d['top12_precision_with_floor']:.3f} | {d['top12_hit_no_floor']:.3f} → {d['top12_hit_with_floor']:.3f} | {d['K_seed_in_top12_no_floor']:.3f} → {d['K_seed_in_top12_with_floor']:.3f} |")
L += ["", f"モデル間の上位10の重なり（200問）: 平均 {s['top10_overlap_between_models']['mean']:.2f}、最小 {s['top10_overlap_between_models']['min']}、最大 {s['top10_overlap_between_models']['max']}。", ""]
open(os.path.join(R, "summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("wrote summary.md", len(L), "lines")
