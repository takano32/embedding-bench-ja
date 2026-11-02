#!/usr/bin/env python3
"""人手（本流が読んで付ける）判定用のシートを出す。モデルのスコアも順位も見せず、候補は乱数で並べる。

  python3 -I scripts/human_sheet.py pick            # 判定する問いを 32 問、乱数で決める（K 16・T 16）→ data/human_qids.json
  python3 -I scripts/human_sheet.py show 0 8        # 32 問のうち 0〜7 番目を表示（並びは data/human_map.json に固定）
  python3 -I scripts/human_sheet.py record answers.txt   # 「qid: 番号 番号 …」（関連ありの番号）を data/human_labels.json へ
"""
import json, os, random, re, sys
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
from common import *


def main():
    cmd = sys.argv[1]
    qs = {q["qid"]: q for q in load_questions()}
    pools = json.load(open(os.path.join(D, "pools.json"))); units = load_units()
    if cmd == "pick":
        r = random.Random(7); ks = [q for q in qs if q.startswith("K")]; ts = [q for q in qs if q.startswith("T")]
        ids = r.sample(ks, 16) + r.sample(ts, 16); r.shuffle(ids)
        json.dump(ids, open(os.path.join(D, "human_qids.json"), "w")); print(ids)
    elif cmd == "show":
        a, b = int(sys.argv[2]), int(sys.argv[3]); ids = json.load(open(os.path.join(D, "human_qids.json")))[a:b]
        mp_path = os.path.join(D, "human_map.json"); mp = json.load(open(mp_path)) if os.path.exists(mp_path) else {}
        for qid in ids:
            q = qs[qid]
            if qid not in mp:
                hp = list(pools[qid]["hpool"]); random.Random(f"h-{qid}").shuffle(hp); mp[qid] = hp
            print(f"=== {qid} ({q['type']})")
            if q["type"] == "K":
                print(f"問い: {q['query']}\n答えの要点: {q['answer']}\n基準: 記憶がその答えの事実を述べているものだけ 1")
            else:
                print(f"受信: {q['query'][:300]}\n基準: 同じ話題・人・案件・前の決定で、思い出すと理解や返答の助けになるもの 1")
            for n, i in enumerate(mp[qid], 1):
                print(f"{n}. {re.sub(chr(10), ' ', units[i]['text'])[:230]}")
        json.dump(mp, open(mp_path, "w"))
    elif cmd == "record":
        mp = json.load(open(os.path.join(D, "human_map.json"))); lp = os.path.join(D, "human_labels.json")
        lab = json.load(open(lp)) if os.path.exists(lp) else {}
        for line in open(sys.argv[2], encoding="utf-8"):
            line = line.strip()
            if not line: continue
            qid, nums = line.split(":", 1)
            rel = [mp[qid][int(x) - 1] for x in nums.split()]
            lab[qid] = {"judged": mp[qid], "rel": rel}
        json.dump(lab, open(lp, "w")); print("recorded", len(lab))


if __name__ == "__main__":
    main()
