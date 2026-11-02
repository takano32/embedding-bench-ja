#!/usr/bin/env python3
"""プールの各候補を Jev（nilkani06 リポの scripts/judge.py 経由）で判定する。結果は data/jev_labels.jsonl（再開できる）。

  python3 -I scripts/jev_label.py --run 1     # 候補の並びを乱数 seed=1 で並べ替えて判定
  python3 -I scripts/jev_label.py --run 2     # 別の並び（同じ入力での揺れを測る）

- K（既知の1件）: 「問いに答える事実を含むか（答えの要点と合うか）」。T（実メッセージ）: 本番のフックと同じ
  「この受信と同じ話題・人・案件・前の決定に関わり、思い出すと助けになるか」。
- 記憶の本文は先頭 300 字（本番のフックの MAX_CONTENT_CHARS）。1 リクエストに候補を全部入れる。
- 判定は yesno の p。ラベルは p>=0.5。judge.py の学習キャッシュ（.agents/configs、git 管理外）は judge.py 自身が更新する。
"""
import argparse, json, os, random, sys, time
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
from common import *

JUDGE_DIR = os.environ.get("JUDGE_DIR", "/home/takano32/nilkani06/scripts")
Q_T = ("`message` is a message a person just sent to a chat bot, and `memories` holds things the bot remembers. Memory `memories.{key}` "
       "is relevant to `message`: it is about the same topic, person, project or earlier decision, so recalling it "
       "would help the bot understand or answer `message`, as opposed to only sharing a few words with it or being "
       "about something else.")
Q_K = ("`question` is something a person asks the bot, and `expected_answer` is the key point a correct answer should contain. "
       "Memory `memories.{key}` states the specific information that answers `question` (it says what `expected_answer` says or an "
       "equivalent), as opposed to only being about the same topic or mentioning related things.")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", type=int, required=True); ap.add_argument("--only", default=""); ap.add_argument("--deep", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, JUDGE_DIR); import judge
    units = load_units(); qs = load_questions(); pools = json.load(open(os.path.join(D, "pools.json")))
    out_path = os.path.join(D, "jev_deep.jsonl" if a.deep else "jev_labels.jsonl"); done = set()
    if os.path.exists(out_path):
        for l in open(out_path, encoding="utf-8"):
            r = json.loads(l); done.add((r["qid"], r["run"]))
    only = set(a.only.split(",")) if a.only else None
    t_all = time.time(); n = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for q in qs:
            if (q["qid"], a.run) in done or (only and q["qid"] not in only): continue
            pl = list(pools[q["qid"]]["deep" if a.deep else "pool"]); random.Random(f"{q['qid']}-{a.run}").shuffle(pl)
            chunks = [pl[i:i + 60] for i in range(0, len(pl), 60)] if a.deep else [pl]
            res = {}; t0 = time.time(); ok = True
            for ch in chunks:
                keys = [f"m{i:02d}" for i in range(len(ch))]
                mem = {k: units[i]["text"][:300] for k, i in zip(keys, ch)}
                if q["type"] == "K":
                    state = {"question": q["query"], "expected_answer": q["answer"], "memories": mem}; tpl = Q_K
                else:
                    state = {"message": q["query"], "memories": mem}; tpl = Q_T
                questions = {k: {"type": "yesno", "question": tpl.format(key=k)} for k in keys}
                ans = None
                for attempt in range(3):
                    ans = judge.ask(state, questions, timeout=90)
                    if isinstance(ans, dict) and all(k in ans for k in keys): break
                    time.sleep(2 + 3 * attempt)
                if not (isinstance(ans, dict) and all(k in ans for k in keys)):
                    ok = False; break
                res.update({str(i): float(ans[k]["p"]) for k, i in zip(keys, ch)})
            if not ok:
                print("FAIL", q["qid"], flush=True); continue
            f.write(json.dumps({"qid": q["qid"], "run": a.run, "p": res, "chunks": len(chunks), "sec": round(time.time() - t0, 2)}) + "\n"); f.flush(); n += 1
            if n % 20 == 0: print(f"{n} done, {time.time()-t_all:.0f}s", flush=True)
    print("finished", n, f"{time.time()-t_all:.0f}s")


if __name__ == "__main__":
    main()
