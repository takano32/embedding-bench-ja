"""ベンチ共通。データの読み込み、スコア計算、字 bigram の BM25（第3の検索。プールを2モデルだけに偏らせないため）。"""
import json, os, re, unicodedata, hashlib
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); D = os.path.join(ROOT, "data"); R = os.path.join(ROOT, "results")
SAR_PREFIX = "[[sarashina:query]]"


def load_units():
    return [json.loads(l) for l in open(os.path.join(D, "units.jsonl"), encoding="utf-8")]


def load_questions():
    return [json.loads(l) for l in open(os.path.join(D, "questions.jsonl"), encoding="utf-8")]


def _key(model, prefix, text):
    return hashlib.sha1((model + "\x00" + prefix + "\x00" + text).encode("utf-8")).hexdigest()


def load_qvecs(qs):
    """{'bge': (n,1024), 'sarashina': (n,1536)}（L2 正規化）と、問いごとの遅延 ms。"""
    out, lat = {}, {}
    for name, model, prefix in (("bge", "bge-m3", ""), ("sarashina", "sarashina3-embedding", SAR_PREFIX)):
        cache = {}
        for l in open(os.path.join(D, "emb_cache", f"{name}.jsonl"), encoding="utf-8"):
            r = json.loads(l); cache[r["key"]] = r
        rows = [cache[_key(model, prefix, q["query"])] for q in qs]
        v = np.array([r["vec"] for r in rows], dtype=np.float32); v /= np.linalg.norm(v, axis=1, keepdims=True)
        out[name] = v; lat[name] = [r["ms"] for r in rows]
    return out, lat


def load_doc_vecs():
    return {"sarashina": np.load(os.path.join(D, "vec_sarashina.npy")), "bge": np.load(os.path.join(D, "vec_bge.npy"))}


def norm_text(s):
    return unicodedata.normalize("NFKC", s).lower()


def bigrams(s):
    s = re.sub(r"\s+", " ", norm_text(s))
    return [s[i:i + 2] for i in range(len(s) - 1)]


class BM25:
    def __init__(self, texts, k1=1.2, b=0.75):
        self.k1, self.b = k1, b
        post = defaultdict(list); self.dl = np.zeros(len(texts), dtype=np.float32)
        for i, t in enumerate(texts):
            bg = bigrams(t); self.dl[i] = len(bg)
            tf = defaultdict(int)
            for g in bg: tf[g] += 1
            for g, c in tf.items(): post[g].append((i, c))
        self.avg = float(self.dl.mean()); self.n = len(texts)
        self.post = {g: (np.array([x[0] for x in v]), np.array([x[1] for x in v], dtype=np.float32)) for g, v in post.items()}

    def scores(self, q):
        s = np.zeros(self.n, dtype=np.float32)
        for g in set(bigrams(q)):
            if g not in self.post: continue
            idx, tf = self.post[g]; df = len(idx)
            idf = np.log(1 + (self.n - df + 0.5) / (df + 0.5))
            s[idx] += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.dl[idx] / self.avg))
        return s
