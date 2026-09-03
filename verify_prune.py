"""Verify a pruned GGUF against its source. Exits non-zero on any failure."""
import sys, hashlib
import numpy as np
from gguf import GGUFReader, GGUFValueType
from gguf.constants import GGML_QUANT_SIZES

src, dst = sys.argv[1], sys.argv[2]
A, B = GGUFReader(src), GGUFReader(dst)
fails, warns = [], []


def chk(cond, msg):
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


def fstr(f, i):
    return str(bytes(f.parts[f.data[i]]), "utf-8", "replace")


def strs(r, key):
    f = r.fields[key]
    return [fstr(f, i) for i in range(len(f.data))]


def ints(r, key):
    f = r.fields[key]
    return [int(f.parts[j][0]) for j in f.data]


print("== tokenizer ==")
ta, tb = strs(A, "tokenizer.ggml.tokens"), strs(B, "tokenizer.ggml.tokens")
na, nb = len(ta), len(tb)
print(f"  vocab {na} -> {nb}")
tya, tyb = ints(A, "tokenizer.ggml.token_type"), ints(B, "tokenizer.ggml.token_type")
chk(len(tyb) == nb, f"token_type length {len(tyb)} == vocab {nb}")

# recover the keep-list by matching kept pieces in order
pos = {}
for i, p in enumerate(ta):
    pos.setdefault(p, i)
keep = []
ok_order = True
for p in tb:
    if p not in pos:
        ok_order = False
        break
    keep.append(pos[p])
chk(ok_order, "every kept token exists in the source vocab")
chk(keep == sorted(keep), "kept tokens preserve their original relative order")
chk(all(tya[o] == tyb[i] for i, o in enumerate(keep)), "token_type values follow their tokens")

print("== specials ==")
sa = {i for i in range(na) if tya[i] != 1}
sb = {i for i in range(nb) if tyb[i] != 1}
chk(len(sb) == len(sa), f"all {len(sa)} non-normal tokens retained (got {len(sb)})")
chk({ta[i] for i in sa} == {tb[i] for i in sb}, "special token strings identical")

nbyte = sum(1 for p in tb if len(p) == 1 and p in {chr(c) for c in range(0x100)} or False)
print("== byte fallback ==")
# a byte-fallback token is any single-byte token under the gpt2 alphabet
sys.path.insert(0, r"C:\Users\bsale\AppData\Local\Temp\claude\C--Users-bsale-Downloads-Qwen3-8-slim\af6dbea9-99a6-406c-b69f-f57f6113ce47\scratchpad")
from vocab_audit import bytes_to_unicode
U2B = {v: k for k, v in bytes_to_unicode().items()}


def tb_bytes(p):
    try:
        return bytes(U2B[c] for c in p)
    except KeyError:
        return None


bf = {tb_bytes(p) for p in tb}
chk(all(bytes([c]) in bf for c in range(256)), "all 256 byte-fallback tokens present")

print("== special ids remapped ==")
for k in ("tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id",
          "tokenizer.ggml.padding_token_id"):
    if k in A.fields:
        oa, ob = int(A.fields[k].contents()), int(B.fields[k].contents())
        chk(ta[oa] == tb[ob], f"{k}: {oa}->{ob} both = {ta[oa]!r}")

print("== merges ==")
ma, mb = strs(A, "tokenizer.ggml.merges"), strs(B, "tokenizer.ggml.merges")
print(f"  merges {len(ma)} -> {len(mb)}")
kept_set = set(tb)
bad = [m for m in mb if not (m.split(" ")[0] in kept_set
                             and m.split(" ")[1] in kept_set
                             and "".join(m.split(" ")) in kept_set)]
chk(not bad, f"every surviving merge references only kept tokens ({len(bad)} bad)")
chk(mb == [m for m in ma if m in set(mb)], "merges preserve original priority order")

print("== chat template / metadata ==")
for k in ("tokenizer.chat_template", "general.architecture", "qwen35.block_count",
          "qwen35.context_length", "tokenizer.ggml.pre", "tokenizer.ggml.model"):
    if k in A.fields:
        chk(A.fields[k].contents() == B.fields[k].contents(), f"{k} preserved")

print("== tensors ==")
TA = {t.name: t for t in A.tensors}
TB = {t.name: t for t in B.tensors}
chk(set(TA) == set(TB), f"same tensor set ({len(TA)} vs {len(TB)})")
chk([t.name for t in A.tensors] == [t.name for t in B.tensors], "same tensor order")
chk(all(TA[n].tensor_type == TB[n].tensor_type for n in TA), "all ggml types unchanged")

VOCAB = ("token_embd.weight", "output.weight")
nonvocab_bad, shape_bad = [], []
for n, t in TA.items():
    u = TB[n]
    if n in VOCAB:
        if [int(x) for x in u.shape] != [int(t.shape[0]), nb]:
            shape_bad.append((n, list(u.shape)))
    else:
        if [int(x) for x in u.shape] != [int(x) for x in t.shape]:
            shape_bad.append((n, list(u.shape)))
chk(not shape_bad, f"all shapes correct ({shape_bad[:3]})")

print("  hashing non-vocab tensor payloads (byte-identity)...")
for n, t in TA.items():
    if n in VOCAB:
        continue
    u = TB[n]
    da = A.data[t.data_offset: t.data_offset + int(t.n_bytes)]
    db = B.data[u.data_offset: u.data_offset + int(u.n_bytes)]
    if int(t.n_bytes) != int(u.n_bytes) or \
       hashlib.blake2b(da, digest_size=16).digest() != hashlib.blake2b(db, digest_size=16).digest():
        nonvocab_bad.append(n)
chk(not nonvocab_bad, f"all {len(TA)-2} non-vocab tensors byte-identical "
                      f"({len(nonvocab_bad)} differ: {nonvocab_bad[:5]})")

print("  verifying gathered vocab rows...")
for n in VOCAB:
    t, u = TA[n], TB[n]
    bs, ts = GGML_QUANT_SIZES[t.tensor_type]
    rb = int(t.shape[0]) // bs * ts
    da = A.data[t.data_offset: t.data_offset + int(t.n_bytes)].reshape(na, rb)
    db = B.data[u.data_offset: u.data_offset + int(u.n_bytes)].reshape(nb, rb)
    step = max(1, nb // 4000)
    idx = list(range(0, nb, step)) + [0, nb - 1]
    mism = [i for i in idx if not np.array_equal(db[i], da[keep[i]])]
    chk(not mism, f"{n}: sampled {len(idx)} rows match source row keep[i] "
                  f"({len(mism)} mismatches)")

print()
if fails:
    print(f"FAILED: {len(fails)} check(s)")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("ALL CHECKS PASSED")
