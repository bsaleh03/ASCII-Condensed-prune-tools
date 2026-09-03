"""Verify a pruned GGUF against its source. Non-zero exit on any failure.

  python verify_prune.py SRC.gguf DST.gguf [--policy P1|P2|P1M|P1G]

Structural checks always run: kept tokens exist in the source and keep their
order, token_type follows its token, specials and all 256 byte-fallback tokens
survive, ids remap to the same strings, merges reference only kept tokens and
keep their priority, non-vocab tensors are byte-identical, sampled vocab rows
match their source rows.

--policy adds the check the structural ones cannot make: that the gather
implemented the policy asked for. A prune run with the wrong --policy produces
a perfectly consistent file containing the wrong tokens and passes everything
else, so this replays build_keep against the source and demands an exact match.
"""
import argparse
import hashlib
import os
import sys

import numpy as np
from gguf import GGUFReader
from gguf.constants import GGML_QUANT_SIZES

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vocab_audit import bytes_to_unicode

VOCAB = ("token_embd.weight", "output.weight")
BPE_MODELS = {"gpt2", "bpe"}

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--policy", choices=["P1", "P2", "P1M", "P1G"],
                help="also verify the kept set matches this policy exactly")
ap.add_argument("--keep-chars", default="", metavar="STR",
                help="must match the --keep-chars the prune was run with")
ap.add_argument("--drop-chars", default="", metavar="STR",
                help="must match the --drop-chars the prune was run with")
args = ap.parse_args()

A, B = GGUFReader(args.src), GGUFReader(args.dst)
fails = []


def chk(cond, msg):
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


def fstr(f, i):
    return str(bytes(f.parts[f.data[i]]), "utf-8", "replace")


def strs(r, key):
    if key not in r.fields:
        return None
    f = r.fields[key]
    return [fstr(f, i) for i in range(len(f.data))]


def ints(r, key):
    if key not in r.fields:
        return None
    f = r.fields[key]
    return [int(f.parts[j][0]) for j in f.data]


def contents(r, key):
    return r.fields[key].contents() if key in r.fields else None


print("== model ==")
arch = str(contents(A, "general.architecture"))
chk(arch == str(contents(B, "general.architecture")), f"architecture preserved ({arch})")
model = str(contents(A, "tokenizer.ggml.model") or "")
print(f"  tokenizer.ggml.model = {model!r}")
if model.lower() not in BPE_MODELS:
    # Reversing the GPT-2 alphabet against an SPM vocab yields wrong bytes
    # silently, which would make every check below meaningless.
    sys.exit(f"\nthis needs a byte-level BPE vocab; file says {model!r}")

print("== tokenizer ==")
ta, tb = strs(A, "tokenizer.ggml.tokens"), strs(B, "tokenizer.ggml.tokens")
na, nb = len(ta), len(tb)
print(f"  vocab {na} -> {nb}")
tya, tyb = ints(A, "tokenizer.ggml.token_type"), ints(B, "tokenizer.ggml.token_type")
have_types = tya is not None and tyb is not None
if have_types:
    chk(len(tyb) == nb, f"token_type length {len(tyb)} == vocab {nb}")
else:
    print("  (no token_type; type checks skipped)")

# Recover the keep-list by matching pieces in order. Collect misses rather than
# breaking - a short keep-list made the row check below raise IndexError
# instead of reporting the failure.
pos = {}
for i, p in enumerate(ta):
    pos.setdefault(p, i)
keep, missing = [], []
for p in tb:
    (keep.append(pos[p]) if p in pos else missing.append(p))
chk(not missing, f"every kept token exists in the source vocab "
                 f"({len(missing)} missing, e.g. {missing[:3]})")
chk(keep == sorted(keep), "kept tokens preserve their original relative order")
if have_types and not missing:
    chk(all(tya[o] == tyb[i] for i, o in enumerate(keep)),
        "token_type values follow their tokens")

if have_types:
    print("== specials ==")
    sa = {i for i in range(na) if tya[i] != 1}
    sb = {i for i in range(nb) if tyb[i] != 1}
    chk(len(sb) == len(sa), f"all {len(sa)} non-normal tokens retained (got {len(sb)})")
    chk({ta[i] for i in sa} == {tb[i] for i in sb}, "special token strings identical")

print("== byte fallback ==")
U2B = {v: k for k, v in bytes_to_unicode().items()}


def tb_bytes(p):
    try:
        return bytes(U2B[c] for c in p)
    except KeyError:
        return None


bf = {tb_bytes(p) for p in tb}
gone = [c for c in range(256) if bytes([c]) not in bf]
chk(not gone, f"all 256 byte-fallback tokens present ({len(gone)} missing)")

print("== special ids remapped ==")
for k in ("tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id",
          "tokenizer.ggml.padding_token_id", "tokenizer.ggml.unknown_token_id",
          "tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id"):
    if k in A.fields and k in B.fields:
        oa, ob = int(A.fields[k].contents()), int(B.fields[k].contents())
        if oa >= na or ob >= nb:
            chk(False, f"{k}: id out of range ({oa} -> {ob})")
        else:
            chk(ta[oa] == tb[ob], f"{k}: {oa}->{ob} both = {ta[oa]!r}")

ma, mb = strs(A, "tokenizer.ggml.merges"), strs(B, "tokenizer.ggml.merges")
print("== merges ==")
if ma is None:
    print("  (none in this model)")
else:
    print(f"  merges {len(ma)} -> {len(mb)}")
    kept = set(tb)
    bad = []
    for m in mb:
        l, _, rr = m.partition(" ")
        if not rr or l not in kept or rr not in kept or (l + rr) not in kept:
            bad.append(m)
    chk(not bad, f"every surviving merge references only kept tokens ({len(bad)} bad)")
    mb_set = set(mb)  # hoisted; rebuilding this per element cost ~49 s at 128k
    chk(mb == [m for m in ma if m in mb_set], "merges preserve original priority order")

print("== metadata ==")
meta = ["tokenizer.chat_template", "general.architecture", "tokenizer.ggml.pre",
        "tokenizer.ggml.model", "general.name", "general.file_type"]
meta += [k for k in A.fields if k.startswith(arch + ".") and "vocab" not in k]
for k in meta:
    if k in A.fields:
        chk(contents(A, k) == contents(B, k), f"{k} preserved")

print("== tensors ==")
TA = {t.name: t for t in A.tensors}
TB = {t.name: t for t in B.tensors}
chk(set(TA) == set(TB), f"same tensor set ({len(TA)} vs {len(TB)})")
chk([t.name for t in A.tensors] == [t.name for t in B.tensors], "same tensor order")
chk(all(TA[n].tensor_type == TB[n].tensor_type for n in TA if n in TB),
    "all ggml types unchanged")

vocab_present = [n for n in VOCAB if n in TA]   # tied embeddings have no output.weight
print(f"  vocab tensors present: {vocab_present}")

shape_bad = []
for n, t in TA.items():
    if n not in TB:
        continue
    want = [int(t.shape[0]), nb] if n in vocab_present else [int(x) for x in t.shape]
    if [int(x) for x in TB[n].shape] != want:
        shape_bad.append((n, list(TB[n].shape), want))
chk(not shape_bad, f"all shapes correct ({shape_bad[:3]})")

print("  hashing non-vocab tensor payloads...")
diff = []
for n, t in TA.items():
    if n in vocab_present or n not in TB:
        continue
    u = TB[n]
    da = A.data[t.data_offset: t.data_offset + int(t.n_bytes)]
    db = B.data[u.data_offset: u.data_offset + int(u.n_bytes)]
    if int(t.n_bytes) != int(u.n_bytes) or \
       hashlib.blake2b(da, digest_size=16).digest() != \
       hashlib.blake2b(db, digest_size=16).digest():
        diff.append(n)
chk(not diff, f"all {len(TA)-len(vocab_present)} non-vocab tensors byte-identical "
              f"({len(diff)} differ: {diff[:5]})")

if missing:
    print("  (skipping row check, keep-list incomplete)")
else:
    print("  verifying gathered vocab rows...")
    for n in vocab_present:
        t, u = TA[n], TB[n]
        bs, ts = GGML_QUANT_SIZES[t.tensor_type]
        rb = int(t.shape[0]) // bs * ts
        da = A.data[t.data_offset: t.data_offset + int(t.n_bytes)].reshape(na, rb)
        db = B.data[u.data_offset: u.data_offset + int(u.n_bytes)].reshape(nb, rb)
        step = max(1, nb // 4000)
        idx = sorted(set(list(range(0, nb, step)) + [0, nb - 1]))
        mism = [i for i in idx if not np.array_equal(db[i], da[keep[i]])]
        chk(not mism, f"{n}: sampled {len(idx)} rows match source row keep[i] "
                      f"({len(mism)} mismatches)")

print("== policy conformance ==")
if not args.policy:
    print("  skipped - pass --policy to check the kept set is the one the policy")
    print("  specifies. Without it a prune that kept the wrong tokens still passes")
    print("  everything above; those checks only prove internal consistency.")
elif not have_types:
    print("  skipped - needs token_type")
else:
    from prune_vocab import build_keep
    expected, stats = build_keep(ta, tya, args.policy,
                                 args.keep_chars, args.drop_chars)
    label = args.policy
    if args.keep_chars or args.drop_chars:
        label += f" +{args.keep_chars!r} -{args.drop_chars!r}"
    print(f"  replayed {label}: expects {len(expected)} tokens")
    print("  breakdown:", {k: v for k, v in stats.items() if v})
    if keep == expected:
        chk(True, f"kept set is exactly what policy {args.policy} specifies")
    else:
        ks, es = set(keep), set(expected)
        chk(False, f"kept set does not match policy {args.policy}: "
                   f"{len(ks - es)} unexpected, {len(es - ks)} missing")
        for label, ids in (("unexpectedly kept", sorted(ks - es)[:5]),
                           ("wrongly dropped", sorted(es - ks)[:5])):
            if ids:
                print(f"    {label}:", [(i, ta[i]) for i in ids])

print()
if fails:
    print(f"FAILED: {len(fails)} check(s)")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("ALL CHECKS PASSED")
