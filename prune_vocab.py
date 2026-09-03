"""Vocabulary prune for byte-level BPE GGUF files.

Row-gathers token_embd.weight and output.weight in quantized space, rewrites
the tokenizer arrays, remaps the special-token ids. No dequant/requant, so
surviving rows are bit-identical to the source.

Policies, all per-character so the keep-set stays closed under substring
(every merge producing a kept token has kept parents, so BPE reachability
survives):

  P1    ASCII
  P2    ASCII + Latin-1/Latin-Extended-A
  P1M   ASCII + math and typography symbols
  P1G   P1M + Greek, CLI box/status glyphs, units, currency

Nothing here is architecture-specific; metadata is copied key by key. The one
requirement is tokenizer.ggml.model = gpt2/bpe. An SPM vocab stores pieces as
raw UTF-8 with a U+2581 marker, so the GPT-2 byte reversal would produce wrong
bytes rather than fail - hence the guard in main().
"""
import argparse
import os
import sys

import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType
from gguf.constants import GGML_QUANT_SIZES

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vocab_audit import bytes_to_unicode

B2U = bytes_to_unicode()
U2B = {v: k for k, v in B2U.items()}

# Counts from 20 ThinkingCap generations at effort=high
# (bench/thinkingcap-iq4xs-pure.json). Emoji are left out deliberately: 4
# tokens each under byte-fallback, and dropping them pushes the output
# register toward plain prose.
#
# Frozen - TC38-ASCII-P1M was built with this set and verify_prune.py
# --policy P1M still has to reproduce it. Additions go in P1G.
#
# Note ~13 of these have no token in the Qwen3.5 vocab at all (nothing
# contains U+221E, U+2308-B, U+2211, ...), so for that model they are inert.
# Harmless, and they still apply to vocabs that do carry them.
P1M_EXTRA = set(
    "→←↔↓↑↕"
    "≤≥≠≈∞"
    "∈∉⊆⊂∪∩"
    "⌈⌉⌊⌋"
    "∑∏√±×÷"
    "²³·…"
    "–—’“”•"
)

# Everything below is verified present in the Qwen3.5 vocab as a single token
# (scan_nonascii.py). Greek is the bulk of it and the main reason this policy
# exists: 38 letters, all single tokens, none kept by P1M, and they turn up
# constantly in ML code, statistics and physics writeups.
P1G_EXTRA = P1M_EXTRA | set(
    "αβγδεζηθικλμνξπρστυφχψωΑΒΓΔΕΖΗΘΛΞΠΡΣΦΨΩ"  # identifiers, math
    "¬∀≫⇒"                                       # logic
    "─│━║═╗╝"                                     # tree output, tables
    "✓✔"                                          # test/CI status lines
    "█░"                                          # progress bars
    "°µ′″"                                        # units
    "€£¥¢"
    "§©®™"
    "¹½¼¾"
    "«»‘"
)

VOCAB_TENSORS = ("token_embd.weight", "output.weight")
SKIP_KEYS = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count",
             "general.architecture"}
ID_KEYS = ("tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id",
           "tokenizer.ggml.padding_token_id", "tokenizer.ggml.unknown_token_id",
           "tokenizer.ggml.seperator_token_id", "tokenizer.ggml.eot_token_id",
           "tokenizer.ggml.eom_token_id")

EXTRA_SETS = {"P1M": P1M_EXTRA, "P1G": P1G_EXTRA}


def field_str(f, i):
    return str(bytes(f.parts[f.data[i]]), "utf-8", "replace")


def token_bytes(piece):
    """Reverse the GPT-2 byte-level alphabet. None if not encodable."""
    try:
        return bytes(U2B[c] for c in piece)
    except KeyError:
        return None


def resolve_extra(policy, keep_chars="", drop_chars=""):
    """Policy set plus --keep-chars minus --drop-chars.

    Dropping only affects the policy's symbol set; ASCII is matched by an
    earlier branch and cannot be dropped without breaking substring closure.
    """
    extra = set(EXTRA_SETS.get(policy) or ())
    extra |= set(keep_chars or "")
    extra -= set(drop_chars or "")
    return extra or None


def inert_chars(pieces, chars):
    """Which of `chars` appear in no token at all, so keeping them is a no-op."""
    wanted = {c for c in chars if not c.isascii()}
    if not wanted:
        return set()
    seen = set()
    for p in pieces:
        raw = token_bytes(p)
        if raw is None:
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        seen |= wanted & set(text)
        if seen == wanted:
            break
    return wanted - seen


def build_keep(pieces, ttypes, policy, keep_chars="", drop_chars=""):
    keep = set()
    stats = dict(special=0, bytefb=0, partial=0, ascii=0, accented=0, extra=0)
    extra = resolve_extra(policy, keep_chars, drop_chars)

    for i in range(len(pieces)):
        if ttypes[i] != 1:                       # specials, control, unused
            keep.add(i); stats["special"] += 1; continue
        raw = token_bytes(pieces[i])
        if raw is None:
            keep.add(i); stats["special"] += 1; continue
        if len(raw) == 1:                        # byte fallback
            keep.add(i); stats["bytefb"] += 1; continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:               # partial-UTF8 BPE fragment
            keep.add(i); stats["partial"] += 1; continue
        if all(ord(c) < 128 for c in text):
            keep.add(i); stats["ascii"] += 1; continue
        if policy == "P2" and all(ord(c) < 128 or 0x00A0 <= ord(c) <= 0x024F
                                  for c in text):
            keep.add(i); stats["accented"] += 1; continue
        if extra and all(ord(c) < 128 or c in extra for c in text):
            keep.add(i); stats["extra"] += 1; continue
    return sorted(keep), stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--policy", default="P1", choices=["P1", "P2", "P1M", "P1G"],
                    help="P1=ASCII  P2=+Latin  P1M=+math symbols  P1G=+Greek/CLI")
    ap.add_argument("--keep-chars", default="", metavar="STR",
                    help="extra characters to keep, e.g. --keep-chars 'πλΩ'")
    ap.add_argument("--drop-chars", default="", metavar="STR",
                    help="characters to remove from the policy set")
    a = ap.parse_args()

    if os.path.abspath(a.src) == os.path.abspath(a.dst):
        raise SystemExit("src and dst are the same file")

    r = GGUFReader(a.src)

    model = str(r.fields["tokenizer.ggml.model"].contents()).lower() \
        if "tokenizer.ggml.model" in r.fields else ""
    if model not in ("gpt2", "bpe"):
        raise SystemExit(f"tokenizer.ggml.model is {model!r}; this needs a "
                         f"byte-level BPE vocab (gpt2/bpe)")

    tf = r.fields["tokenizer.ggml.tokens"]
    n_vocab = len(tf.data)
    pieces = [field_str(tf, i) for i in range(n_vocab)]

    if "tokenizer.ggml.token_type" not in r.fields:
        raise SystemExit("no tokenizer.ggml.token_type; specials cannot be "
                         "identified, so they cannot be protected")
    ttf = r.fields["tokenizer.ggml.token_type"]
    ttypes = [int(ttf.parts[j][0]) for j in ttf.data]
    if len(ttypes) != n_vocab:
        raise SystemExit(f"token_type has {len(ttypes)} entries, vocab has {n_vocab}")

    if a.drop_chars and any(c.isascii() for c in a.drop_chars):
        raise SystemExit("--drop-chars cannot drop ASCII; that would break "
                         "substring closure and byte fallback")

    keep, stats = build_keep(pieces, ttypes, a.policy, a.keep_chars, a.drop_chars)
    old2new = {o: i for i, o in enumerate(keep)}
    n_new = len(keep)
    label = a.policy
    if a.keep_chars or a.drop_chars:
        label += f" (+{len(set(a.keep_chars))} -{len(set(a.drop_chars))} chars)"
    print(f"policy {label}: keeping {n_new} / {n_vocab} "
          f"({100*n_new/n_vocab:.2f}%), dropping {n_vocab-n_new}")
    print("  breakdown:", stats)

    if a.keep_chars:
        dead = inert_chars(pieces, a.keep_chars)
        if dead:
            print(f"  note: no token contains {''.join(sorted(dead))!r} - "
                  f"keeping {'them' if len(dead) > 1 else 'it'} has no effect")

    keep_pieces = {pieces[i] for i in keep}

    # A merge survives only if both parents and the product do.
    new_merges = None
    if "tokenizer.ggml.merges" in r.fields:
        mf = r.fields["tokenizer.ggml.merges"]
        n_merges = len(mf.data)
        new_merges = []
        for i in range(n_merges):
            m = field_str(mf, i)
            l, _, rr = m.partition(" ")
            if l in keep_pieces and rr in keep_pieces and (l + rr) in keep_pieces:
                new_merges.append(m)
        print(f"  merges {n_merges} -> {len(new_merges)} "
              f"(dropped {n_merges-len(new_merges)})")

    arch = r.fields["general.architecture"].contents()
    w = GGUFWriter(a.dst, arch, use_temp_file=False)

    new_types = [ttypes[i] for i in keep]
    for key, f in r.fields.items():
        if key in SKIP_KEYS:
            continue
        vt = f.types[0]
        if key == "tokenizer.ggml.tokens":
            w.add_key_value(key, [pieces[i] for i in keep],
                            GGUFValueType.ARRAY, GGUFValueType.STRING)
        elif key == "tokenizer.ggml.token_type":
            w.add_key_value(key, new_types,
                            GGUFValueType.ARRAY, GGUFValueType.INT32)
        elif key == "tokenizer.ggml.merges":
            w.add_key_value(key, new_merges,
                            GGUFValueType.ARRAY, GGUFValueType.STRING)
        elif key in ID_KEYS:
            old = int(f.contents())
            if old not in old2new:
                raise SystemExit(f"{key}={old} was pruned")
            print(f"  remap {key}: {old} -> {old2new[old]}")
            w.add_key_value(key, old2new[old], vt)
        elif vt == GGUFValueType.ARRAY:
            sub = f.types[1]
            if sub == GGUFValueType.STRING:
                vals = [field_str(f, i) for i in range(len(f.data))]
            else:
                vals = [f.parts[j].tolist()[0] for j in f.data]
            w.add_key_value(key, vals, GGUFValueType.ARRAY, sub)
        else:
            w.add_key_value(key, f.contents(), vt)

    plan = []
    for t in r.tensors:
        ne0 = int(t.shape[0])
        bs, ts = GGML_QUANT_SIZES[t.tensor_type]
        if ne0 % bs:
            raise SystemExit(f"{t.name}: ne0 {ne0} not divisible by block {bs}")
        row_bytes = ne0 // bs * ts
        ne = [int(d) for d in t.shape]
        higher = list(reversed(ne[1:]))
        dt = t.data.dtype

        if t.name in VOCAB_TENSORS:
            if ne[1] != n_vocab:
                raise SystemExit(f"{t.name}: rows {ne[1]} != vocab {n_vocab}")
            higher = [n_new]
            nbytes = n_new * row_bytes
            print(f"  {t.name}: {t.tensor_type.name} {row_bytes} B/row, "
                  f"{ne[1]} -> {n_new} rows, "
                  f"{t.n_bytes/1e9:.3f} -> {nbytes/1e9:.3f} GB")
        else:
            nbytes = int(t.n_bytes)

        # uint8 dtype means the writer reads the last dim as bytes and converts
        # back to elements, so hand it the byte shape.
        shape = tuple(higher + ([row_bytes] if dt == np.uint8 else [ne[0]]))
        w.add_tensor_info(t.name, shape, dt, nbytes, t.tensor_type)
        plan.append((t, row_bytes))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()

    idx = np.array(keep, dtype=np.int64)
    for t, row_bytes in plan:
        raw = r.data[t.data_offset: t.data_offset + int(t.n_bytes)]
        if t.name in VOCAB_TENSORS:
            out = np.ascontiguousarray(raw.reshape(int(t.shape[1]), row_bytes)[idx])
            w.write_tensor_data(out)
            del out
        else:
            w.write_tensor_data(raw)
    w.close()
    print(f"\nwrote {a.dst} ({os.path.getsize(a.dst)/1e9:.3f} GB)")


if __name__ == "__main__":
    main()
