"""Symmetric vocabulary prune for Qwen3.8 GGUF files.

Row-gathers token_embd.weight and output.weight in QUANTIZED space (no
dequant/requant, so no added error), rewrites the tokenizer arrays, and
remaps the special-token ids.

Policy P1 = ASCII-only, plus an always-keep set of:
  - every non-normal token (specials, control, user-defined, unused)
  - all 256 byte-fallback tokens
  - every token whose bytes are not valid standalone UTF-8 (BPE fragments)

P1's keep-set is closed under substring, so every merge rule producing a
kept token has kept parents: BPE reachability is preserved exactly.
"""
import sys, os, argparse
import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType
from gguf.constants import GGML_QUANT_SIZES

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vocab_audit import bytes_to_unicode

B2U = bytes_to_unicode()
U2B = {v: k for k, v in B2U.items()}

# P1M keep-set: characters a reasoning model actually emits, measured rather
# than guessed. Counts are from 20 generations of ThinkingCap at effort=high
# (bench/thinkingcap-iq4xs-pure.json):
#
#   U+2192 ->  x81    U+2264 <=  x35    U+221E inf x21    U+2265 >=  x4
#   U+2308/9/A/B ceil+floor brackets x4 each          U+2013 en-dash x7
#   U+00B2 superscript-2 x6   U+2019 curly apostrophe x9   U+2022 bullet
#
# Under a pure-ASCII prune each of these becomes 3 byte-fallback tokens instead
# of 1, inflating the token count of a model finetuned to REDUCE it. Keeping
# ~20 codepoints costs a negligible number of rows.
#
# Emoji are deliberately NOT kept: they are decoration, they cost 4 tokens each
# under byte-fallback, and dropping them pushes the output register toward
# plain technical prose.
#
# Per-CHARACTER test, exactly like P2, so the keep-set stays closed under
# substring and BPE reachability is preserved.
P1M_EXTRA = set(
    "→←↔↓↑↕"      # arrows
    "≤≥≠≈∞"            # <= >= != ~= inf
    "∈∉⊆⊂∪∩"      # set relations
    "⌈⌉⌊⌋"                  # ceiling / floor
    "∑∏√±×÷"      # sum prod sqrt +- x /
    "²³·…"                  # squared cubed middot ellipsis
    "–—’“”•"      # dashes, smart quotes, bullet
)

VOCAB_TENSORS = ("token_embd.weight", "output.weight")
SKIP_KEYS = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count",
             "general.architecture"}
ID_KEYS = ("tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id",
           "tokenizer.ggml.padding_token_id", "tokenizer.ggml.unknown_token_id",
           "tokenizer.ggml.seperator_token_id", "tokenizer.ggml.eot_token_id",
           "tokenizer.ggml.eom_token_id")


def field_str(f, i):
    return str(bytes(f.parts[f.data[i]]), "utf-8", "replace")


def token_bytes(piece):
    """Reverse the GPT-2 byte-level alphabet. None if not encodable."""
    try:
        return bytes(U2B[c] for c in piece)
    except KeyError:
        return None


def build_keep(pieces, ttypes, policy):
    n = len(pieces)
    keep = set()
    stats = dict(special=0, bytefb=0, partial=0, ascii=0, accented=0, mathsym=0)

    for i in range(n):
        if ttypes[i] != 1:                       # specials / control / unused
            keep.add(i); stats["special"] += 1; continue
        raw = token_bytes(pieces[i])
        if raw is None:                          # not byte-level encodable
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
        if policy == "P1M" and all(ord(c) < 128 or c in P1M_EXTRA for c in text):
            keep.add(i); stats["mathsym"] += 1; continue
    return sorted(keep), stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--policy", default="P1", choices=["P1", "P2", "P1M"],
                    help="P1=ASCII  P2=ASCII+Latin  P1M=ASCII+measured math/arrow symbols")
    a = ap.parse_args()

    r = GGUFReader(a.src)
    tf = r.fields["tokenizer.ggml.tokens"]
    n_vocab = len(tf.data)
    pieces = [field_str(tf, i) for i in range(n_vocab)]

    ttf = r.fields["tokenizer.ggml.token_type"]
    ttypes = [int(ttf.parts[j][0]) for j in ttf.data]

    keep, stats = build_keep(pieces, ttypes, a.policy)
    old2new = {o: i for i, o in enumerate(keep)}
    n_new = len(keep)
    print(f"policy {a.policy}: keeping {n_new} / {n_vocab} "
          f"({100*n_new/n_vocab:.2f}%), dropping {n_vocab-n_new}")
    print("  breakdown:", stats)

    keep_pieces = {pieces[i] for i in keep}

    # ---- merges: keep iff both parents and the product survive ----
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

    # ---- copy KV, substituting tokenizer arrays and remapped ids ----
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
                raise SystemExit(f"FATAL: {key}={old} was pruned")
            new = old2new[old]
            print(f"  remap {key}: {old} -> {new}")
            w.add_key_value(key, new, vt)
        elif vt == GGUFValueType.ARRAY:
            sub = f.types[1]
            if sub == GGUFValueType.STRING:
                vals = [field_str(f, i) for i in range(len(f.data))]
            else:
                vals = [f.parts[j].tolist()[0] for j in f.data]
            w.add_key_value(key, vals, GGUFValueType.ARRAY, sub)
        else:
            w.add_key_value(key, f.contents(), vt)

    # ---- tensor info, original order ----
    plan = []
    for t in r.tensors:
        ne0 = int(t.shape[0])
        bs, ts = GGML_QUANT_SIZES[t.tensor_type]
        if ne0 % bs:
            raise SystemExit(f"{t.name}: ne0 {ne0} not divisible by block {bs}")
        row_bytes = ne0 // bs * ts
        ne = [int(d) for d in t.shape]          # ne0 first (reader order)
        higher = list(reversed(ne[1:]))         # numpy order, minus last dim
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

        # When dtype is uint8 the writer treats the last dim as BYTES and
        # converts it back to elements, so hand it the byte shape.
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
            mat = raw.reshape(int(t.shape[1]), row_bytes)
            out = np.ascontiguousarray(mat[idx])
            w.write_tensor_data(out)
            del out
        else:
            w.write_tensor_data(raw)
    w.close()
    print(f"\nwrote {a.dst} ({os.path.getsize(a.dst)/1e9:.3f} GB)")


if __name__ == "__main__":
    main()
