"""List non-ASCII characters the vocab can represent as single tokens.

Under a pure-ASCII prune each of these costs len(utf8) byte-fallback tokens
instead of 1. This ranks candidates for the P1M keep-set by category so the
decision is made against the actual vocab rather than from memory.
"""
import sys
import unicodedata
from collections import defaultdict

from gguf import GGUFReader

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from vocab_audit import bytes_to_unicode, utf8_console
from prune_vocab import P1M_EXTRA

U2B = {v: k for k, v in bytes_to_unicode().items()}

CATEGORIES = [
    ("greek/math letters", "αβγδεζηθικλμνξπρστυφχψωΑΒΓΔΕΖΗΘΛΞΠΡΣΦΨΩ"),
    ("math operators",     "≤≥≠≈≡∞±×÷√∑∏∫∂∇∈∉⊂⊆∪∩∧∨¬⊕⊗∝∀∃≪≫∴∵"),
    ("arrows",             "→←↑↓↔↕⇒⇐⇔⟶⟵↦"),
    ("box drawing",        "─│┌┐└┘├┤┬┴┼━┃╔╗╚╝║═╠╣╦╩╬"),
    ("block/shade",        "█▀▄░▒▓▏▎▍▌▋▊▉"),
    ("check/cross/bullet", "✓✔✗✘✕×•·∙◦‣▪▫■□●○◆◇★☆"),
    ("units/scientific",   "°µμΩÅ℃℉‰′″"),
    ("currency",           "€£¥¢₹₽₩¤"),
    ("legal/editorial",    "©®™§¶†‡"),
    ("quotes/dashes",      "–—―‘’“”«»‹›„‚"),
    ("superscripts",       "⁰¹²³⁴⁵⁶⁷⁸⁹ⁿ½¼¾"),
    ("brackets",           "⌈⌉⌊⌋⟨⟩「」【】"),
    ("spaces/invisible",   " ​  ﻿"),
    ("ellipsis/misc",      "…‥⋯№℠"),
]


def main(path):
    utf8_console()
    r = GGUFReader(path)
    fld = r.fields["tokenizer.ggml.tokens"]
    n = len(fld.data)

    # every single-character token the vocab holds, by character
    single = {}
    for i in range(n):
        piece = str(bytes(fld.parts[fld.data[i]]), "utf-8", "replace")
        try:
            raw = bytes(U2B[c] for c in piece)
        except KeyError:
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if len(text) == 1 and not text.isascii():
            single.setdefault(text, i)

    print(f"vocab {n}: {len(single)} distinct non-ASCII single-char tokens\n")
    seen = set()
    for label, chars in CATEGORIES:
        rows = []
        for ch in chars:
            if ch in seen:
                continue
            seen.add(ch)
            if ch in single:
                cost = len(ch.encode("utf-8"))
                kept = "KEPT" if ch in P1M_EXTRA else "  - "
                try:
                    name = unicodedata.name(ch)
                except ValueError:
                    name = "?"
                rows.append((kept, ch, cost, name))
        if rows:
            print(f"== {label} ==")
            for kept, ch, cost, name in rows:
                print(f"  {kept}  {ch!r:>8}  {cost}B  {name}")
            print()

    # anything in P1M_EXTRA the vocab cannot represent as one token
    absent = [c for c in P1M_EXTRA if c not in single]
    if absent:
        print("== in P1M_EXTRA but NOT a single token in this vocab ==")
        for c in sorted(absent):
            print(f"  {c!r}")


if __name__ == "__main__":
    main(sys.argv[1])
