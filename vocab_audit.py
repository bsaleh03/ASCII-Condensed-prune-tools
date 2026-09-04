"""Phase 2b: bucket the Qwen3.8 vocab by Unicode script.

Reads tokenizer.ggml.tokens straight out of a GGUF, reverses the GPT-2
byte-to-unicode map to recover each token's raw bytes, then classifies.
"""
import sys, json, collections, unicodedata
from gguf import GGUFReader

try:
    import regex as re2
except ImportError:
    re2 = None


def utf8_console():
    """Make stdout/stderr able to carry the characters these tools discuss.

    The Windows console defaults to cp1252, so printing a Greek or math glyph
    raised UnicodeEncodeError and killed the run - fatal for tools whose whole
    output is non-ASCII characters. errors="replace" so an exotic console
    encoding degrades to '?' rather than aborting.

    Lives here because vocab_audit is the base module the others import.
    """
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass  # not a real stream (pipe, pytest capture) - nothing to fix


def bytes_to_unicode():
    """The GPT-2 map, verbatim."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


B2U = bytes_to_unicode()
U2B = {v: k for k, v in B2U.items()}

SCRIPT_PATTERNS = [
    ("Latin",    "Latin"),
    ("Han",      "Han"),
    ("Hiragana", "Hiragana"),
    ("Katakana", "Katakana"),
    ("Hangul",   "Hangul"),
    ("Cyrillic", "Cyrillic"),
    ("Arabic",   "Arabic"),
    ("Hebrew",   "Hebrew"),
    ("Greek",    "Greek"),
    ("Devanagari", "Devanagari"),
    ("Thai",     "Thai"),
    ("Bengali",  "Bengali"),
    ("Tamil",    "Tamil"),
    ("Telugu",   "Telugu"),
    ("Georgian", "Georgian"),
    ("Armenian", "Armenian"),
    ("Ethiopic", "Ethiopic"),
    ("Khmer",    "Khmer"),
    ("Myanmar",  "Myanmar"),
    ("Lao",      "Lao"),
    ("Sinhala",  "Sinhala"),
    ("Gujarati", "Gujarati"),
    ("Gurmukhi", "Gurmukhi"),
    ("Kannada",  "Kannada"),
    ("Malayalam", "Malayalam"),
    ("Oriya",    "Oriya"),
    ("Tibetan",  "Tibetan"),
    ("Mongolian", "Mongolian"),
    ("Syriac",   "Syriac"),
    ("Thaana",   "Thaana"),
    ("Cherokee", "Cherokee"),
]
if re2:
    COMPILED = [(n, re2.compile(r"\p{Script=%s}" % p)) for n, p in SCRIPT_PATTERNS]
else:
    COMPILED = []


def script_of_char(ch):
    """Best-effort script name for one character."""
    if ch.isascii():
        if ch.isalpha():
            return "Latin"
        if ch.isdigit():
            return "Digit"
        if ch.isspace():
            return "Space"
        return "ASCII-punct"
    for name, rx in COMPILED:
        if rx.match(ch):
            return name
    try:
        uname = unicodedata.name(ch)
    except ValueError:
        return "Unnamed"
    for name, _ in SCRIPT_PATTERNS:
        if uname.startswith(name.upper()):
            return name
    if uname.startswith("CJK"):
        return "Han"
    cat = unicodedata.category(ch)
    if cat.startswith("P") or cat.startswith("S"):
        return "Symbol/Punct"
    if cat.startswith("N"):
        return "Digit"
    return "Other"


def classify(raw: bytes, piece: str):
    """Return (bucket, decoded_text_or_None)."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return "PARTIAL-UTF8", None
    if text == "":
        return "EMPTY", text
    scripts = {script_of_char(c) for c in text}
    letters = scripts - {"Space", "Digit", "ASCII-punct", "Symbol/Punct", "Other", "Unnamed"}
    if not letters:
        if scripts <= {"Space", "Digit", "ASCII-punct"}:
            return "ASCII-nonalpha", text
        return "Symbol/Other", text
    if letters == {"Latin"}:
        return "Latin", text
    if len(letters) == 1:
        return next(iter(letters)), text
    return "Mixed:" + "+".join(sorted(letters)), text


def main(path, out_json):
    utf8_console()
    r = GGUFReader(path)
    fld = r.fields["tokenizer.ggml.tokens"]
    n = len(fld.data)
    print(f"vocab size: {n}")

    buckets = collections.Counter()
    rows = []
    n_partial = 0
    for i in range(n):
        piece = str(bytes(fld.parts[fld.data[i]]), encoding="utf-8", errors="replace")
        # reverse the byte-level BPE alphabet
        try:
            raw = bytes(U2B[c] for c in piece)
        except KeyError:
            # token is not in the byte-level alphabet (special token, etc.)
            raw = piece.encode("utf-8")
            bucket = "SPECIAL-or-raw"
            buckets[bucket] += 1
            rows.append((i, bucket, piece))
            continue
        bucket, text = classify(raw, piece)
        if bucket == "PARTIAL-UTF8":
            n_partial += 1
        buckets[bucket] += 1
        rows.append((i, bucket, text if text is not None else piece))

    print(f"partial-UTF8 tokens: {n_partial}")
    print()
    print(f"{'bucket':28s} {'count':>8s}  {'pct':>6s}")
    for b, c in buckets.most_common(45):
        print(f"{b:28s} {c:8d}  {100*c/n:5.2f}%")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"vocab_size": n,
                   "buckets": dict(buckets),
                   "token_bucket": [b for _, b, _ in rows]}, f)
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
