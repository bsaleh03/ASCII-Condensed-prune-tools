# tools-prune

Vocabulary pruning for byte-level BPE GGUF models.

A multilingual vocab is mostly dead weight for English and code work. Dropping
the unused tokens shrinks `token_embd.weight` and `output.weight`, which is
where a large fraction of a small quant's file size lives. Rows are gathered in
quantized space — no dequant/requant — so every surviving row is bit-identical
to the source.

## Tools

| | |
|---|---|
| `vocab_audit.py SRC.gguf OUT.json` | Bucket the vocab by Unicode script. Run first to see what is actually in there. |
| `scan_nonascii.py SRC.gguf` | List non-ASCII characters the vocab holds as single tokens, by category. Use it to decide a keep-set against the real vocab instead of from memory. |
| `prune_vocab.py SRC.gguf DST.gguf --policy P1M` | Do the prune. |
| `verify_prune.py SRC.gguf DST.gguf --policy P1M` | Check the result. Non-zero exit on any failure. |

## Policies

Each is defined per-character, which keeps the set closed under substring: every
merge producing a kept token has kept parents, so BPE reachability survives.

| | |
|---|---|
| `P1` | ASCII |
| `P2` | ASCII + Latin-1 / Latin-Extended-A |
| `P1M` | ASCII + math and typography symbols |
| `P1G` | P1M + Greek, box-drawing, status glyphs, units, currency |

`--keep-chars 'πλΩ'` adds characters; `--drop-chars` removes them from the
policy set. ASCII cannot be dropped — that would break substring closure and
byte fallback.

Byte-fallback tokens, specials, and partial-UTF-8 fragments are always kept, so
a pruned model can still represent any text; dropped characters just cost one
token per UTF-8 byte instead of one per character.

## Verifying

`verify_prune.py` always runs the structural checks — kept tokens exist in the
source and keep their order, specials and all 256 byte-fallback tokens survive,
merges reference only kept tokens and keep their priority, non-vocab tensors are
byte-identical, sampled vocab rows match their source rows.

Passing `--policy` adds the check the structural ones cannot make: that the
gather implemented the policy you asked for. A prune run with the wrong
`--policy` produces a perfectly consistent file containing the wrong tokens and
passes everything else, so this replays the keep-set construction against the
source and demands an exact match. Pass the same flags you pruned with.

## Requirements

Python 3, `numpy`, `gguf`. `regex` is optional and improves script detection in
`vocab_audit.py`.

The vocab must be byte-level BPE (`tokenizer.ggml.model` = `gpt2` or `bpe`) —
checked, and refused otherwise. An SPM vocab stores pieces as raw UTF-8 with a
U+2581 marker, so the GPT-2 byte reversal would silently produce wrong bytes
rather than fail. Nothing else is architecture-specific; metadata is copied key
by key.

## Notes

`P1M_EXTRA` is frozen: a model was built with it, and `verify_prune --policy
P1M` still has to reproduce that set exactly. New characters go in `P1G`.

Some kept characters are inert on any given model — nothing in the Qwen3.5
vocab contains U+221E or U+2211, for instance. Harmless, and they still apply to
vocabs that do carry them.
