# IEEE DOCX build pipeline

Rebuilds `../Trustworthy_RCA_Framework_Paper_IEEE.docx` — the IEEE
Transactions two-column manuscript for TNSM submission — from Markdown
sources.

```bash
./build.sh                 # writes ../Trustworthy_RCA_Framework_Paper_IEEE.docx
./build.sh /tmp/draft.docx # or somewhere else
```

Requires `pandoc` and `python-docx`. No LaTeX toolchain.

## Why a pipeline and not just a .docx

The IEEE layout needs three things Word will not do from a plain conversion:
a full-width title block over a two-column body, wide tables and figures
that span both columns, and IEEE section numbering (`I.` / `A.`). Each is a
section-properties or style manipulation applied after conversion.

## Editing the paper

**Edit the Markdown in `src/`, then re-run `./build.sh`.** Edits made
directly to the generated `.docx` are lost on the next build.

| File | Contents |
|---|---|
| `src/paper.md` | The original draft, extracted from the pre-revision `.docx`. Treated as read-only input. |
| `src/new_abstract.txt` | Abstract body (no `Abstract—` prefix; the assembler adds it). |
| `src/new_sec5.md` | §V Experimental Setup — three-dataset architecture. |
| `src/new_sec6{a,b,c,d}.md` | §VI Results, in four parts, concatenated in order. `a` = early warning + calibration, `b` = attribution + causal, `c` = Component 4 + uncertainty + lead time, `d` = graph view. |
| `src/new_sec7.md` … `new_sec10.md` | §VII Discussion, §VIII Limitations, §IX Future Work, §X Conclusion. |
| `src/title_override.txt` | The paper title. One line. |

Sections I–IV and the reference list still live inside `src/paper.md` and are
patched by `splice.py`; to change them, edit the corresponding `sub(...)` call
there or move the section into its own file following the existing pattern.

Write headings as `# 5. Title` and `## 5.1 Subtitle`. `transform.py` converts
them to `V.` and `A.` — do not write Roman numerals by hand, or cross-section
numbering will drift.

Reference figures as `figures/figN_name.png` (resolved against `paper/`).
A figure wider than 5 inches is placed as a full-width span; anything else
sits in one column. Figures are regenerated separately by
`../../source/src/figures.py`.

## Stages

| Script | Does |
|---|---|
| `make_ref.py` | Builds the pandoc reference doc: IEEE styles (Times 10pt justified, small-caps section heads, 8pt references with hanging indent) plus page geometry — US Letter, 0.625" sides, two columns with a 0.25" gutter. |
| `splice.py` | Assembles the revised manuscript from `src/paper.md` plus the replacement sections. Prints an `OK`/`MISS` line per edit — **a `MISS` means an anchor no longer matches and that edit silently did not apply.** |
| `transform.py` | Splits front matter from body, renumbers headings and `Section N` cross-references to IEEE style, sets figure widths, and unescapes the docx round-trip artifacts. |
| `assemble.py` | Prepends the single-column title block, wraps wide tables/figures in full-width sections, and sets tables to 8pt autofit. |

## Verifying a build

`libreoffice --headless --convert-to pdf` renders it for checking. Confirm
page count, that figure and table numbering has no gaps, and that no stray
`\\` escapes or literal `##` survive into the text.
