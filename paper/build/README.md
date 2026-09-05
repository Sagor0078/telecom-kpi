# IEEE DOCX build pipeline

Rebuilds `../Trustworthy_RCA_Framework_Paper_IEEE.docx` — the IEEE
Transactions two-column manuscript for TNSM submission — from Markdown.

```bash
./build.sh                 # writes ../Trustworthy_RCA_Framework_Paper_IEEE.docx
./build.sh /tmp/draft.docx # or somewhere else
```

Requires `pandoc` and `python-docx`. No LaTeX toolchain.

## Why a pipeline and not just a .docx

The IEEE layout needs three things a plain conversion will not do: a
full-width title block over a two-column body, wide tables and figures that
span both columns, and IEEE section numbering (`I.` / `A.`). Each is a
section-properties or style manipulation applied after conversion.

## Editing the paper

**Edit the Markdown in `src/`, then re-run `./build.sh`.** Edits made
directly to the generated `.docx` are lost on the next build.

Every section is its own file. `splice.py` concatenates them in filename
order, so the numeric prefixes are what determine sequence — rename to
reorder, and keep the prefix in step with the section number.

| File | Section |
|---|---|
| `frontmatter.md` | Title, authors, affiliation, abstract, index terms |
| `01_introduction.md` | §I |
| `02_related_work.md` | §II (includes Table 1) |
| `03_problem_formulation.md` | §III |
| `04_framework.md` | §IV (includes Figure 1) |
| `05_experimental_setup.md` | §V — three-dataset architecture |
| `06a_results_early_warning.md` | §VI.A–B — early warning, calibration |
| `06b_results_attribution_causal.md` | §VI.C–D — attribution, causal filter |
| `06c_results_explanation_uncertainty.md` | §VI.E–G — Component 4, uncertainty, lead time |
| `06d_results_graph_view.md` | §VI.H — topology-based ranking |
| `07_discussion.md` | §VII |
| `08_limitations.md` | §VIII |
| `09_future_work.md` | §IX |
| `10_conclusion.md` | §X |
| `11_references.md` | References |

### Conventions

`frontmatter.md` is `KEY: value`, one field per key, values may wrap across
lines. All five keys are required.

Write headings as `# 5. Title` and `## 5.1 Subtitle` using arabic numerals.
`transform.py` converts them to `V.` and `A.`, and rewrites `Section N`
cross-references to match — so never write Roman numerals by hand, or the
numbering will drift out of step with the text.

Reference figures as `figures/figN_name.png`, resolved against `paper/`. A
figure declared wider than 5 inches becomes a full-width span; anything else
sits in one column. Figures are regenerated separately by
`../../source/src/figures.py`.

Tables with 5 or more columns are automatically spanned full-width and set
at 8pt. Keep first-column labels short — a long label in a many-column table
wraps badly even at full width.

### Math and code

Write math as LaTeX: `$\hat{y}_t$` inline, `$$...$$` for display. Pandoc
converts it to native Word equations (OMML), so Greek renders as symbols and
subscripts are real subscripts. **Never write math as ASCII prose**
(`phi_i`, `alpha`, `y_hat_t`) — that was the pre-existing style and it is
not IEEE.

Display equations are numbered automatically in document order, centred with
the number at the right margin. Numbering comes from position, so inserting
an equation renumbers everything after it — check any in-text references to
equation numbers after adding one.

Keep display equations narrow enough for a 3.5in column. Factor a long
expression into two equations rather than letting it run into the gutter;
`$$\phi_i(x) = \sum ... w_S[...]$$` plus a separate `$$w_S = ...$$` is the
pattern used in §III.

Dataset field names and library parameters go in backticks — `` `affected_kpis` ``,
`` `scale_pos_weight` `` — which sets them in Courier via the `VerbatimChar`
style. Metric names that read as prose should be prose ("rank stability
$\rho$", not `` `stability_rho` ``).

### Previewing math

`libreoffice --headless --convert-to pdf` **mis-renders OMML**: closing `]`
appears as `)`, `\setminus` and some delimiters show as `¿`. These are
LibreOffice import bugs, not defects in the document — the underlying XML
carries the correct `begChr`/`endChr`. Use the PDF to check layout, column
fit and numbering, but **open the .docx in Word to check the equations
themselves**.

## Stages

| Script | Does |
|---|---|
| `make_ref.py` | Builds the pandoc reference doc: IEEE styles (Times 10pt justified, small-caps section heads, 8pt references with hanging indent) plus page geometry — US Letter, 0.625" sides, two columns with a 0.25" gutter. |
| `splice.py` | Concatenates `src/*.md` in filename order and **validates**: sections numbered 1..N with no gaps, subsections belonging to their parent and lettered without gaps, References last, every referenced figure present. Any failure aborts the build with a specific message. |
| `transform.py` | Parses `frontmatter.md`, renumbers headings and cross-references to IEEE style, sets figure widths, and normalises dashes and escapes. |
| `assemble.py` | Prepends the single-column title block, wraps wide tables and figures in full-width sections, sets tables to 8pt autofit. |

## Verifying a build

`libreoffice --headless --convert-to pdf` renders it for checking. Confirm
the page count, that figure and table numbering has no gaps, and that no
stray `\` escapes or literal `##` survive into the text.
