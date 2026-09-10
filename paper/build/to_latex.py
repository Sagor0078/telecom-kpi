"""Render the spliced Markdown body as an IEEEtran LaTeX manuscript.

Usage: to_latex.py <body.md> <frontmatter.md> <out.tex>

Pandoc does the prose, math and inline markup; everything IEEE-specific is
applied to its output here. Four constructs need rewriting because pandoc has
no way to know the target is a two-column IEEE journal class:

  * longtable          -> table/table* + tabular, since longtable cannot be
                          used in twocolumn mode at all.
  * bare includegraphics -> figure/figure* float with a numbered caption.
  * \\[ ... \\]          -> equation, so display equations are numbered.
  * "Section 5.3"      -> Section~\\ref{sec:5.3}, so IEEE section numerals are
                          produced by LaTeX rather than written by hand.

Figure and table numbers are taken from the caption text, not from LaTeX's
counters: the manuscript numbers figures after the notebook that produced
them, so document order and figure number deliberately disagree. Each float
sets its counter explicitly, which keeps the .tex and the .docx numbering
identical.
"""
import re
import sys
import subprocess
import pathlib

body_md, fm_md, out_tex = sys.argv[1], sys.argv[2], sys.argv[3]

# Whether a table spans both columns. Five columns is the .docx rule
# (assemble.py); the width estimate additionally catches the narrow-but-long
# tables, which a single column cannot hold either and which silently overrun
# into the neighbouring column rather than reporting an overfull box.
#
# CHAR_IN is calibrated against a compiled proof: a four-column table whose
# widest row carries 44 printable characters measured 3.95in, so at
# \footnotesize a character costs about 0.078in and each column boundary
# another 2\tabcolsep. COL_IN and TEXT_IN are IEEEtran's letterpaper column
# and text widths less a safety margin.
SPAN_MIN_COLS = 5
CHAR_IN = 0.078
GUTTER_IN = 2 * 6 / 72
COL_IN = 3.3
TEXT_IN = 6.9

DEFAULT_JOURNAL = 'IEEE Transactions on Network and Service Management'


def pandoc(text):
    """Markdown to a LaTeX fragment, one paragraph per line."""
    return subprocess.run(['pandoc', '-f', 'markdown', '-t', 'latex',
                           '--wrap=none'],
                          input=text, capture_output=True, text=True,
                          check=True).stdout


def open_brace_of(s, i):
    """Index of the brace opening the one at s[i]."""
    depth = 0
    while i >= 0:
        if s[i] == '}':
            depth += 1
        elif s[i] == '{':
            depth -= 1
            if depth == 0:
                return i
        i -= 1
    raise ValueError('unbalanced braces')


def close_brace(s, i):
    """Index of the brace closing the one at s[i]."""
    depth = 0
    while i < len(s):
        if s[i] == '{':
            depth += 1
        elif s[i] == '}':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError('unbalanced braces')


def plain_len(cell):
    """Rough printed width of a LaTeX cell, in characters."""
    s = re.sub(r'\\[a-zA-Z]+\s*', '', cell)
    return len(s.replace('{', '').replace('}', '').strip())


# ---------------------------------------------------------------- front matter
fields, key = {}, None
for line in pathlib.Path(fm_md).read_text().splitlines():
    m = re.match(r'^([A-Z][A-Z ]*):\s*(.*)$', line)
    if m:
        key = m.group(1).strip()
        fields[key] = m.group(2).strip()
    elif key:
        fields[key] = (fields[key] + ' ' + line.strip()).strip()

missing = {'TITLE', 'AUTHORS', 'AFFILIATION', 'ABSTRACT', 'INDEX TERMS'} - fields.keys()
if missing:
    sys.exit(f'error: frontmatter.md missing {sorted(missing)}')

# One pandoc pass for all five values; @@ is left alone by every reader.
SEP = '@@FIELD@@'
order = ['TITLE', 'AUTHORS', 'AFFILIATION', 'ABSTRACT', 'INDEX TERMS']
joined = f'\n\n{SEP}\n\n'.join(fields[k].replace("\\'", "'").replace('\\"', '"')
                               for k in order)
converted = [p.strip() for p in pandoc(joined).split(SEP)]
front = dict(zip(order, converted))

journal = fields.get('JOURNAL', DEFAULT_JOURNAL)
short_title = fields.get('SHORT TITLE') or front['TITLE'].split(':')[0]

# ---------------------------------------------------------------------- body
raw = pathlib.Path(body_md).read_text()
head, sep, refs_md = raw.partition('# References')
if not sep:
    sys.exit('error: no "# References" section')

# Escaped quotes and apostrophes came from the original .docx conversion.
# Unescaping them lets pandoc's smart-quote pass produce real typographic
# quotes instead of \textquotesingle.
head = head.replace("\\'", "'").replace('\\"', '"')
refs_md = refs_md.replace("\\'", "'").replace('\\"', '"')

# Numbered headings become plain headings carrying a section label; IEEEtran
# supplies the numeral and \ref reproduces it at every cross-reference.
head = re.sub(r'^# (\d+)\.\s*(.+)$',
              lambda m: f'# {m.group(2).strip()} {{#sec:{m.group(1)}}}',
              head, flags=re.M)
head = re.sub(r'^## (\d+)\.(\d+)\s*(.+)$',
              lambda m: f'## {m.group(3).strip()} {{#sec:{m.group(1)}.{m.group(2)}}}',
              head, flags=re.M)

tex = pandoc(head)

# ---- pandoc's hypertarget wrapper is redundant next to \label -------------
tex = re.sub(r'\\hypertarget\{[^}]*\}\{%\n(\\(?:sub)?section\{.*?\}\\label\{[^}]*\})\}',
             r'\1', tex)

# ---- display math --------------------------------------------------------
tex, n_eq = re.subn(r'^\\\[(.+)\\\]$',
                    lambda m: '\\begin{equation}\n' + m.group(1).strip() + '\n\\end{equation}',
                    tex, flags=re.M)

# ---- citations -----------------------------------------------------------
tex = re.sub(r'\{\[\}(\d+)\{\]\}', r'\\cite{ref\1}', tex)
while True:
    tex, n = re.subn(r'\\cite\{([^}]+)\},\s*\\cite\{([^}]+)\}', r'\\cite{\1,\2}', tex)
    if not n:
        break

# ---- figures -------------------------------------------------------------
FIG = re.compile(r'\\includegraphics\[width=([\d.]+)in[^\]]*\]\{([^}]+)\}\s*\n\s*\n\\emph\{Figure (\d+)\.\s*')


figs = []
for m in FIG.finditer(tex):
    open_brace = tex.rindex('{', m.start(), m.end())
    end = close_brace(tex, open_brace)
    caption = tex[m.end():end].strip()
    figs.append((m.start(), end + 1, float(m.group(1)), m.group(2),
                 int(m.group(3)), caption))

for start, end, width, path, num, caption in reversed(figs):
    wide = width >= 5.0
    env = 'figure*' if wide else 'figure'
    size = r'\textwidth' if wide else r'\columnwidth'
    tex = tex[:start] + (
        f'\\begin{{{env}}}[!t]\n'
        f'\\centering\n'
        f'\\includegraphics[width={size}]{{{path}}}\n'
        f'\\setcounter{{figure}}{{{num - 1}}}\n'
        f'\\caption{{{caption}}}\n'
        f'\\label{{fig:{num}}}\n'
        f'\\end{{{env}}}\n'
    ) + tex[end:]

# ---- tables --------------------------------------------------------------
# A header cell can be empty, so the content group has to be allowed to be
# empty too; pandoc emits the opening and closing tags on adjacent lines.
MINIPAGE = re.compile(
    r'\\begin\{minipage\}\[[bt]\]\{\\linewidth\}\\raggedright\s*(.*?)\s*\\end\{minipage\}',
    re.S)
# longtable scaffolding that is not part of any row.
NOISE = re.compile(r'\\(?:toprule|midrule|bottomrule|endhead|endfirsthead'
                   r'|endfoot|endlastfoot|noalign\{\})')


def split_row(row):
    """Split a LaTeX table row on top-level & only."""
    cells, depth, cur, esc = [], 0, '', False
    for ch in row:
        if esc:
            esc = False
        elif ch == '\\':
            esc = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
        elif ch == '&' and depth == 0:
            cells.append(cur.strip())
            cur = ''
            continue
        cur += ch
    cells.append(cur.strip())
    return cells


def rows_of(chunk):
    """Rows of a longtable chunk. A row may open with a control sequence
    (\\textbf in a header, \\cite in Table 1), so scaffolding is stripped by
    name rather than by guessing from the first character."""
    out = []
    for line in chunk.split('\\\\'):
        line = NOISE.sub('', line).strip()
        if line:
            out.append(split_row(line))
    return out


tables = []
pos = 0
while True:
    start = tex.find('\\begin{longtable}[]{', pos)
    if start < 0:
        break
    spec_open = tex.index('{', start + len('\\begin{longtable}[]'))
    spec_end = close_brace(tex, spec_open)
    spec = tex[spec_open + 1:spec_end]
    end = tex.index('\\end{longtable}', spec_end) + len('\\end{longtable}')
    inner = tex[spec_end + 1:end - len('\\end{longtable}')]

    header_chunk = MINIPAGE.sub(lambda m: m.group(1).strip(),
                                inner.split('\\midrule')[0])
    body_chunk = inner.split('\\endlastfoot', 1)[1] if '\\endlastfoot' in inner else ''

    header = rows_of(header_chunk)
    rows = rows_of(body_chunk)
    if not rows:
        sys.exit(f'error: table near offset {start} came out empty')

    # The caption is the \textbf{Table N. ...} paragraph directly above.
    cap_num, cap_text, cap_start = None, '', start
    before = tex[:start].rstrip()
    if before.endswith('}'):
        open_brace = open_brace_of(before, len(before) - 1)
        cmd = before.rfind('\\textbf', 0, open_brace)
        m = re.match(r'\\textbf\{Table (\d+)\.\s*', before[cmd:]) if cmd == open_brace - 7 else None
        if m:
            cap_num = int(m.group(1))
            cap_text = before[cmd + len(m.group(0)):-1].strip()
            cap_start = cmd

    tables.append((cap_start, end, spec, header, rows, cap_num, cap_text))
    pos = end

for cap_start, end, spec, header, rows, num, caption in reversed(tables):
    ncols = len(header[0]) if header else (len(rows[0]) if rows else 1)
    chars = max((sum(plain_len(c) for c in r) for r in header + rows), default=0)
    est = chars * CHAR_IN + (ncols - 1) * GUTTER_IN
    wide = ncols >= SPAN_MIN_COLS or est > COL_IN
    env = 'table*' if wide else 'table'
    if wide:
        spec = spec.replace('\\columnwidth', '\\textwidth')
    # A spec pandoc gave explicit p{} widths always fits, because those are
    # fractions of the enclosing width. A bare l/c/r spec takes its natural
    # width, so it is the only kind that can overrun.
    if 'p{' not in spec and est > (TEXT_IN if wide else COL_IN):
        print(f'  warning: table {num} estimated at {est:.1f}in in a '
              f'{TEXT_IN if wide else COL_IN}in {env}', file=sys.stderr)

    lines = [f'\\begin{{{env}}}[!t]', '\\centering']
    if num is not None:
        lines.append(f'\\setcounter{{table}}{{{num - 1}}}')
        lines.append(f'\\caption{{{caption}}}')
        lines.append(f'\\label{{tab:{num}}}')
    lines += ['\\footnotesize', f'\\begin{{tabular}}{{{spec}}}', '\\toprule']
    for r in header:
        lines.append(' & '.join(r) + ' \\\\')
    if header:
        lines.append('\\midrule')
    for r in rows:
        lines.append(' & '.join(r) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', f'\\end{{{env}}}']
    tex = tex[:cap_start] + '\n'.join(lines) + '\n' + tex[end:]

# ---- section cross-references -------------------------------------------
tex = re.sub(r'\bSections (\d+) and (\d+)\b',
             r'Sections~\\ref{sec:\1} and~\\ref{sec:\2}', tex)
tex = re.sub(r'\bSection (\d+)\.(\d+)', r'Section~\\ref{sec:\1.\2}', tex)
tex = re.sub(r'\bSection (\d+)\b', r'Section~\\ref{sec:\1}', tex)

defined = set(re.findall(r'\\label\{(sec:[\d.]+)\}', tex))
dangling = sorted(set(re.findall(r'\\ref\{(sec:[\d.]+)\}', tex)) - defined)
if dangling:
    sys.exit(f'error: cross-reference to undefined section(s) {dangling}')

# ------------------------------------------------------------------ references
refs_tex = pandoc(refs_md.lstrip())
entries = re.split(r'\{\[\}(\d+)\{\]\}', refs_tex)[1:]
if not entries:
    sys.exit('error: no numbered entries in the References section')
bib = []
for num, text in zip(entries[0::2], entries[1::2]):
    bib.append(f'\\bibitem{{ref{num}}}\n' + ' '.join(text.split()))

# ---------------------------------------------------------------------- output
PREAMBLE = r"""%% TrustNet-RCA, IEEE Transactions two-column manuscript.
%% GENERATED FILE. Edit paper/build/src/*.md and re-run paper/build/build_tex.sh.
%% Compile with pdflatex (IEEEtran ships in texlive-publishers):
%%     latexmk -pdf Trustworthy_RCA_Framework_Paper_IEEE.tex
\documentclass[journal]{IEEEtran}

\usepackage{iftex}
\ifPDFTeX
  \usepackage[T1]{fontenc}
  \usepackage[utf8]{inputenc}
\else
  \usepackage{fontspec}
\fi
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{array}
\usepackage{booktabs}
\usepackage{calc}          % \real{} in the generated column widths
\usepackage{textcomp}
\usepackage{url}

% The manuscript numbers figures and tables in arabic, matching the .docx.
\renewcommand{\figurename}{Figure}
\renewcommand{\tablename}{Table}
\renewcommand{\thetable}{\arabic{table}}

% This paper carries 12 figures and 13 tables against ~14 pages of prose, so
% the default float budget would push most of them past their discussion.
\setcounter{topnumber}{3}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{5}
\setcounter{dbltopnumber}{3}
\renewcommand{\topfraction}{0.92}
\renewcommand{\dbltopfraction}{0.92}
\renewcommand{\bottomfraction}{0.5}
\renewcommand{\textfraction}{0.07}
\renewcommand{\floatpagefraction}{0.75}
\renewcommand{\dblfloatpagefraction}{0.75}

\providecommand{\tightlist}{%
  \setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}

\begin{document}
"""

doc = [PREAMBLE]
doc.append('\\title{%s}\n' % front['TITLE'])
doc.append('\\author{%s%%\n\\thanks{%s}}\n' % (front['AUTHORS'], front['AFFILIATION']))
doc.append('\\markboth{%s}{%s}\n' % (journal, short_title))
doc.append('\\maketitle\n')
doc.append('\\begin{abstract}\n%s\n\\end{abstract}\n' % front['ABSTRACT'])
doc.append('\\begin{IEEEkeywords}\n%s\n\\end{IEEEkeywords}\n' % front['INDEX TERMS'])
doc.append('\\IEEEpeerreviewmaketitle\n')
doc.append(tex.strip() + '\n')
doc.append('\\begin{thebibliography}{%d}\n' % len(bib))
doc.append('\n\n'.join(bib) + '\n')
doc.append('\\end{thebibliography}\n')
doc.append('\\end{document}\n')

pathlib.Path(out_tex).write_text('\n'.join(doc))
print(f'  {len(figs)} figures, {len(tables)} tables, {n_eq} equations, '
      f'{len(bib)} references -> {out_tex}')
