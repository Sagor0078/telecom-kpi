"""Split the draft into IEEE front matter + body, renumbering to IEEE style."""
import re, json, sys

src = open(sys.argv[1]).read()
ROMAN = ['0','I','II','III','IV','V','VI','VII','VIII','IX','X','XI','XII']

split = re.search(r'^# 1\. Introduction', src, re.M)
front, body = src[:split.start()], src[split.start():]

# ---------- front matter ----------
bolds = re.findall(r'\*\*(.+?)\*\*', front, re.S)
italics = re.findall(r'^\*([^*\n]+?)\*\s*$', front, re.M)
def unescape(s):
    s = s.replace(r'\--', '—').replace(' -- ', ' — ')
    for ch in '~[]|*_$#<>"':
        s = s.replace('\\' + ch, ch)
    return s

def flat(s): return unescape(' '.join(s.split()))

title = flat(bolds[0].rstrip(':')) + ': ' + flat(bolds[1])
author = flat(italics[0]) if italics else 'Author Name(s) to be added'
affil  = flat(italics[1]) if len(italics) > 1 else 'Affiliation to be added'

abstract = flat(re.search(r'\*\*ABSTRACT\*\*\s*(.+?)\s*\*\*Keywords',
                          front, re.S).group(1))
keywords = flat(re.search(r'\*\*Keywords:?\*\*\s*(.+)$', front, re.S).group(1))

json.dump({'title': title, 'author': author, 'affil': affil,
           'abstract': abstract, 'keywords': keywords},
          open(sys.argv[3], 'w'), indent=2)

# ---------- body: IEEE heading numbering ----------
def h1(m):
    n, txt = int(m.group(1)), m.group(2).strip()
    return f'# {ROMAN[n]}. {txt}'

def h2(m):
    sub, txt = int(m.group(2)), m.group(3).strip()
    return f'## {chr(64 + sub)}. {txt}'

body = re.sub(r'^# (\d+)\.\s*(.+)$', h1, body, flags=re.M)
body = re.sub(r'^## (\d+)\.(\d+)\s*(.+)$', h2, body, flags=re.M)
# In-text cross-references follow the section numbering.
body = re.sub(r'\bSection (\d+)\b', lambda m: f'Section {ROMAN[int(m.group(1))]}', body)
body = re.sub(r'\bSections (\d+) and (\d+)\b',
              lambda m: f'Sections {ROMAN[int(m.group(1))]} and {ROMAN[int(m.group(2))]}', body)
# Escaped en-dashes from the docx round-trip.
body = body.replace(r'\--', '—').replace(' -- ', ' — ')
# Narrow figures fit one 3.5in column; originally-wide ones span both.
def fig_width(m):
    return '{width="7.0in"}' if float(m.group(1)) >= 5.0 else '{width="3.4in"}'
body = re.sub(r'\{width="([\d.]+)in"\s+height="[\d.]+in"\}', fig_width, body)
# References heading carries no numeral in IEEE.
body = body.replace('# References', '# References')

open(sys.argv[2], 'w').write(body)
print('sections:', len(re.findall(r'^# ', body, re.M)),
      '| subsections:', len(re.findall(r'^## ', body, re.M)))
