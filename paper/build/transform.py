"""Apply IEEE numbering to the body and parse the front matter.

Usage: transform.py <body.md> <frontmatter.md> <out_body.md> <out_front.json>
"""
import re
import sys
import json
import pathlib

ROMAN = ['0', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII']

body = pathlib.Path(sys.argv[1]).read_text()
fm_raw = pathlib.Path(sys.argv[2]).read_text()


def unescape(s):
    s = s.replace(r'\--', '—').replace(' -- ', ' — ')
    for ch in '~[]|*_$#<>"':
        s = s.replace('\\' + ch, ch)
    return s


# ---- front matter: KEY: value, values may span lines ---------------------
fields, key = {}, None
for line in fm_raw.splitlines():
    m = re.match(r'^([A-Z][A-Z ]*):\s*(.*)$', line)
    if m:
        key = m.group(1).strip()
        fields[key] = m.group(2).strip()
    elif key:
        fields[key] = (fields[key] + ' ' + line.strip()).strip()

missing = {'TITLE', 'AUTHORS', 'AFFILIATION', 'ABSTRACT', 'INDEX TERMS'} - fields.keys()
if missing:
    sys.exit(f'error: frontmatter.md missing {sorted(missing)}')

json.dump({'title': unescape(fields['TITLE']),
           'author': unescape(fields['AUTHORS']),
           'affil': unescape(fields['AFFILIATION']),
           'abstract': unescape(fields['ABSTRACT']),
           'keywords': unescape(fields['INDEX TERMS'])},
          open(sys.argv[4], 'w'), indent=2)

# ---- body: IEEE heading numbering ---------------------------------------
body = re.sub(r'^# (\d+)\.\s*(.+)$',
              lambda m: f'# {ROMAN[int(m.group(1))]}. {m.group(2).strip()}',
              body, flags=re.M)
body = re.sub(r'^## (\d+)\.(\d+)\s*(.+)$',
              lambda m: f'## {chr(64 + int(m.group(2)))}. {m.group(3).strip()}',
              body, flags=re.M)
# In-text cross-references follow the section numbering.
body = re.sub(r'\bSection (\d+)\b',
              lambda m: f'Section {ROMAN[int(m.group(1))]}', body)
body = re.sub(r'\bSections (\d+) and (\d+)\b',
              lambda m: f'Sections {ROMAN[int(m.group(1))]} and {ROMAN[int(m.group(2))]}',
              body)
body = body.replace(r'\--', '—').replace(' -- ', ' — ')
# Figures wider than one column are placed as full-width spans by assemble.py.
body = re.sub(r'\{width="([\d.]+)in"\s+height="[\d.]+in"\}',
              lambda m: '{width="7.0in"}' if float(m.group(1)) >= 5.0 else '{width="3.4in"}',
              body)

pathlib.Path(sys.argv[3]).write_text(body)
print(f'  {len(re.findall(r"^# ", body, re.M))} sections, '
      f'{len(re.findall(r"^## ", body, re.M))} subsections')
