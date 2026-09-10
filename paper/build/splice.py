"""Concatenate the manuscript sections, in filename order, into one Markdown body.

Section files are plain Markdown; the only rule is that a blank line must
separate them, or pandoc absorbs a leading heading into the preceding
paragraph. Numbering is validated here rather than trusted, because a gap or
a duplicate silently produces wrong cross-references downstream.
"""
import re
import sys
import pathlib

src_dir = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])

parts = sorted(p for p in src_dir.glob('*.md') if p.name != 'frontmatter.md')
if not parts:
    sys.exit(f'error: no section files in {src_dir}')

body = '\n\n'.join(p.read_text().strip() for p in parts) + '\n'

# ---- validation ---------------------------------------------------------
problems = []

secs = re.findall(r'^# (\d+)\.\s*(.+)$', body, re.M)
nums = [int(n) for n, _ in secs]
if nums != list(range(1, len(nums) + 1)):
    problems.append(f'numbered sections are {nums}, expected 1..{len(nums)}')

if not re.search(r'^# References\s*$', body, re.M):
    problems.append('no "# References" section')
elif body.rindex('# References') < body.rindex('\n# %d.' % nums[-1]):
    problems.append('"# References" must come last')

# Subsections must belong to their parent and run A, B, C... without gaps.
for i, (num, _) in enumerate(secs):
    start = body.index(f'# {num}. {secs[i][1]}')
    end = body.index(f'# {secs[i+1][0]}. {secs[i+1][1]}') if i + 1 < len(secs) else len(body)
    subs = re.findall(r'^## (\d+)\.(\d+)\s', body[start:end], re.M)
    wrong = {p for p, _ in subs} - {num}
    if wrong:
        problems.append(f'section {num} contains subsections numbered {sorted(wrong)}')
    idx = [int(s) for _, s in subs]
    if idx != list(range(1, len(idx) + 1)):
        problems.append(f'section {num} subsections are {idx}, expected 1..{len(idx)}')

# Figure paths are written relative to paper/, which holds this script's
# directory. Anchoring to __file__ keeps the check correct wherever src lives.
paper_dir = pathlib.Path(__file__).resolve().parent.parent
for ref in re.findall(r'!\[\]\(([^)]+)\)', body):
    rel = ref.split('{')[0]
    if not (paper_dir / rel).exists():
        problems.append(f'missing figure: {rel}')

# A soft line break after a hyphen is rendered as a space by every pandoc
# target, so "root-\ncause" silently ships as "root- cause". Reject rather
# than join: a line-final hyphen is occasionally a deliberate suspended
# hyphen ("pre- and post-onset"), and only the author can tell them apart.
# frontmatter.md is checked here too even though it is not part of the body.
for path in sorted(src_dir.glob('*.md')):
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if (line.endswith('-') and not line.lstrip().startswith('|')
                and not set(line.strip()) <= set('-: ')):
            problems.append(f'{path.name}:{n}: line ends in a hyphen, which '
                            f'renders as "{line.split()[-1]} <next word>"')

# Em dashes are not used in this paper; commas, colons, parentheses or a
# sentence break carry the same joins. Reject them rather than converting.
for m in re.finditer(r'\\--| -- |\u2014', body):
    ctx = ' '.join(body[max(0, m.start() - 45):m.end() + 45].split())
    problems.append(f'em dash: ...{ctx}...')

if problems:
    sys.exit('error: ' + '\nerror: '.join(problems))

out.write_text(body)
print(f'  {len(parts)} sections, {len(secs)} numbered + references, '
      f'{len(re.findall(r"^## ", body, re.M))} subsections')
