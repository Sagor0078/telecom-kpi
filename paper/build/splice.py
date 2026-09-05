"""Build the revised manuscript from the original draft plus new sections."""
import re, sys, pathlib

sp = pathlib.Path(sys.argv[1])
out_path = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else sp / 'paper_v2.md'
src = (sp / 'paper.md').read_text()

def load(n):
    # Section 6 is authored in four parts; a blank line must separate them or
    # the following heading is absorbed into the preceding paragraph.
    if n == 'new_sec6.md':
        parts = [(sp / f'new_sec6{c}.md').read_text().strip() for c in 'abcd']
        return '\n\n'.join(parts)
    return (sp / n).read_text().strip()

edits = []
def sub(pattern, repl, label, count=1, flags=re.S):
    global src
    new, n = re.subn(pattern, lambda m: repl, src, count=count, flags=flags)
    edits.append((label, n))
    src = new

# 1. Abstract -------------------------------------------------------------
sub(r'(\*\*ABSTRACT\*\*\n\n).*?(\n\n\*\*Keywords)',
    '**ABSTRACT**\n\n' + load('new_abstract.txt') + '\n\n**Keywords',
    'abstract')

# 2. Keywords -------------------------------------------------------------
sub(r'trustworthy AI; root-cause analysis; AIOps; conformal prediction; causal\ninference; explainable AI; telecom network management; time-series\nanomaly prediction',
    'trustworthy AI; root-cause analysis; AIOps; 5G RAN; conformal '
    'prediction; causal inference; explainable AI; explanation faithfulness; '
    'telecom network management; time-series anomaly prediction',
    'keywords')

# 3. Introduction: contributions ------------------------------------------
sub(r'This paper makes three contributions\..*?generalizable\s+evidence\.',
    """This paper makes three contributions. First, we propose
**TrustNet-RCA**, a five-component framework \\-- predict, attribute,
causally filter, calibrate, explain \\-- that makes each of these questions
an explicit, separately-evaluable pipeline stage rather than an implicit
property of one end-to-end model. Second, we implement it with standard,
off-the-shelf methods at each stage and evaluate it under one identical
method on three public datasets with complementary jobs: TelecomTS \\[22\\],
a 5G testbed dataset with 18 named PHY/MAC/network-layer KPIs, as the
primary 5G RAN benchmark; RCAEval \\[5\\], with annotated microservice
failures, for service-management and RCA rigor; and the Server Machine
Dataset (SMD) \\[7\\] strictly as an out-of-domain generalization check.
Running one method across three domains is what allows the paper to
separate what is a property of the framework from what is a property of a
dataset. Third, we report what we found rather than what we hoped, and two
of the most useful results are negative: attention models lose to
gradient-boosted trees on every leg at 4-82x the training cost, and none of
the confidence or explanation-quality signals the framework produces
distinguishes a correct root-cause localization from an incorrect one. We
define the concrete experimental programme (Section 9) required to elevate
these findings to generalizable evidence.""",
    'intro-contributions')

# 4. Introduction: roadmap sentence ---------------------------------------
sub(r'Section 5\ndetails the experimental setup\. Section 6 reports results for each\ncomponent\.',
    'Section 5 details the three-dataset experimental setup. Section 6 '
    'reports results for each component across all three legs.',
    'intro-roadmap')

# 5. Section 4: name the framework ----------------------------------------
sub(r'The framework decomposes the problem in Section 1 into five components,',
    'TrustNet-RCA decomposes the problem in Section 1 into five components,',
    'sec4-name')
sub(r'\*Figure 1\. Proposed five-component framework\.',
    '*Figure 1. The TrustNet-RCA five-component framework and its '
    'three-leg evaluation architecture.',
    'fig1-caption')
# Figure 1 becomes the regenerated three-leg architecture diagram.
sub(r'!\[\]\(\./media/image1\.png\)\{width="5\.8in" height="[\d.]+in"\}',
    '![](figures/fig1_framework.png){width="7.0in" height="4.1in"}',
    'fig1-image')

# 6-9. Whole-section replacements -----------------------------------------
for num, name, fname in ((5, 'Experimental Setup', 'new_sec5.md'),
                         (6, 'Results', 'new_sec6.md'),
                         (7, 'Discussion', 'new_sec7.md'),
                         (8, 'Limitations and Threats to Validity', 'new_sec8.md'),
                         (9, 'Practical Implications and Future Work', 'new_sec9.md'),
                         (10, 'Conclusion', 'new_sec10.md')):
    nxt = r'\n# %d\. ' % (num + 1) if num < 10 else r'\n# References'
    sub(r'\n# %d\. %s\n.*?(?=%s)' % (num, re.escape(name), nxt),
        '\n' + load(fname) + '\n',
        f'section-{num}')

sub(r"the gap this paper\\'s\nproof of concept targets\.",
    "the gap this paper targets.", 'relwork-scope')

# 10. New reference -------------------------------------------------------
ref22 = ('\n\n\\[22\\] S. Feng et al., "TelecomTS: A multimodal observability '
         'dataset for time series reasoning in telecom networks," '
         'arXiv:2510.06063, 2025.\n')
assert src.rstrip().endswith('.'), 'unexpected document tail'
src = src.rstrip() + ref22
edits.append(('reference-22', 1))

out_path.write_text(src)
for label, n in edits:
    print(f'{"OK " if n else "MISS"} {label} ({n})')
