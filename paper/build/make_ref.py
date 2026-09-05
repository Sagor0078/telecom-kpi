"""Build an IEEE Transactions-style reference.docx for pandoc."""
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import sys

OUT = sys.argv[1]
doc = Document()

# ---- page geometry: IEEE Trans, US Letter, two columns ----
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
sec.top_margin, sec.bottom_margin = Inches(0.75), Inches(1.0)
sec.left_margin, sec.right_margin = Inches(0.625), Inches(0.625)

sectPr = sec._sectPr
cols = sectPr.find(qn('w:cols'))
if cols is None:
    cols = OxmlElement('w:cols'); sectPr.append(cols)
cols.set(qn('w:num'), '2')
cols.set(qn('w:space'), str(int(0.25 * 1440)))   # 0.25in gutter
cols.set(qn('w:equalWidth'), '1')

def style(name, *, size=10, bold=False, italic=False, align='just',
          before=0, after=0, indent=0.0, left=0.0, hanging=0.0,
          smallcaps=False, base='Normal', keep_next=False):
    """Create or update a paragraph style."""
    styles = doc.styles
    try:
        st = styles[name]
    except KeyError:
        from docx.enum.style import WD_STYLE_TYPE
        st = styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
        st.base_style = styles[base]
    f = st.font
    f.name, f.size, f.bold, f.italic = 'Times New Roman', Pt(size), bold, italic
    f.color.rgb = RGBColor(0, 0, 0)
    rPr = st.element.get_or_add_rPr()
    rf = rPr.find(qn('w:rFonts'))
    if rf is None:
        rf = OxmlElement('w:rFonts'); rPr.append(rf)
    for a in ('w:ascii', 'w:hAnsi', 'w:cs', 'w:eastAsia'):
        rf.set(qn(a), 'Times New Roman')
    if smallcaps:
        sc = OxmlElement('w:smallCaps'); sc.set(qn('w:val'), '1'); rPr.append(sc)
    p = st.paragraph_format
    p.alignment = {'just': WD_ALIGN_PARAGRAPH.JUSTIFY,
                   'center': WD_ALIGN_PARAGRAPH.CENTER,
                   'left': WD_ALIGN_PARAGRAPH.LEFT}[align]
    p.space_before, p.space_after = Pt(before), Pt(after)
    p.first_line_indent = Inches(indent) if indent else None
    p.left_indent = Inches(left) if left else None
    if hanging:
        p.left_indent, p.first_line_indent = Inches(hanging), Inches(-hanging)
    p.line_spacing = 1.0
    p.widow_control = True
    p.keep_with_next = keep_next
    return st

# Normal drives everything pandoc doesn't style explicitly.
style('Normal', size=10, align='just', indent=0.2)
# IEEE section head: "I. INTRODUCTION" centered small caps.
style('Heading 1', size=10, align='center', before=12, after=6,
      smallcaps=True, keep_next=True)
# IEEE subsection: "A. Dataset" italic, flush left.
style('Heading 2', size=10, italic=True, align='left', before=8, after=3,
      keep_next=True)
style('Heading 3', size=10, italic=True, align='left', indent=0.2,
      before=6, after=2, keep_next=True)
# Pandoc body styles.
style('Body Text', size=10, align='just', indent=0.2)
style('First Paragraph', size=10, align='just', indent=0.0)
style('Compact', size=10, align='just', indent=0.2)
style('Image Caption', size=8, align='left', before=6, after=10)
style('Table Caption', size=8, align='left', before=8, after=3)
style('Bibliography', size=8, align='just', hanging=0.22, after=2)
# Front-matter styles (used by the assembler, not pandoc).
style('IEEE Title', size=24, align='center', before=0, after=8)
style('IEEE Author', size=11, align='center', after=4)
style('IEEE Affil', size=9, italic=True, align='center', after=14)
style('IEEE Abstract', size=9, bold=True, align='just', after=6, indent=0.2)
style('IEEE Terms', size=9, bold=True, italic=True, align='just', after=10,
      indent=0.2)

# Inline code (dataset field names, library parameters). Pandoc tags these
# runs "VerbatimChar"; without the style they fall back to body Times and the
# underscores read as prose.
from docx.enum.style import WD_STYLE_TYPE
vc = doc.styles.add_style('VerbatimChar', WD_STYLE_TYPE.CHARACTER)
vc.font.name = 'Courier New'
vc.font.size = Pt(8.5)
_rPr = vc.element.get_or_add_rPr()
_rf = OxmlElement('w:rFonts')
for _a in ('w:ascii', 'w:hAnsi', 'w:cs'):
    _rf.set(qn(_a), 'Courier New')
_rPr.append(_rf)

doc.save(OUT)
print("wrote", OUT)
