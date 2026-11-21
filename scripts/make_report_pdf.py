import re
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, ListFlowable, ListItem, Preformatted, Image, PageBreak, KeepTogether
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "REPORT.md"
OUT = ROOT / "REPORT.pdf"
LOGO = ROOT / "assets" / "uni_bamberg_logo.png"

UBA_BLUE = colors.HexColor('#00467F')
UBA_LIGHT = colors.HexColor('#E8F1F8')
GREY = colors.HexColor('#444444')
LIGHT_GREY = colors.HexColor('#F5F7FA')

styles = {
    'cover_title': ParagraphStyle('CoverTitle', fontName='Helvetica-Bold', fontSize=24, leading=30, textColor=UBA_BLUE, alignment=TA_CENTER, spaceAfter=12),
    'cover_sub': ParagraphStyle('CoverSub', fontName='Helvetica', fontSize=12, leading=16, textColor=GREY, alignment=TA_CENTER, spaceAfter=6),
    'cover_meta': ParagraphStyle('CoverMeta', fontName='Helvetica', fontSize=10.5, leading=15, textColor=colors.black, alignment=TA_CENTER, spaceAfter=4),
    'h2': ParagraphStyle('H2', fontName='Helvetica-Bold', fontSize=12, leading=15, textColor=UBA_BLUE, spaceBefore=10, spaceAfter=5),
    'h3': ParagraphStyle('H3', fontName='Helvetica-Bold', fontSize=10, leading=13, textColor=colors.HexColor('#1A1A1A'), spaceBefore=6, spaceAfter=3),
    'body': ParagraphStyle('Body', fontName='Helvetica', fontSize=9.5, leading=13, spaceAfter=4, alignment=TA_JUSTIFY),
    'small': ParagraphStyle('Small', fontName='Helvetica', fontSize=9, leading=12, spaceAfter=2),
    'cell': ParagraphStyle('Cell', fontName='Helvetica', fontSize=7.2, leading=9),
    'cellh': ParagraphStyle('CellH', fontName='Helvetica-Bold', fontSize=7.2, leading=9, textColor=colors.white),
    'code': ParagraphStyle('Code', fontName='Courier', fontSize=7.5, leading=9.5, backColor=LIGHT_GREY, borderPadding=4, spaceAfter=4, leftIndent=6),
}


def inline_md(s):
    s = s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    s = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', s)
    s = re.sub(r'`(.+?)`', r'<font name="Courier" size="8">\1</font>', s)
    return s


def header_footer(c: canvas.Canvas, doc):
    c.saveState()
    w, h = A4
    c.setStrokeColor(colors.HexColor('#CCCCCC'))
    c.setLineWidth(0.5)
    c.line(25*mm, 15*mm, w - 25*mm, 15*mm)
    c.setFont('Helvetica', 7.5)
    c.setFillColor(colors.HexColor('#666666'))
    c.drawString(25*mm, 11*mm, 'University of Bamberg | GdI-Proj-B/M: LLMs and Linguistic Theory | Muhammad Baqir')
    c.drawRightString(w - 25*mm, 11*mm, f'Page {doc.page}')
    if doc.page > 1:
        c.setStrokeColor(UBA_BLUE)
        c.setLineWidth(1)
        c.line(25*mm, h - 18*mm, w - 25*mm, h - 18*mm)
    c.restoreState()


def build_cover(story):
    if LOGO.exists():
        img = Image(str(LOGO), width=120*mm, height=29*mm)
        img.hAlign = 'CENTER'
        story.append(Spacer(1, 20*mm))
        story.append(img)
    story.append(Spacer(1, 18*mm))
    story.append(Paragraph('LLMs and Linguistic Inferences', styles['cover_title']))
    story.append(Paragraph('Evaluating Large Language Models on Entailment, Presupposition, and Implicature', styles['cover_sub']))
    story.append(Spacer(1, 8*mm))
    # info box
    info = [
        ['Author', 'Muhammad Baqir (2184791)'],
        ['Program', 'M.Sc. International Software Systems Science'],
        ['Course', 'GdI-Proj-B/M: LLMs and Linguistic Theory (Projektmodul)'],
        ['Supervision', 'Dr. Luke Burke, University of Bamberg'],
        ['Semester', 'WS 2026/27'],
        ['Date', 'November 22, 2026'],
        ['Repository', 'https://github.com/Baqir110/llm_linguistic_inferences'],
    ]
    data = [[Paragraph(f'<b>{k}</b>', styles['small']), Paragraph(v, styles['small'])] for k, v in info]
    t = Table(data, colWidths=[45*mm, 105*mm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), UBA_LIGHT),
        ('BOX', (0,0), (-1,-1), 0.8, UBA_BLUE),
        ('INNERGRID', (0,0), (-1,-1), 0.3, colors.HexColor('#B0C4D8')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 10*mm))
    story.append(Paragraph('<i>This report summarizes the benchmark, evaluation pipeline, real model runs, analysis, and limitations. It is intended for academic project submission.</i>', ParagraphStyle('CoverNote', parent=styles['small'], alignment=TA_CENTER, textColor=GREY, leftIndent=15*mm, rightIndent=15*mm)))
    story.append(PageBreak())


def parse_markdown(md_text):
    story = []
    lines = md_text.splitlines()
    i = 0
    first_h1_seen = False
    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
            continue
        if line.startswith('```'):
            i += 1
            buf = []
            while i < len(lines) and not lines[i].startswith('```'):
                buf.append(lines[i])
                i += 1
            i += 1
            txt = '\n'.join(buf).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            story.append(Preformatted(txt, styles['code']))
            continue
        if line.startswith('# '):
            # skip first markdown H1 because cover already has title
            if not first_h1_seen:
                first_h1_seen = True
                i += 1
                continue
            story.append(Paragraph(inline_md(line[2:]), styles['h2']))
        elif line.startswith('## '):
            story.append(Paragraph(inline_md(line[3:]), styles['h2']))
        elif line.startswith('### '):
            story.append(Paragraph(inline_md(line[4:]), styles['h3']))
        elif line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                rows.append([c.strip() for c in lines[i].strip().strip('|').split('|')])
                i += 1
            if len(rows) >= 2 and all(set(c) <= set('-: ') for c in ''.join(rows[1])):
                rows = [rows[0]] + rows[2:]
            data = []
            for r_idx, row in enumerate(rows):
                st = styles['cellh'] if r_idx == 0 else styles['cell']
                data.append([Paragraph(inline_md(c), st) for c in row])
            n = max(1, len(data[0]))
            available = A4[0] - 50*mm
            t = Table(data, colWidths=[available / n] * n, repeatRows=1)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), UBA_BLUE),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, UBA_LIGHT]),
                ('GRID', (0,0), (-1,-1), 0.25, colors.HexColor('#9BB3C9')),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('PADDING', (0,0), (-1,-1), 3),
            ]))
            story.append(t)
            story.append(Spacer(1, 2*mm))
            continue
        elif line.startswith('- '):
            items = []
            while i < len(lines) and lines[i].startswith('- '):
                items.append(ListItem(Paragraph(inline_md(lines[i][2:].strip()), styles['small']), leftIndent=8))
                i += 1
            story.append(ListFlowable(items, bulletType='bullet', leftIndent=10, bulletFontSize=7, bulletColor=UBA_BLUE))
            continue
        elif re.match(r'^\d+\.\s', line.strip()):
            items = []
            while i < len(lines) and re.match(r'^\d+\.\s', lines[i].strip()):
                txt = re.sub(r'^\d+\.\s', '', lines[i].strip())
                items.append(ListItem(Paragraph(inline_md(txt), styles['small']), leftIndent=8))
                i += 1
            story.append(ListFlowable(items, bulletType='1', leftIndent=10, bulletFontSize=7))
            continue
        else:
            story.append(Paragraph(inline_md(line), styles['body']))
        i += 1
    return story


def main():
    md = SRC.read_text(encoding='utf-8')
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=25*mm, leftMargin=25*mm, topMargin=25*mm, bottomMargin=22*mm)
    story = []
    build_cover(story)
    story.extend(parse_markdown(md))
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print('ok', OUT)


if __name__ == '__main__':
    main()
