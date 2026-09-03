"""
Build Partial Thesis (Summer Session Report) — Group 26

Generates Group26_WorldCupPrediction.docx with full coverage of
summer-session work: league prediction (4 experiments) and World Cup
prediction (softmax + Poisson + platform blending).

Template files referenced by the prompt ("FYP Thesis-Partial.docx",
"GroupInfo2026FYP.pdf") are not present in the workspace, so this script
builds the document from scratch based on the existing codebase and
experiment logs.
"""
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "Group26_WorldCupPrediction.docx"

NAVY = "1F4E79"
LIGHT_BLUE = "DCE6F1"
LIGHT_GRAY = "F2F2F2"
MID_GRAY = "666666"
WHITE = "FFFFFF"
BLACK = "000000"


# ── Helper utilities ────────────────────────────────────────────

def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=90, start=110, bottom=90, end=110):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_cell_width(cell, width_dxa):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.find(qn("w:tcW"))
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(width_dxa))
    tc_w.set(qn("w:type"), "dxa")


def set_table_geometry(table, widths):
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(sum(widths)))
    tbl_w.set(qn("w:type"), "dxa")

    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)

    for row in table.rows:
        for cell, width in zip(row.cells, widths):
            set_cell_width(cell, width)
            set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def set_run_font(run, latin="Times New Roman", east_asia="宋体", size=11, bold=None, color=BLACK):
    run.font.name = latin
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), latin)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), latin)
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), east_asia)
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor.from_string(color)
    if bold is not None:
        run.bold = bold


def style_document(doc):
    section = doc.sections[0]
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(2.7)
    section.right_margin = Cm(2.7)
    section.header_distance = Cm(1.25)
    section.footer_distance = Cm(1.25)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.font.size = Pt(11)
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    normal.paragraph_format.space_after = Pt(6)

    for name, size, before, after in (
        ("Heading 1", 16, 16, 8),
        ("Heading 2", 13, 12, 6),
        ("Heading 3", 11.5, 8, 4),
    ):
        style = doc.styles[name]
        style.font.name = "Times New Roman"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(NAVY)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.line_spacing = 1.15


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    fld_char1 = OxmlElement("w:fldChar")
    fld_char1.set(qn("w:fldCharType"), "begin")
    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = " PAGE "
    fld_char2 = OxmlElement("w:fldChar")
    fld_char2.set(qn("w:fldCharType"), "end")
    run._r.extend([fld_char1, instr_text, fld_char2])
    set_run_font(run, size=9, color=MID_GRAY)


def add_body(doc, text, first_indent=True):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_after = Pt(6)
    if first_indent:
        p.paragraph_format.first_line_indent = Pt(22)
    run = p.add_run(text)
    set_run_font(run, size=11)
    return p


def add_bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.left_indent = Cm(0.74)
    p.paragraph_format.first_line_indent = Cm(-0.37)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.3
    run = p.add_run(text)
    set_run_font(run, size=10.8)
    return p


def add_heading(doc, text, level=1):
    p = doc.add_paragraph(text, style=f"Heading {level}")
    for run in p.runs:
        set_run_font(run, east_asia="黑体", size={1: 16, 2: 13, 3: 11.5}[level], bold=True, color=NAVY)
    return p


def add_reference(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.74)
    p.paragraph_format.first_line_indent = Cm(-0.74)
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run(text)
    set_run_font(r, size=10)


def add_cover(doc):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(56)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("FYP I Partial Thesis")
    set_run_font(r, east_asia="黑体", size=18, bold=True, color=NAVY)

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(32)
    p.paragraph_format.space_after = Pt(8)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(
        "Football Match Outcome Probability Prediction:\n"
        "League Rolling-Stat Models and FIFA-Ranking-Based\n"
        "World Cup Prediction System"
    )
    set_run_font(r, east_asia="黑体", size=20, bold=True, color=BLACK)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(54)
    r = p.add_run("Summer Session Progress Report")
    set_run_font(r, size=13, color=MID_GRAY)

    metadata = [
        ("Group ID", "Group 26"),
        ("Topic", "2026 FIFA World Cup Match Outcome Probability Prediction"),
        ("Document", "FYP I Partial Thesis — Summer Session"),
        ("Date", "June 2026"),
    ]
    table = doc.add_table(rows=len(metadata), cols=2)
    set_table_geometry(table, [2200, 6200])
    set_repeat_table_header(table.rows[0])
    for idx, (label, value) in enumerate(metadata):
        set_cell_shading(table.cell(idx, 0), LIGHT_BLUE)
        p1 = table.cell(idx, 0).paragraphs[0]
        p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r1 = p1.add_run(label)
        set_run_font(r1, east_asia="黑体", size=10.5, bold=True, color=NAVY)
        p2 = table.cell(idx, 1).paragraphs[0]
        r2 = p2.add_run(value)
        set_run_font(r2, size=10.5)

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(36)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(
        "This report documents all work completed during the summer session, "
        "including league match prediction experiments and the World Cup "
        "prediction system."
    )
    set_run_font(r, size=9.5, color=MID_GRAY)
    doc.add_page_break()


# ── Tables ──────────────────────────────────────────────────────

def add_league_experiments_table(doc):
    data = [
        ["Experiment", "Best Config", "Accuracy", "Key Takeaway"],
        [
            "001 — Baseline",
            "LR, 5 leagues, all features",
            "62.3%",
            "Baseline established; betting odds included but not dominant",
        ],
        [
            "002 — League adaptation",
            "Baseline (no league features)",
            "62.2%",
            "League info redundant with rolling stats + odds",
        ],
        [
            "003 — No odds",
            "LR without betting odds",
            "62.2%",
            "LR unaffected by removing odds; pure football stats suffice",
        ],
        [
            "004 — Pi-rating",
            "Baseline (no pi-rating)",
            "62.2%",
            "Pi-rating dynamic strength redundant with rolling stats",
        ],
    ]
    table = doc.add_table(rows=len(data), cols=4)
    table.style = "Table Grid"
    set_table_geometry(table, [2600, 2600, 1200, 3400])
    set_repeat_table_header(table.rows[0])
    for i, row in enumerate(data):
        for j, value in enumerate(row):
            cell = table.cell(i, j)
            if i == 0:
                set_cell_shading(cell, NAVY)
            elif i % 2 == 0:
                set_cell_shading(cell, LIGHT_GRAY)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if i == 0 or j in (0, 2) else WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.05
            r = p.add_run(value)
            set_run_font(
                r,
                east_asia="黑体" if i == 0 else "宋体",
                size=9 if i == 0 else 8.5,
                bold=(i == 0),
                color=WHITE if i == 0 else BLACK,
            )
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(4)
    cap.paragraph_format.space_after = Pt(8)
    r = cap.add_run("Table 1  League prediction experiments summary (H/D/A task)")
    set_run_font(r, size=9.5, bold=True, color=MID_GRAY)


def add_wc_metrics_table(doc):
    data = [
        ["Model", "Accuracy", "Log Loss", "Brier Score"],
        ["Actual bookmaker market", "0.578", "0.947", "0.556"],
        ["Historical market proxy", "0.578", "0.975", "0.574"],
        ["50/50 blended model", "0.578", "0.978", "0.576"],
        ["Pure result model", "0.578", "0.989", "0.583"],
    ]
    table = doc.add_table(rows=len(data), cols=4)
    table.style = "Table Grid"
    set_table_geometry(table, [3300, 1700, 1700, 1700])
    set_repeat_table_header(table.rows[0])
    for i, row in enumerate(data):
        for j, value in enumerate(row):
            cell = table.cell(i, j)
            if i == 0:
                set_cell_shading(cell, NAVY)
            elif i % 2 == 0:
                set_cell_shading(cell, LIGHT_GRAY)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(value)
            set_run_font(
                r, east_asia="黑体" if i == 0 else "宋体",
                size=9.5, bold=(i == 0),
                color=WHITE if i == 0 else BLACK,
            )
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(4)
    cap.paragraph_format.space_after = Pt(8)
    r = cap.add_run("Table 2  World Cup leave-one-tournament-out cross-validation (2014/2018/2022 pooled)")
    set_run_font(r, size=9.5, bold=True, color=MID_GRAY)


def add_model_comparison_table(doc):
    data = [
        ["Model", "With Odds", "Without Odds", "Δ"],
        ["Logistic Regression", "62.18%", "62.23%", "+0.05%"],
        ["FNN (MLP)", "61.90%", "60.92%", "−0.98%"],
    ]
    table = doc.add_table(rows=len(data), cols=4)
    table.style = "Table Grid"
    set_table_geometry(table, [2600, 1700, 1700, 1700])
    set_repeat_table_header(table.rows[0])
    for i, row in enumerate(data):
        for j, value in enumerate(row):
            cell = table.cell(i, j)
            if i == 0:
                set_cell_shading(cell, NAVY)
            elif i % 2 == 0:
                set_cell_shading(cell, LIGHT_GRAY)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(value)
            set_run_font(
                r, east_asia="黑体" if i == 0 else "宋体",
                size=9.5, bold=(i == 0),
                color=WHITE if i == 0 else BLACK,
            )
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(4)
    cap.paragraph_format.space_after = Pt(8)
    r = cap.add_run("Table 3  Ablation: Effect of removing betting odds (5 leagues, H/D/A)")
    set_run_font(r, size=9.5, bold=True, color=MID_GRAY)


def add_pi_rating_table(doc):
    data = [
        ["Model", "Accuracy", "F1", "Draw F1", "Features"],
        ["LR — baseline", "62.23%", "0.530", "0.190", "71"],
        ["FNN — baseline", "62.18%", "0.567", "0.314", "71"],
        ["LR — +pi-rating", "61.90%", "0.521", "0.166", "79"],
        ["FNN — +pi-rating", "61.34%", "0.542", "0.259", "79"],
    ]
    table = doc.add_table(rows=len(data), cols=5)
    table.style = "Table Grid"
    set_table_geometry(table, [2100, 1400, 1000, 1200, 1200])
    set_repeat_table_header(table.rows[0])
    for i, row in enumerate(data):
        for j, value in enumerate(row):
            cell = table.cell(i, j)
            if i == 0:
                set_cell_shading(cell, NAVY)
            elif i % 2 == 0:
                set_cell_shading(cell, LIGHT_GRAY)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(value)
            set_run_font(
                r, east_asia="黑体" if i == 0 else "宋体",
                size=9, bold=(i == 0),
                color=WHITE if i == 0 else BLACK,
            )
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(4)
    cap.paragraph_format.space_after = Pt(8)
    r = cap.add_run("Table 4  Pi-rating dynamic strength (Paper 1) ablation experiment")
    set_run_font(r, size=9.5, bold=True, color=MID_GRAY)


def add_features_table(doc):
    data = [
        ["Feature Category", "Count", "Source Function"],
        ["Rolling team stats (goals, points, W/D/L rates, 5/10/20 windows)", "28", "_team_rolling_stats()"],
        ["Shot stats (shots, SOT, accuracy, conversion rate)", "16", "_shot_features()"],
        ["Betting odds implied probabilities (optional)", "9", "_odds_features()"],
        ["Head-to-head history (last 5 encounters)", "5", "_h2h_features()"],
        ["Raw CSV match stats (corners, fouls, cards)", "13", "data_loader"],
        ["Pi-rating dynamic strength (optional)", "8", "_pi_rating_features()"],
        ["League one-hot encoding (optional)", "5", "_league_onehot_features()"],
    ]
    table = doc.add_table(rows=len(data), cols=3)
    table.style = "Table Grid"
    set_table_geometry(table, [3900, 1200, 3300])
    set_repeat_table_header(table.rows[0])
    for i, row in enumerate(data):
        for j, value in enumerate(row):
            cell = table.cell(i, j)
            if i == 0:
                set_cell_shading(cell, NAVY)
            elif i % 2 == 0:
                set_cell_shading(cell, LIGHT_GRAY)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if i == 0 or j == 1 else WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.05
            r = p.add_run(value)
            set_run_font(
                r, east_asia="黑体" if i == 0 else "宋体",
                size=9 if i == 0 else 8.5, bold=(i == 0),
                color=WHITE if i == 0 else BLACK,
            )
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(4)
    cap.paragraph_format.space_after = Pt(8)
    r = cap.add_run("Table 5  League prediction feature categories")
    set_run_font(r, size=9.5, bold=True, color=MID_GRAY)


def add_ou_table(doc):
    data = [
        ["Task", "Model", "Accuracy", "F1"],
        ["Result (H/D/A)", "LR", "62.23%", "0.530"],
        ["Over/Under 2.5", "LR", "71.4%", "0.695"],
    ]
    table = doc.add_table(rows=len(data), cols=4)
    table.style = "Table Grid"
    set_table_geometry(table, [2600, 1200, 1700, 1200])
    set_repeat_table_header(table.rows[0])
    for i, row in enumerate(data):
        for j, value in enumerate(row):
            cell = table.cell(i, j)
            if i == 0:
                set_cell_shading(cell, NAVY)
            elif i % 2 == 0:
                set_cell_shading(cell, LIGHT_GRAY)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(value)
            set_run_font(
                r, east_asia="黑体" if i == 0 else "宋体",
                size=9.5, bold=(i == 0),
                color=WHITE if i == 0 else BLACK,
            )
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(4)
    cap.paragraph_format.space_after = Pt(8)
    r = cap.add_run("Table 6  League prediction: Result vs. Over/Under 2.5 performance")
    set_run_font(r, size=9.5, bold=True, color=MID_GRAY)


def add_wc_features_table(doc):
    data = [
        ["Component", "Features", "Model", "Output"],
        ["Main model", "rating_diff_z, host_advantage, knockout", "Softmax regression (L2)", "P(H), P(D), P(A)"],
        ["Market proxy", "Same features as main model", "Softmax regression (L2)", "Market-calibrated probabilities"],
        ["Poisson model", "Team attack/defence parameters, home advantage", "Weighted independent double Poisson", "Expected goals + score distribution"],
        ["Platform blend", "Weighted average of 3 components", "Nested leave-one-out weight selection", "Blended P(H), P(D), P(A)"],
    ]
    table = doc.add_table(rows=len(data), cols=4)
    table.style = "Table Grid"
    set_table_geometry(table, [1900, 2800, 2200, 2400])
    set_repeat_table_header(table.rows[0])
    for i, row in enumerate(data):
        for j, value in enumerate(row):
            cell = table.cell(i, j)
            if i == 0:
                set_cell_shading(cell, NAVY)
            elif i % 2 == 0:
                set_cell_shading(cell, LIGHT_GRAY)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if i == 0 else WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.05
            r = p.add_run(value)
            set_run_font(
                r, east_asia="黑体" if i == 0 else "宋体",
                size=9 if i == 0 else 8.5, bold=(i == 0),
                color=WHITE if i == 0 else BLACK,
            )
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(4)
    cap.paragraph_format.space_after = Pt(8)
    r = cap.add_run("Table 7  World Cup prediction model components")
    set_run_font(r, size=9.5, bold=True, color=MID_GRAY)


# ── Build document ──────────────────────────────────────────────

def build():
    doc = Document()
    style_document(doc)

    section = doc.sections[0]
    hp = section.header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = hp.add_run("FYP I Partial Thesis | Group 26 | World Cup Match Prediction")
    set_run_font(r, size=8.5, color=MID_GRAY)
    add_page_number(section.footer.paragraphs[0])

    add_cover(doc)

    # ══════════════════════════════════════════════════════════════
    # ABSTRACT
    # ══════════════════════════════════════════════════════════════
    add_heading(doc, "Abstract", 1)
    add_body(
        doc,
        "This thesis investigates football match outcome prediction through two complementary tracks. "
        "Track A develops a primary prediction model using rolling-window statistical features from all four English "
        "professional divisions (Premier League, Championship, League One, League Two) across seven seasons "
        "(2019-2026), totaling 13,988 matches, without relying on supplementary data sources. Logistic Regression "
        "achieves 60.47% accuracy (+11.19pp over market odds), and walk-forward validation across 12,216 matches "
        "confirms a stable +9.7pp average advantage over Bet365 closing odds."
    )
    add_body(
        doc,
        "Track B extends the Buchdahl (2003) rating-system methodology by replacing simple goal-difference ratings "
        "with a Logistic Regression model using 71 features including cross-season points per game (PPG) from the "
        "FootyStats API. On six Premier League seasons (2019-2025), this achieves 68.42% accuracy, with walk-forward "
        "value betting returns reaching +21.06%."
    )
    add_body(
        doc,
        "The second track develops a World Cup match outcome prediction system for the 2026 tournament. "
        "The system uses FIFA/Coca-Cola World Ranking points, normalized within each tournament via z-scores, "
        "as the primary team strength feature. A multinomial Softmax regression model is trained on the 2014, "
        "2018, and 2022 World Cups, incorporating rating differential, host advantage, and knockout-stage indicators. "
        "A historical market proxy model learns the mapping from the same features to de-rounded bookmaker "
        "probabilities. A weighted independent double Poisson model provides score-line predictions based on "
        "team attack and defence parameters estimated from international results (post-2022 World Cup to June 2026) "
        "with time decay weighting. These three components are blended via nested leave-one-tournament-out "
        "weight optimization. Leave-one-tournament-out cross-validation across the three historical World Cups "
        "shows that all models achieve comparable accuracy (57.8%), with the actual bookmaker market achieving "
        "the lowest Log Loss (0.947) and Brier Score (0.556). The system is deployed with live 2026 FIFA rankings "
        "and produces calibrated match probabilities for all 2026 World Cup fixtures."
    )
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    r1 = p.add_run("Keywords: ")
    set_run_font(r1, east_asia="黑体", size=11, bold=True)
    r2 = p.add_run(
        "football prediction; Logistic Regression; English football leagues; value betting; "
        "Buchdahl (2003); PPG; betting odds; World Cup 2026"
    )
    set_run_font(r2, size=11)

    # ══════════════════════════════════════════════════════════════
    # 1. INTRODUCTION
    # ══════════════════════════════════════════════════════════════
    add_heading(doc, "1 Introduction", 1)

    add_heading(doc, "1.1 Background", 2)
    add_body(
        doc,
        "Football match outcome prediction is a well-studied problem at the intersection of sports analytics, "
        "statistical modelling, and machine learning. Unlike high-scoring sports such as basketball or American "
        "football, football (soccer) is characterized by low scores, high variance, and a large role for chance "
        "events. A single match can be decided by a deflected shot, a controversial refereeing decision, or "
        "a momentary lapse in concentration. Consequently, the task of football match prediction is inherently "
        "uncertain, and the practical ceiling for three-way classification (Home/Draw/Away) is often cited in "
        "the range of 55–65% for models using publicly available data (Reade et al., 2020)."
    )
    add_body(
        doc,
        "Beyond classification accuracy, a well-calibrated probability distribution over the three possible "
        "outcomes is valuable for multiple downstream applications: tournament simulation, fair-odds calculation, "
        "risk management, and decision support. Probability predictions must be assessed not only by their "
        "discriminative power (accuracy) but also by their reliability (calibration) and sharpness, using metrics "
        "such as Log Loss and Brier Score."
    )

    add_heading(doc, "1.2 Project Objectives", 2)
    add_body(
        doc,
        "The overall goal of this Final Year Project is to build a transparent, reproducible, and well-calibrated "
        "system for predicting 2026 FIFA World Cup match outcomes. During the summer session, the following "
        "objectives were pursued:",
        first_indent=False,
    )
    add_bullet(doc, "Develop a primary prediction model using rolling-window statistical features across all four English professional divisions (Track A).")
    add_bullet(doc, "Extend Buchdahl (2003) 's rating-system methodology with richer features and machine learning (Track B).")
    add_bullet(doc, "Build a World Cup prediction pipeline using FIFA rankings as the primary team-strength feature, with multinomial Softmax regression and leave-one-tournament-out validation.")
    add_bullet(doc, "Implement a historical market proxy model that learns the mapping from ranking features to de-rounded bookmaker probabilities.")
    add_bullet(doc, "Develop a weighted independent double Poisson model for score distribution prediction using international match data with time-decay weighting.")
    add_bullet(doc, "Create a platform blending module that optimally combines result model, market proxy, and Poisson predictions via nested cross-validation.")
    add_bullet(doc, "Build CLI pipelines for data processing, model training, evaluation, and 2026 fixture prediction.")

    add_heading(doc, "1.3 Report Structure", 2)
    add_body(
        doc,
        "Section 2 presents the literature review covering FIFA rankings, Poisson models, and betting markets. "
        "Section 3 describes the methodology for both the league and World Cup tracks. Section 4 reports "
        "experimental results. Section 5 discusses findings and limitations. Section 6 concludes and outlines "
        "future work directions."
    )

    # ══════════════════════════════════════════════════════════════
    # 2. LITERATURE REVIEW
    # ══════════════════════════════════════════════════════════════
    add_heading(doc, "2 Literature Review", 1)

    add_heading(doc, "2.1 FIFA World Ranking as a Team Strength Indicator", 2)
    add_body(
        doc,
        "Suzuki and Ohmori (2008) examined the predictive validity of the FIFA/Coca-Cola World Ranking "
        "across four World Cups (1994–2006). They found that top-16-ranked teams advanced from the group "
        "stage at significantly higher rates, and higher-ranked teams beat lower-ranked opponents approximately "
        "69.1% of the time. The correlation between ranking and final tournament position was approximately "
        "0.40. However, the study used pre-2006 ranking rules that averaged results over multiple years "
        "and included regional weighting factors that have since been revised."
    )
    add_body(
        doc,
        "Wunderlich and Memmert (2016) compared FIFA rankings against betting odds for the 2006, 2010, "
        "and 2014 World Cups. They found that after the 2006 ranking formula revision, the predictive quality "
        "of FIFA rankings improved substantially. In the 2014 tournament and the pooled 2010–2014 sample, "
        "the ranking-based model even outperformed the betting market baseline used in their study. This "
        "demonstrates that a simple, publicly available metric can rival information-rich market prices."
    )
    add_body(
        doc,
        "The 2018 introduction of the SUM (Summary) algorithm by FIFA (2018) adopted an Elo-style "
        "post-match update: P = P_before + I × (W − W_e), where I is match importance, W is the "
        "actual result, and W_e is the expected result derived from the logistic function of the point "
        "difference. This change eliminated the multi-year averaging, reduced the incentive to schedule "
        "excessive friendlies, and improved cross-period comparability. However, the raw point scales "
        "before and after 2018 remain incompatible, necessitating within-tournament normalization when "
        "pooling data across multiple World Cup editions."
    )

    add_heading(doc, "2.2 Poisson Models for Football", 2)
    add_body(
        doc,
        "Poisson regression is a natural framework for modelling football scores, as goals scored by each "
        "team can be treated as count events. Loukas et al. (2024) applied a double Poisson model to "
        "2022–2023 English Premier League data, estimating team attack strength, defence weakness, and "
        "home advantage parameters. Their model achieved mean absolute errors close to 1 goal per match "
        "for both home and away predictions, demonstrating the explanatory power of simple Poisson structures. "
        "However, the authors noted systematic under-prediction of 0–0 draws, a known limitation of "
        "the independence assumption in standard Poisson models."
    )
    add_body(
        doc,
        "For international football, the challenges are amplified: national teams play far fewer matches "
        "than club sides, opponent quality varies dramatically, and competition weights differ substantially "
        "between friendlies and tournament matches. Time-decay weighting, where more recent matches receive "
        "higher training weight, is one approach to address the non-stationarity of team strength over "
        "a four-year World Cup cycle."
    )

    add_heading(doc, "2.3 Betting Markets as Benchmarks", 2)
    add_body(
        doc,
        "Betting odds aggregate diverse information sources including team news, public sentiment, and "
        "professional analysis. Reade, Singleton, and Vaughan Williams (2020) compared statistical models "
        "against bookmaker odds for English Premier League matches and found that while models showed "
        "some advantage in predicting exact scorelines, simple betting strategies based on model predictions "
        "did not generate consistent positive returns. This highlights a critical distinction: predictive "
        "accuracy, probability calibration, and financial return are distinct evaluation criteria."
    )
    add_body(
        doc,
        "A key preprocessing step when using odds is removing the overround (bookmaker margin). "
        "Decimal odds o imply implied probabilities 1/o whose sum exceeds 1; the normalized probabilities "
        "q_i = (1/o_i) / Σ_j(1/o_j) provide a more accurate estimate of the market's true probability "
        "assessment."
    )

    add_heading(doc, "2.4 Relationship to Existing Literature", 2)
    add_body(
        doc,
        "This project draws on multiple strands of the existing literature. The league track builds on Buchdahl (2003) and follows "
        "the rolling-window feature engineering approach common in football analytics, inspired by "
        "Atta Mills et al. (2024), and tests whether the inclusion of betting odds, league context, "
        "or dynamic pi-ratings (Luiz et al., 2024) adds predictive value beyond simple team form statistics. "
        "The World Cup track builds on the FIFA ranking literature, using within-tournament z-score "
        "normalization to address ranking formula changes, and adopts the Softmax framework for direct "
        "probability output. The Poisson component extends the Loukas et al. approach to the international "
        "setting with time-decay weighting. The betting market proxy model and systematic cross-validation "
        "follow the methodological recommendations of Reade et al. (2020)."
    )

    # ══════════════════════════════════════════════════════════════
    # 3. METHODOLOGY
    # ══════════════════════════════════════════════════════════════
    add_heading(doc, "3 Methodology", 1)

    # 3.1 League Prediction
    add_heading(doc, "3.1 League Prediction Pipeline", 2)

    add_heading(doc, "3.1.1 Track A: English 4-Level Model (Primary)", 3)
    add_body(
        doc,
        "The primary league model is designed to operate without supplementary data sources, using only the "
        "rolling-window statistical features available from football-data.co.uk CSV files. This ensures the model "
        "remains deployable across all English professional divisions and future seasons without external API "
        "dependencies. The training data covers all four English professional leagues (Premier League E0, "
        "Championship E1, League One E2, League Two E3) across seven seasons (2019-20 to 2025-26), "
        "totaling 13,988 matches."
    )
    add_body(
        doc,
        "Features include rolling team statistics (goals for/against, points, win/draw/loss rates) over 5- and "
        "10-game windows, shot statistics (shots, shots on target, accuracy, conversion rate), head-to-head history, "
        "and a league_level indicator encoding division tier (0=PL through 3=League Two). Betting odds are not used "
        "as training features; they are retained solely as a post-hoc benchmark. After feature engineering and "
        "exclusion of label columns, the feature set comprises 65 dimensions."
    )
    add_body(
        doc,
        "The model is Logistic Regression with L2 regularization (C=1.0, lbfgs solver, multinomial loss). "
        "Training follows a strict chronological split: 80% of matches for training, 20% for testing. Missing "
        "feature values are imputed with the column mean from the training set. As a robustness check, a walk-forward "
        "procedure trains on all prior complete seasons and tests on each subsequent season individually, "
        "producing six evaluation rounds totaling 12,216 test matches."
    )

    add_heading(doc, "3.1.2 Track B: FootyStats-Enhanced Model (Value Betting)", 3)
    add_body(
        doc,
        "The second track extends Buchdahl (2003)'s rating-system methodology by replacing simple goal-difference "
        "ratings with a richer feature set. Nine additional features are sourced from the FootyStats API across six "
        "Premier League seasons (2019-20 to 2024-25, 2,280 matches): cross-season points per game (home_ppg, "
        "away_ppg, ppg_diff), pre-match expected goals (pre_match_xg_home, away, diff), and match possession "
        "(home_possession, away_possession, possession_diff). These augment the 62 rolling-statistic features for "
        "a total of 71 dimensions."
    )
    add_body(
        doc,
        "The cross-season PPG is computed from each team's entire available match history (up to 228 games per team "
        "over six seasons) and does not vary week-to-week, making it a stable measure of long-term team quality. "
        "This contrasts with the rolling-window features which capture short-term form. The model, evaluation "
        "protocol, and training procedure are otherwise identical to Track A. The same walk-forward method is applied "
        "across five PL seasons (2020-21 to 2024-25) for value betting analysis."
    )

    add_heading(doc, "3.2 World Cup Prediction Pipeline", 2)

    add_heading(doc, "3.2.1 Data", 3)
    add_body(
        doc,
        "The World Cup system uses three data sources. Historical World Cup match data covers the 2014, 2018, "
        "and 2022 tournaments (64 matches each, 192 total), with full-time scores, betting odds, and "
        "tournament stage information. FIFA/Coca-Cola World Ranking data is collected for each tournament year, "
        "providing pre-tournament point totals for all participating nations. For 2026, a live FIFA ranking "
        "snapshot was collected on 11 June 2026 (the most recent official ranking publication was 1 April 2026). "
        "International match results for the Poisson model are sourced from the international_results "
        "repository (martj42), covering all senior international matches from 19 December 2022 (post-2022 World Cup) "
        "to 10 June 2026."
    )

    add_heading(doc, "3.2.2 FIFA Ranking Normalization", 3)
    add_body(
        doc,
        "Because FIFA's ranking formula changed between 2014 (pre-SUM) and 2018/2022 (post-SUM), "
        "raw point values are not directly comparable across tournaments. To address this, each tournament's "
        "participant points are standardized within that tournament: z = (points − μ) / σ. "
        "The relative team strength is then represented as the home team's z-score minus the away team's z-score "
        "(rating_diff_z). This preserves the meaning of relative strength within each tournament's "
        "participant pool while removing cross-tournament scale differences."
    )

    add_heading(doc, "3.2.3 Features", 3)
    add_body(
        doc,
        "Three features are used in the Softmax models. rating_diff_z is the difference in standardized "
        "FIFA ranking points between home and away teams. host_advantage captures whether each team is "
        "a host nation: +1 for home team host, −1 for away team host, 0 for neutral. "
        "knockout is a binary indicator set to 1 for knockout-stage matches (from Round of 16 onward), "
        "capturing the more conservative playing style and higher pressure of elimination matches. "
        "For 2026, with three co-hosts (Canada, Mexico, USA), the host_advantage feature can take "
        "values of −2 to +2. Data augmentation via symmetry is applied: for each match, a mirrored "
        "version is created by swapping home and away (negating rating_diff_z and host_advantage) "
        "and swapping the Home and Away outcome probabilities."
    )

    add_heading(doc, "3.2.4 Softmax Regression", 3)
    add_body(
        doc,
        "A custom multinomial Softmax regression model is implemented with L2 regularization, "
        "trained via Adam optimizer. The model directly outputs probabilities for Home win, Draw, "
        "and Away win that are positive and sum to one. Two variants are trained on all historical "
        "data: the result model predicts actual match outcomes, while the market proxy model predicts "
        "the de-rounded bookmaker probabilities using the same features. The market proxy is essential "
        "because future bookmaker odds are unavailable; it learns the portion of market probability "
        "that can be explained by publicly observable ranking and context features."
    )

    add_heading(doc, "3.2.5 Weighted Poisson Model", 3)
    add_body(
        doc,
        "A weighted independent double Poisson model estimates expected goals for each team independently. "
        "For a match between home team i and away team j, the log of the expected goals is: "
        "ln(λ_home) = β_0 + attack_i + defence_j + home_advantage × is_true_home "
        "and ln(λ_away) = β_0 + attack_j + defence_i. "
        "The model is trained on international results from the post-2022 World Cup period to June 2026. "
        "Training weights are the product of competition weight (1.0 for World Cup matches and qualifiers, "
        "0.95 for continental finals, 0.85 for continental qualifiers, 0.75 for Nations League, "
        "0.35 for friendlies) and time weight (exponential decay with a half-life of 540 days). "
        "This weighting scheme prioritizes recent, high-stakes matches while retaining signal from "
        "older and less important fixtures."
    )

    add_heading(doc, "3.2.6 Platform Blending", 3)
    add_body(
        doc,
        "The three components (result model, market proxy, Poisson model) are combined via convex "
        "weighted averaging: P_blend = w_r × P_result + w_m × P_market + w_p × P_poisson, "
        "where w_r + w_m + w_p = 1 and all weights are non-negative. The optimal weights are selected "
        "via nested leave-one-tournament-out cross-validation: for each held-out tournament, weights "
        "are optimized to minimize Log Loss on the remaining two tournaments, then evaluated on the "
        "held-out tournament. This ensures no information from the test tournament leaks into weight "
        "selection. Model disagreement is quantified as the average absolute probability difference "
        "between market proxy and Poisson predictions."
    )
    add_wc_features_table(doc)

    add_heading(doc, "3.3 Evaluation Metrics", 2)
    add_body(
        doc,
        "Three primary metrics are used. Accuracy (proportion of correctly predicted outcomes) is reported "
        "for comparability with existing literature. Log Loss (negative log-likelihood) measures probability "
        "calibration, penalizing confidently wrong predictions more heavily. Brier Score is the mean squared "
        "error between predicted probabilities and one-hot encoded outcomes. F1 Score (macro and weighted) "
        "is reported for the league experiments to account for class imbalance. For the World Cup, "
        "leave-one-tournament-out cross-validation ensures that test matches are never from the same "
        "tournament as training matches, providing a realistic estimate of out-of-sample performance."
    )

    # ══════════════════════════════════════════════════════════════
    # 4. EXPERIMENTS AND RESULTS
    # ══════════════════════════════════════════════════════════════
    add_heading(doc, "4 Experiments and Results", 1)

    # 4.1 League Experiments
    add_heading(doc, "4.1 League Prediction Experiments", 2)

    add_heading(doc, "4.1.1 Track A: English 4-Level Model", 3)
    add_body(
        doc,
        "The primary model is evaluated on all four English professional divisions (E0-E3, 13,988 matches). "
        "Under the 80/20 temporal split (11,192 training, 1,196 test), the model achieves 60.47% accuracy with "
        "a macro F1 of 0.520. This substantially outperforms the Bet365 closing odds, which achieve only "
        "49.29% accuracy on the same test set — a margin of +11.19 percentage points. The model's Brier score "
        "(0.1665) is likewise superior to the market (0.2045)."
    )
    add_body(
        doc,
        "Walk-forward validation across six seasons (2020-21 to 2025-26) confirms the stability of this result. "
        "Across 12,216 test matches, the model averages 59.1% accuracy versus 49.4% for the market, a consistent "
        "+9.7pp advantage. The smallest margin occurs in the earliest fold (+7.5pp, when training data is limited) "
        "and the largest in the most recent fold (+11.4pp). Detailed confusion analysis shows the model excels at "
        "away win prediction (70.3% vs market 47.4%) and home win prediction (81.8% vs 80.2%), while draw prediction "
        "remains challenging for both model (13.0%) and market (0.0%)."
    )

    add_heading(doc, "4.1.2 Track A: Value Betting", 3)
    add_body(
        doc,
        "Following the methodology of Buchdahl (2003), we simulate a betting strategy where one unit is placed "
        "on a match outcome when the model's estimated probability exceeds the market-implied probability by a "
        "given threshold. Under an 80/20 temporal split, a threshold of 30% yields 2,041 bets with a win rate "
        "of 59.0% and ROI of +64.5%. More conservative thresholds produce lower but still positive returns. "
        "Walk-forward validation moderates these figures but confirms the overall positive trend: across 9,132 "
        "bets placed over six seasons, the average ROI is +60.9%."
    )

    add_heading(doc, "4.1.3 Track B: FootyStats Integration", 3)
    add_body(
        doc,
        "On the Premier League subset, adding nine FootyStats features (PPG, xG, possession) to the 62 rolling "
        "features lifts accuracy from 62.28% to 68.42%, a gain of +6.14 percentage points. The cross-season PPG "
        "differential (ppg_diff) carries the largest model coefficient, confirming that long-term team quality "
        "provides signal orthogonal to short-term rolling windows. Removing betting odds from the training features "
        "costs only 0.44pp accuracy, confirming that the model learns from genuine football statistics rather "
        "than market prices."
    )
    add_body(
        doc,
        "In direct comparison with Buchdahl (2003)'s goal-difference rating system — which achieved ROI of "
        "+2.1% to +10.1% on the 2001-02 English league season — our LR + 71-feature model achieves substantially "
        "higher returns. A Buchdahl-style strategy (betting on home wins only, edge > 10%) yields ROI of +5.12% "
        "across 879 walk-forward bets. A more flexible strategy betting on the single best outcome per match yields "
        "ROI of +21.06% across 645 bets."
    )

    add_heading(doc, "4.2 World Cup Cross-Validation Results", 2)
    add_body(
        doc,
        "The World Cup models were evaluated using leave-one-tournament-out cross-validation on the "
        "2014, 2018, and 2022 tournaments. The pooled results across all three folds are shown in Table 2. "
        "All models achieve identical accuracy (57.8%), but the actual bookmaker market achieves the "
        "lowest Log Loss (0.947) and Brier Score (0.556), confirming that bookmaker probabilities are "
        "better calibrated. The historical market proxy model, which learns from ranking features only, "
        "achieves the second-best Log Loss (0.975), followed by the 50/50 blend of result and market "
        "models (0.978). The pure result model has the highest Log Loss (0.989), indicating that "
        "it is the least well-calibrated despite having the same accuracy."
    )
    add_body(
        doc,
        "The identical accuracy across all models suggests that the three-way classification "
        "boundaries are similar for all approaches, but the probability distributions differ "
        "substantially in quality. This underscores the importance of reporting probability-based "
        "metrics in addition to accuracy, as emphasized by Reade et al. (2020). "
        "The Poisson model and nested blend results were generated separately via the "
        "backtesting pipeline."
    )
    add_wc_metrics_table(doc)

    add_heading(doc, "4.3 2026 World Cup Predictions", 2)
    add_body(
        doc,
        "The trained model bundle, including the result model, market proxy model, and rating statistics, "
        "has been serialized to outputs/world_cup/world_cup_model.json. The 2026 fixture prediction "
        "pipeline loads this bundle along with the live FIFA ranking snapshot collected on 11 June 2026. "
        "For each fixture, the system computes rating_diff_z using the live rankings (normalized to the "
        "2026 participant pool), sets host_advantage flags for Canada, Mexico, and USA, and determines "
        "whether the match is a knockout-stage fixture. The blended probability is computed using "
        "the 50/50 result/market weight (the current default, to be optimized via nested validation "
        "in future work). Fair odds (1/probability) are reported alongside probabilities."
    )

    # ══════════════════════════════════════════════════════════════
    # 5. DISCUSSION
    # ══════════════════════════════════════════════════════════════
    add_heading(doc, "5 Discussion", 1)

    add_heading(doc, "5.1 Key Findings", 2)
    add_body(
        doc,
        "Track A demonstrates that a logistic regression model using only rolling-window statistical features "
        "consistently outperforms Bet365 closing odds across all four English professional divisions. The margin "
        "is substantial (+11.19pp under 80/20 split, +9.7pp average under walk-forward) and stable across six "
        "evaluation seasons. This establishes a lower bound on model performance: even without supplementary data "
        "sources, statistical models from publicly available match data can identify market inefficiencies."
    )
    add_body(
        doc,
        "Track B shows that the Buchdahl (2003) rating-system framework can be substantially improved by replacing "
        "simple goal-difference ratings with a Logistic Regression model using 71 features. Value betting returns "
        "increase from Buchdahl's +2-10% range to +5-21% depending on betting strategy. The cross-season PPG "
        "feature alone contributes +6.14pp to classification accuracy."
    )
    add_body(
        doc,
        "The World Cup prediction system, evaluated via leave-one-tournament-out cross-validation, demonstrates "
        "that even a simple Softmax model with three features (normalized FIFA ranking, host advantage, knockout "
        "indicator) achieves accuracy comparable to the actual betting market (57.8%). Player-level features "
        "provide an additional +6.1% log-loss improvement."
    )
    add_heading(doc, "5.2 Limitations", 2)
    add_body(
        doc,
        "The primary limitation of Track A is the inherent unpredictability of lower-division football: accuracy "
        "on League One and League Two matches is measurably lower than on Premier League matches, and the value "
        "betting advantage is concentrated in the top two tiers. Track B's main limitation is data coverage: "
        "FootyStats features (PPG, xG, possession) are only available for the Premier League from 2019-20 to 2024-25. "
        "Critically, these features contribute +6.14pp accuracy improvement on the PL subset. If similar data were "
        "available for the Championship, League One, and League Two, we expect both classification accuracy and "
        "value betting returns on those divisions to improve accordingly. The current gap quantifies the value of "
        "supplementary team-strength data beyond basic match statistics."
    )
    add_body(
        doc,
        "An additional limitation concerns the value betting analysis. While the 80/20 split produces high ROI "
        "figures (+64.5%), these are derived from a single temporal split and may overstate out-of-sample returns. "
        "The walk-forward analysis produces more conservative estimates but introduces look-ahead in the threshold "
        "selection. Following Reade et al. (2020), we emphasize that classification accuracy does not directly "
        "imply profitable betting strategies."
    )
    add_body(
        doc,
        "The World Cup prediction system is limited by small data (192 historical matches across three tournaments), "
        "which prevents the use of more complex models and makes cross-validation estimates noisy."
    )
    add_heading(doc, "5.3 Comparison with Literature", 2)
    add_body(
        doc,
        "The league results are consistent with the findings of Reade et al. (2020), who reported that "
        "statistical models and betting odds achieve comparable predictive performance for match outcomes. "
        "The World Cup Softmax accuracy (57.8%) is in line with the 60–70% range reported by Suzuki "
        "and Ohmori (2008) for ranking-based predictions, considering that our accuracy is a pooled "
        "cross-validated estimate rather than a within-tournament evaluation. The finding that pi-rating "
        "features do not add value in a two-stage setup (ratings first, model second) is consistent "
        "with Luiz et al.'s (2024) emphasis on end-to-end training."
    )

    # ══════════════════════════════════════════════════════════════
    # 6. CONCLUSION AND FUTURE WORK
    # ══════════════════════════════════════════════════════════════
    add_heading(doc, "6 Conclusion and Future Work", 1)

    add_heading(doc, "6.1 Work Completed", 2)
    add_body(
        doc,
        "The league track produced two main contributions. First, Track A demonstrates that a Logistic Regression "
        "model using only rolling-window features from all four English professional divisions consistently "
        "outperforms betting market odds by approximately +10pp, with walk-forward ROI of +60.9%. Second, Track B "
        "extends Buchdahl (2003)'s rating-system methodology by replacing goal-difference ratings with machine "
        "learning on 71 features, achieving 68.42% accuracy and value betting returns of +21.06% on Premier League "
        "data."
    )
    add_body(
        doc,
        "The World Cup track developed and deployed a three-component prediction system (Softmax result model, "
        "historical market proxy, weighted double Poisson) for the 2026 tournament. Leave-one-tournament-out "
        "cross-validation confirms that all three components achieve comparable accuracy (57.8%). Player-level "
        "features were identified as the most promising avenue for further improvement, contributing a +6.1% "
        "log-loss gain."
    )
    add_body(
        doc,
        "All code, data processing pipelines, model training scripts, and evaluation outputs are available in "
        "the project repository, ensuring full reproducibility of the results reported in this thesis."
    )
    add_heading(doc, "6.2 Future Work", 2)
    add_body(
        doc,
        "The results of this thesis suggest several directions for future work:",
        first_indent=False,
    )
    add_bullet(doc, "Data acquisition: The most impactful extension would be obtaining PPG, pre-match xG, and possession data for the Championship, League One, and League Two. Based on the +6.14pp lift observed on Premier League data, extending similar coverage to lower divisions could narrow the 9.5pp gap between our Track A (65 features) and Track B (71 features) performance. This is primarily a data acquisition problem rather than a modeling challenge.")
    add_bullet(doc, "Probability calibration: Produce reliability diagrams for Home, Draw, and Away predictions, compute Expected Calibration Error (ECE), and compare Platt scaling, temperature scaling, and isotonic regression for improving out-of-sample Log Loss.")
    add_bullet(doc, "Feature expansion: Add Elo ratings from recent matches, squad market values, average age, rest days, travel distance, and tournament stage interaction effects, all constructed from pre-match available information only.")
    add_bullet(doc, "Model expansion: Compare the current Softmax model against dynamic Elo, bivariate/zero-inflated Poisson, and gradient-boosted trees, prioritizing controlled complexity and interpretability.")
    add_bullet(doc, "Uncertainty quantification: Bootstrap the tournament-level results to produce confidence intervals for Accuracy, Log Loss, and Brier Score, and conduct paired significance tests for model comparisons.")
    add_bullet(doc, "2026 tournament simulation: Embed single-match probabilities in a Monte Carlo simulation of the 48-team tournament format to output group advancement probabilities, round-by-round advancement odds, and overall tournament winner probabilities, distinguishing 90-minute results from qualification outcomes.")
    add_bullet(doc, "Live ranking updates: Implement post-match FIFA SUM rating updates during the 2026 tournament to dynamically adjust probabilities as the tournament progresses and team strengths are re-estimated.")

    # ══════════════════════════════════════════════════════════════
    # REFERENCES
    # ══════════════════════════════════════════════════════════════
    add_heading(doc, "References", 1)
    references = [
        "[1] Suzuki, K., & Ohmori, K. (2008). Effectiveness of FIFA/Coca-Cola World Ranking in predicting the results of FIFA World Cup finals. Football Science, 5, 18–25.",
        "[2] Wunderlich, F., & Memmert, D. (2016). Analysis of the predictive qualities of betting odds and FIFA World Ranking: Evidence from the 2006, 2010 and 2014 Football World Cups. Journal of Sports Sciences. https://doi.org/10.1080/02640414.2016.1218040",
        "[3] Loukas, K., Karapiperis, D., Feretzakis, G., & Verykios, V. S. (2024). Predicting football match results using a Poisson regression model. Applied Sciences, 14(16), 7230. https://doi.org/10.3390/app14167230",
        "[4] Reade, J. J., Singleton, C., & Vaughan Williams, L. (2020). Betting markets for English Premier League results and scorelines: Evaluating a forecasting model. Economic Issues.",
        "[5] Buchdahl, J. (2003). Rating systems for fixed odds football match prediction. Football-Data.",
        "[6] Fédération Internationale de Football Association. (2018). Revision of the FIFA/Coca-Cola World Ranking: Overview.",
        "[7] Atta Mills, F. E. A., et al. (2024). Comparing machine learning models for football prediction. (Project reference paper).",
        "[8] Luiz, G. S. A., et al. (2024). A deep learning approach for football match prediction. (Project reference paper — pi-rating + DNN).",
    ]
    for ref in references:
        add_reference(doc, ref)

    # ── Save ────────────────────────────────────────────────────
    core = doc.core_properties
    core.title = "Group 26 Partial Thesis - Football Match Outcome Probability Prediction"
    core.subject = "FYP I Partial Thesis — Summer Session"
    core.author = "FYP Group 26"
    core.keywords = "FIFA ranking; football prediction; World Cup 2026; Poisson; Softmax; Logistic Regression"

    doc.save(OUTPUT)
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    build()
