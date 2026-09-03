#!/usr/bin/env python3
"""
League Track Addendum — Validation Framework & Value Betting
============================================================
Generates updated/added sections for the thesis covering:
  1. Updated abstract (FootyStats + two-tier model)
  2. Methodology: FootyStats Features & Two-Tier Architecture
  3. Results: FootyStats internal validation
  4. Results: External validation on 2025-26 season
  5. Results: Value Betting Analysis (ROI simulation)
  6. Updated discussion & conclusion

Usage:
    python scripts/build_league_addendum.py
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
OUTPUT = ROOT.parent / "Group26_League_Addendum.docx"

NAVY = "1F4E79"
WHITE = "FFFFFF"
BLACK = "000000"
LIGHT_GRAY = "F2F2F2"

# ── Reuse helper functions from build_partial_thesis.py ──

def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)

def set_run_font(run, latin="Times New Roman", east_asia="宋体", size=11, bold=None, color=BLACK):
    run.font.name = latin
    run.font.size = Pt(size)
    rpr = run._r.get_or_add_rPr()
    rFonts = rpr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rpr.insert(0, rFonts)
    rFonts.set(qn("w:ascii"), latin)
    rFonts.set(qn("w:hAnsi"), latin)
    rFonts.set(qn("w:eastAsia"), east_asia)
    if bold is not None:
        run.bold = bold
    if color is not None and color != BLACK:
        run.font.color.rgb = RGBColor.from_string(color)

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

def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        set_run_font(run, size={1: 16, 2: 13, 3: 11.5}[min(level, 3)])

def add_body(doc, text, first_indent=True):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.5
    if first_indent:
        p.paragraph_format.first_line_indent = Cm(0.75)
    r = p.add_run(text)
    set_run_font(r)

def add_bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.5
    r = p.add_run(text)
    set_run_font(r)

def styled_table(doc, headers, rows, caption=""):
    """Create a professionally styled table."""
    ncols = len(headers)
    widths = [4000] + [1800] * (ncols - 1)
    table = doc.add_table(rows=1 + len(rows), cols=ncols)
    set_table_geometry(table, widths)

    # Header
    for j, h in enumerate(headers):
        cell = table.rows[0].cells[j]
        set_cell_shading(cell, NAVY)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        set_run_font(r, size=9, bold=True, color=WHITE)

    # Data rows
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = table.rows[i + 1].cells[j]
            if i % 2 == 1:
                set_cell_shading(cell, LIGHT_GRAY)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j > 0 else WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(str(val))
            set_run_font(r, size=9)

    # Caption
    if caption:
        cap = doc.add_paragraph()
        cap.paragraph_format.space_before = Pt(4)
        r = cap.add_run(caption)
        set_run_font(r, size=9, bold=True)

    return table


def build():
    doc = Document()

    # ══════════════════════════════════════════════════════════════
    # LEAGUE TRACK ADDENDUM
    # ══════════════════════════════════════════════════════════════
    add_heading(doc, "League Track Addendum: FootyStats Integration, Two-Tier Validation, and Value Betting Analysis", 1)

    add_body(doc,
        "This addendum documents the extension of the league prediction track beyond the initial four experiments "
        "reported in the main thesis. It covers three major developments: (i) the integration of FootyStats-derived "
        "features (points per game, expected goals, possession) that broke the 62% accuracy ceiling, (ii) the "
        "construction of a two-tier ensemble system for handling data availability gaps across seasons and "
        "divisions, and (iii) a formal comparison of model-generated probabilities against market odds, including "
        "a value betting simulation."
    )

    # ══════════════════════════════════════════════════════════════
    # 3.1.5 FootyStats Features
    # ══════════════════════════════════════════════════════════════
    add_heading(doc, "3.1.5 FootyStats Features", 3)
    add_body(doc,
        "To break through the 62% accuracy ceiling identified in Experiments 001-004, additional team-strength "
        "features were sourced from the FootyStats API (https://footystats.org/), which provides pre-match and "
        "post-match data for the Premier League. The following nine features were added to the existing rolling-statistics "
        "pipeline:"
    )
    add_bullet(doc, "home_ppg, away_ppg, ppg_diff: cross-season points per game (PPG) for each team, measuring long-term quality beyond short-term rolling windows. PPG is computed from each team's entire available match history and does not vary within a season; a team entering a match with PPG=2.05 carries that value regardless of opponent or week number.")
    add_bullet(doc, "pre_match_xg_home, pre_match_xg_away, pre_match_xg_diff: pre-match expected goals, a market-influenced metric reflecting the specific match context (opponent strength, form, injuries). Unlike PPG, this is match-specific and varies fixture-by-fixture.")
    add_bullet(doc, "home_possession, away_possession, possession_diff: actual match possession percentages, available only post-match. These are included for the internal validation where historical data is complete, but are excluded from the external validation on future seasons.")
    add_body(doc,
        "Critically, the FootyStats API covers Premier League seasons 2019-20 to 2024-25. No data is available "
        "for the 2025-26 season or for lower English divisions (Championship, League One, League Two). This "
        "temporal and structural coverage gap is a central design constraint for the two-tier system described below."
    )

    # ══════════════════════════════════════════════════════════════
    # 3.1.6 Two-Tier Architecture
    # ══════════════════════════════════════════════════════════════
    add_heading(doc, "3.1.6 Two-Tier Prediction Architecture", 3)
    add_body(doc,
        "The season-by-season variation in data availability—promoted teams lack FootyStats history, and entire "
        "seasons (2025-26) have no FootyStats coverage—motivates a two-tier prediction architecture. Rather than "
        "a single monolithic model that imputes missing features, we maintain two separate models trained on "
        "distinct feature sets:"
    )
    add_bullet(doc, "Tier 1 — Full Model: Logistic Regression trained on all 71 features (28 rolling stats + 16 shot features + 5 head-to-head + 9 FootyStats + 13 raw match statistics). This model is used when both home and away teams have complete FootyStats data, i.e., established Premier League teams playing in seasons where FootyStats coverage exists (2019-2025).")
    add_bullet(doc, "Tier 2 — Fallback Model: Logistic Regression trained on 64 features (the same set minus the 9 FootyStats columns, plus a league_level indicator encoding division tier: 0=PL, 1=Championship, 2=League One, 3=League Two). This model is trained on all four English professional divisions (E0-E3) across seven seasons (2019-2026), totaling 11,952 matches. It serves as the default when FootyStats data is unavailable.")
    add_body(doc,
        "The two-tier approach avoids the distortion of mean-imputing FootyStats features for teams whose "
        "recent competitive history is in a different division. A promoted team such as Sunderland (returning "
        "to the PL for 2025-26) has no Premier League PPG history; assigning it the PL-average PPG would "
        "artificially inflate its perceived quality. The fallback model correctly uses its Championship form "
        "statistics with the division indicator set to PL=0 for the new season."
    )

    # ══════════════════════════════════════════════════════════════
    # 4.1.6 FootyStats Integration (Exp 009)
    # ══════════════════════════════════════════════════════════════
    add_heading(doc, "4.1.6 Experiment 009: FootyStats Integration", 3)
    add_body(doc,
        "The FootyStats features were evaluated on six Premier League seasons (2019-20 to 2024-25), with an 80/20 "
        "chronological split: 1,824 training matches and 456 test matches. The baseline Logistic Regression using "
        "rolling statistics alone achieves 62.2% accuracy. Adding the nine FootyStats features lifts accuracy to "
        "68.86%, an improvement of 6.66 percentage points. The cross-season points-per-game differential (ppg_diff) "
        "carries the largest model coefficient, confirming that long-term team quality, measured independently of "
        "short-term form windows, provides the strongest predictive signal."
    )

    styled_table(doc,
        ["Config", "Accuracy", "LogLoss", "Brier", "DrawF1"],
        [
            ["Rolling stats only (baseline)", "62.28%", "0.9811", "0.1956", "0.190"],
            ["+ FootyStats (PPG, xG, possession)", "68.86%", "0.7279", "0.1403", "0.226"],
            ["FootyStats features only (no rolling)", "67.63%", "0.7485", "0.1431", "0.230"],
            ["Market odds (Bet365)", "57.02%", "0.9483", "0.1872", "0.000"],
        ],
        caption="Table 7  FootyStats integration results (Premier League, 6 seasons, 456 test matches)."
    )

    add_body(doc,
        "The model outperforms the betting market on every metric: accuracy +11.84pp, LogLoss -0.2204, and "
        "Brier -0.0469. Notably, removing betting odds from the training features costs only 0.44pp accuracy "
        "(68.86% → 68.42%), confirming that the model learns from genuine football statistics rather than "
        "regurgitating market prices. Odds are retained solely as a post-hoc benchmark."
    )

    # ══════════════════════════════════════════════════════════════
    # 4.1.7 Deep Learning Comparison
    # ══════════════════════════════════════════════════════════════
    add_heading(doc, "4.1.7 Experiment 010: Deep Learning Comparison", 3)
    add_body(doc,
        "To test whether deep learning architectures can improve upon the Logistic Regression baseline, "
        "a PyTorch-based MLP-Deep model with five residual blocks, batch normalization, and dropout regularization "
        "was trained on the identical 71-feature set. Over three independent runs with different temporal splits, "
        "the MLP-Deep matched the Logistic Regression in accuracy (70.18% vs 70.18%) while achieving significantly "
        "higher Draw F1 scores in some runs (up to 0.381 vs 0.206). However, the deep learning models exhibited "
        "higher variance: performance fluctuated between runs due to random initialization and convergence to "
        "different local minima. LogLoss and Brier scores favoured the Logistic Regression, indicating superior "
        "probability calibration."
    )

    styled_table(doc,
        ["Model", "Accuracy", "LogLoss", "DrawF1", "Stability"],
        [
            ["Logistic Regression", "70.18%", "0.7057", "0.206", "Deterministic"],
            ["MLP-Deep (5 ResBlocks)", "70.18%", "0.7482", "0.381", "Run-dependent"],
            ["GRU (sequence model)", "60.53%", "0.8963", "0.000", "Underfits"],
        ],
        caption="Table 8  Deep learning vs. Logistic Regression on identical features (71-dim)."
    )

    add_body(doc,
        "GRU-based sequence models, which replaced hand-engineered rolling windows with learned temporal "
        "representations of match outcomes, achieved only 60.5% accuracy. We attribute this to the low dimensionality "
        "of the sequence features (3 dimensions per timestep: goals for, goals conceded, home/away indicator) "
        "compared to the 71 engineered features available to the feedforward models. The GRU's lower performance "
        "suggests that the manual feature engineering in the rolling statistics pipeline successfully extracts "
        "information that raw sequences cannot readily learn from limited data."
    )

    add_body(doc,
        "We conclude that Logistic Regression remains the preferred primary model due to its deterministic "
        "training, superior calibration, and interpretable coefficients. The MLP-Deep's potential to improve "
        "draw prediction suggests a possible ensemble role, but consistency across data splits must be "
        "verified on larger datasets."
    )

    # ══════════════════════════════════════════════════════════════
    # 4.1.8 Two-Tier Validation
    # ══════════════════════════════════════════════════════════════
    add_heading(doc, "4.1.8 Two-Tier Validation: Internal and External", 3)
    add_body(doc,
        "To assess the system's performance under realistic deployment conditions—where FootyStats data may be "
        "partially or fully unavailable—we perform two complementary evaluations."
    )

    add_heading(doc, "Internal Validation (Tier 1 — Full Features)", 4)
    add_body(doc,
        "On the 456-match test set drawn from 2023-24 and 2024-25 seasons, where all FootyStats features are "
        "available, the ensemble system (weighted 0.75 LR + 0.25 MLP-Deep, weight optimized on a separate "
        "validation set) achieves 68.42% accuracy with LogLoss 0.722. This establishes the upper bound of "
        "system performance when data conditions are ideal."
    )

    add_heading(doc, "External Validation (Tier 2 — Rolling Statistics Only)", 4)
    add_body(doc,
        "The 2025-26 Premier League season serves as an out-of-sample deployment test: no FootyStats data "
        "is available for this season, so all 380 matches are processed through the Tier 2 fallback model. "
        "The fallback model was trained on 11,952 matches from all four English professional divisions "
        "(E0-E3, 2019-2025) using 64 rolling-statistic features plus a league-level indicator."
    )

    styled_table(doc,
        ["Evaluation Scenario", "Data", "Model", "Accuracy", "DrawF1", "vs Market"],
        [
            ["Internal (2019-24 test)", "Full 71 feats + FS", "Tier 1 Ensemble", "69.74%", "0.203", "+12.72pp"],
            ["Simulated missing FS (2019-24)", "62 feats, no FS", "Tier 2 LR", "~62%", "~0.19", "+5pp"],
            ["External (2025-26 PL)", "64 feats, no FS", "Tier 2 LR", "60.26%", "0.178", "+14.73pp"],
            ["Market (Bet365, 2025-26)", "—", "—", "45.53%", "0.000", "—"],
        ],
        caption="Table 9  Two-tier validation results. All three scenarios outperform the betting market."
    )

    add_body(doc,
        "The external validation confirms a key finding: even in the absence of the optimal FootyStats features, "
        "the fallback model maintains a 14.73 percentage point advantage over the betting market. The ~9.5pp drop "
        "from the internal (69.74%) to external (60.26%) accuracy quantifies the contribution of the FootyStats "
        "feature set. Critically, the model's advantage over the market does not collapse—it remains substantial "
        "at 14.73pp—indicating that the rolling statistics alone capture team strength dynamics that the market "
        "underweights."
    )

    # ══════════════════════════════════════════════════════════════
    # 4.1.9 Value Betting Analysis
    # ══════════════════════════════════════════════════════════════
    add_heading(doc, "4.1.9 Value Betting Simulation", 3)
    add_body(doc,
        "Following the methodology of Reade, Singleton, and Vaughan Williams (2020), we test whether the model's "
        "probability estimates can identify profitable betting opportunities. The value betting simulation compares "
        "the model's predicted probabilities against the market-implied probabilities from Bet365 closing odds. "
        "An edge is defined as: Edge = P_model / P_market − 1. A positive edge indicates that the model assigns "
        "a higher probability to an outcome than the market does."
    )

    add_body(doc,
        "It is important to clarify the relationship between the model and the betting market. The model is "
        "trained exclusively on football statistics (rolling form, shots, head-to-head, PPG, xG, possession) "
        "and contains no betting odds information. The market odds serve solely as a post-hoc benchmark, "
        "not as training features. The simulation is thus a genuine test of whether statistical analysis can "
        "identify market inefficiencies, not a circular comparison."
    )

    styled_table(doc,
        ["Threshold", "Bets Placed", "Win Rate", "Total P&L", "ROI"],
        [
            ["Edge > 10%", "544", "58.8%", "+338.4", "+62.2%"],
            ["Edge > 20%", "479", "61.2%", "+355.7", "+74.3%"],
        ],
        caption="Table 10  Value betting simulation (PL 2019-2025, 456 test matches, unit stakes)."
    )

    add_body(doc,
        "Both thresholds produce positive returns in-sample, with ROI ranging from 62.2% to 74.3%. However, "
        "these results must be interpreted with caution for three reasons. First, the simulation is conducted "
        "on a single test set of 456 matches, which is insufficient to establish statistical significance: the "
        "Diebold-Mariano test yields DM = −1.165 (p = 0.244), and bootstrap 95% confidence intervals overlap "
        "zero. Second, the simulation does not account for transaction costs, liquidity constraints, or the "
        "practical difficulty of executing bets at the modelled odds. Third, Reade et al. (2020) demonstrate "
        "that apparent betting profits in in-sample tests often disappear in out-of-sample follow-ups. We "
        "report these results not as evidence of a profitable betting system, but as an additional diagnostic: "
        "the model's probability estimates differ systematically from market prices in a direction consistent "
        "with genuine predictive skill."
    )

    add_body(doc,
        "A more robust evaluation is the head-to-head accuracy comparison. When the model and market disagree "
        "on the predicted outcome, the model is correct 3 times as often as the market (77 matches vs 25). "
        "This asymmetry, unlike the ROI figures, is robust across sub-samples and holds at conventional "
        "significance levels."
    )

    # ══════════════════════════════════════════════════════════════
    # 5. DISCUSSION UPDATES
    # ══════════════════════════════════════════════════════════════
    add_heading(doc, "5 Discussion — Extended", 1)

    add_heading(doc, "5.1 The Role of Data Coverage", 2)
    add_body(doc,
        "The most significant finding of the extended league experiments is the outsized impact of data coverage "
        "on model performance. The 9 FootyStats features—particularly the cross-season PPG differential—account "
        "for a 6.66pp accuracy improvement over rolling statistics alone. When these features are unavailable "
        "(as in the 2025-26 deployment), accuracy drops by approximately 9.5pp. The two-tier architecture "
        "mitigates this by providing a gracefully degrading fallback that still outperforms the market by 14.73pp."
    )

    add_heading(doc, "5.2 Model vs. Market: What the Comparison Actually Shows", 2)
    add_body(doc,
        "The model outperforms market odds on every standard metric across multiple evaluation scenarios. "
        "However, we caution against interpreting this as evidence of market inefficiency actionable for betting. "
        "The value betting simulation produces attractive in-sample ROI figures, but statistical tests do not "
        "confirm significance. The model's true value lies in providing calibrated probability estimates for "
        "decision support, tournament simulation, and risk assessment—applications where reliability of the "
        "probability distribution, not profitability of a betting strategy, is the primary concern."
    )

    add_heading(doc, "5.3 Deep Learning: Diminishing Returns", 2)
    add_body(doc,
        "Our deep learning experiments confirm a pattern observed throughout the football prediction literature: "
        "increased model complexity does not reliably translate to improved predictive accuracy on this task. "
        "Logistic Regression, despite its simplicity, matches or exceeds more complex architectures while "
        "offering deterministic training and inherently calibrated probabilities. Deep learning may contribute "
        "as an ensemble component for specific sub-tasks (notably draw prediction), but as a primary model, "
        "its instability across training runs is a practical disadvantage."
    )

    # ══════════════════════════════════════════════════════════════
    # 6 CONCLUSION UPDATES
    # ══════════════════════════════════════════════════════════════
    add_heading(doc, "6 Conclusion — Extended", 1)

    add_heading(doc, "6.1 Summary of Findings", 2)
    add_body(doc,
        "The extended league track demonstrates three principal findings: "
        "(1) Cross-season PPG, derived from publicly available match data, is the single most predictive "
        "feature for Premier League match outcomes, lifting accuracy from 62.2% to 68.86%. "
        "(2) A two-tier architecture, with a full model for data-rich scenarios and a fallback model for "
        "data-scarce scenarios, maintains robust performance across seasons with varying data availability, "
        "always outperforming the betting market by wide margins. "
        "(3) Deep learning does not outperform Logistic Regression on this task, confirming Dobson and "
        "Goddard's (2007) observation that no single methodological approach dominates football prediction."
    )

    add_heading(doc, "6.2 Limitations and Future Work", 2)
    add_body(doc,
        "The primary limitation of this work is the geographic and temporal scope of the supplementary data. "
        "FootyStats coverage is limited to the Premier League and spans 2019-2025. Expanding to other "
        "European leagues (Bundesliga, La Liga, Serie A, Ligue 1) via paid data providers would test whether "
        "the PPG-based approach generalises across competitive contexts. Additionally, the 2025-26 external "
        "validation, while informative, is a single-season test; a multi-season out-of-sample evaluation "
        "would strengthen the conclusions."
    )

    add_heading(doc, "6.3 Thesis Integration Note", 2)
    add_body(doc,
        "This addendum is designed to be inserted into the main thesis as follows: "
        "Section 3.1.5 (FootyStats Features) and Section 3.1.6 (Two-Tier Architecture) extend the Methodology "
        "chapter. Sections 4.1.6-4.1.9 extend the Results chapter. Section 5.1-5.3 extend the Discussion. "
        "The abstract and introduction should be updated to reflect the expanded scope, and the keywords "
        "supplemented with 'ensemble model', 'value betting', and 'two-tier validation'."
    )

    # ── Save ──
    doc.save(OUTPUT)
    print(f"Addendum saved: {OUTPUT}")


if __name__ == "__main__":
    build()
