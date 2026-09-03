#!/usr/bin/env python3
"""Generate detailed project overview docx EN."""
from pathlib import Path
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor

PROJ = Path(__file__).resolve().parent.parent
OUT = PROJ / "Project_Overview.docx"
doc = Document()

NAVY = RGBColor.from_string("1F4E79")
BLACK = RGBColor(0, 0, 0)

def add_heading(text, level=1):
    h = doc.add_heading(text, level=level)
    for r in h.runs:
        r.font.color.rgb = NAVY

def add_body(text):
    p = doc.add_paragraph(text)
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_after = Pt(6)

def add_bullet(text):
    p = doc.add_paragraph(text, style="List Bullet")
    p.paragraph_format.space_after = Pt(2)

# ═════════════════════════════════════════════════════════
add_heading("Project Overview: Football Match Prediction", 1)
add_body("Final Year Project — Group 26")
add_body("Logistic Regression-based system for predicting English professional football outcomes using rolling-window statistics from publicly available match data.")

# ═════════════════════════════════════════════════════════
add_heading("1. Two-Track Structure", 1)

add_heading("Track A: English 4-Level Model (Primary)", 2)
add_bullet("Question: Can a statistical model using only free match data consistently outperform betting market odds?")
add_bullet("Data: 11,952 matches, E0-E3 (all English divisions), 6 seasons (2019-2025)")
add_bullet("Features: 62 dimensions (rolling stats, shots, h2h, league level). NO betting odds in training.")
add_bullet("Model: Logistic Regression (L2, lbfgs, C=1.0)")
add_bullet("Core: Walk-forward 57.25% vs market 49.40% (+7.85pp)")
add_bullet("Reference: Feature engineering from Atta Mills et al. (2024); LR selection based on our experiments")

add_heading("Track B: FootyStats-Enhanced Model", 2)
add_bullet("Question: Can cross-season PPG improve Buchdahl (2003)'s rating framework?")
add_bullet("Data: 2,280 PL matches, 6 seasons, 71 features (62 + 9 FootyStats)")
add_bullet("Core: 68.42% (+6.14pp over rolling baseline). Buchdahl: +2-10% vs ours +5-22%.")
add_bullet("Limitation: PL only, 2019-2025. ~9.5pp gap if lower divisions had similar data.")

# ═════════════════════════════════════════════════════════
add_heading("2. Value Betting Methodology", 1)

add_heading("2.1 Edge", 2)
add_body("Edge = P_model / P_market - 1")
add_body("P_model from logistic regression; P_market from de-ordered Bet365 closing odds.")
add_body("Example — Liverpool vs Crystal Palace (2023-24):")
add_bullet("Model predicted away win: P=0.281; Market: P=0.086 (odds 11.0); Edge = 227%")
add_bullet("Actual: Crystal Palace won. 1-unit bet returns 10 units profit.")

add_heading("2.2 Threshold", 2)
add_body("Minimum edge required to place a bet. Higher threshold = fewer bets, higher quality per bet.")
add_body("Walk-forward full metrics (10,180 test matches):")
add_bullet("0%: 10,180 bets, ROI +43.1% | 30%: 7,782 bets, ROI +55.0%")
add_bullet("50%: 5,658 bets, ROI +66.0% | 100%: 2,441 bets, ROI +99.3%")
add_body("The 30% threshold is recommended, balancing sample size and ROI.")

add_heading("2.3 Betting Strategy", 2)
add_body("Best per Match: for each match, select the outcome (H/D/A) with the highest edge. Max 1 bet per match.")

add_heading("2.4 Market Odds Processing", 2)
add_body("Raw Bet365 closing odds -> 1/odds -> de-order (normalise to sum=1) -> market probability.")
add_body("Example: Liverpool vs Crystal Palace odds 1.2/7.5/11.0 -> de-ordered: 78.8%/12.6%/8.6%.")

add_heading("2.5 Comparison with Buchdahl (2003)", 2)
add_body("Original: goal-difference rating -> historical probability table -> fair odds -> compare with bookmaker -> bet home only.")
add_body("Ours: LR outputs probabilities -> Edge = P_model/P_market - 1 -> compare with threshold -> bet any outcome.")

# ═════════════════════════════════════════════════════════
add_heading("3. Validation Method: Walk-forward", 1)
add_body("Walk-forward validation: train on all completed seasons, test on the next full season. Expand training set, repeat. 5 folds, 10,180 test matches total. This simulates real deployment and tests cross-season robustness.")
add_body("Each match's features use only data from before that match (shift(1) in rolling statistics), ensuring no future information leakage.")

add_heading("3.1 Track A Results", 2)
add_body("Walk-forward (5 folds, 10,180 test matches):")
add_bullet("Model: 57.25%, Market: 49.40%, Margin: +7.85pp")
add_bullet("No single season underperforms market by more than 2pp")
add_bullet("Value betting (30% threshold): 7,782 bets, ROI +55.0%")
add_bullet("Per-class: Away 70.3% (market 47.4%), Draw 13.0% (market 0.0%), Home 81.8% (market 80.2%)")

add_heading("3.2 Track B Results", 2)
add_bullet("Rolling baseline: 62.28% | +FootyStats: 68.42% (+6.14pp) | Market: 57.02%")

add_heading("3.3 Model vs Market Disagreement", 2)
add_body("On 1,900 PL walk-forward matches, model and market disagree on 38.6% of matches. When they disagree, model is correct 45.8% vs market 25.6% (ratio 1.8x).")
add_body("Example 1 — Liverpool vs Crystal Palace (2023-24): market priced home win at 78.8%; model gave away win 28.1% (Edge 227%). Actual: Crystal Palace 1-0.")
add_body("Example 2 — Bournemouth vs Arsenal (2024-25): market favoured home (33%); model predicted away win at 98% (Edge 200%). Actual: Arsenal away win.")

# ═════════════════════════════════════════════════════════
add_heading("4. Limitations & Future Work", 1)
add_bullet("FootyStats covers PL only (2019-2025). Expanding to lower divisions is the clearest improvement path.")
add_bullet("Lower division accuracy is lower due to increased competitive randomness.")
add_bullet("Draw prediction is limited: model only 13.0%, market 0.0%. Draw is the most difficult outcome.")
add_bullet("Value betting ROI is threshold-sensitive (+43% to +99%). Classification advantage is far more stable.")
add_bullet("Future work: PPG/xG for lower divisions, player-level features (value, injuries, squad depth).")

# ═════════════════════════════════════════════════════════
add_heading("5. Key Papers Referenced", 1)
add_bullet("Buchdahl, J. (2003). Rating Systems for Fixed Odds Football Match Prediction.")
add_bullet("Reade, J., Singleton, C. & Vaughan Williams, L. (2020). Betting markets and statistical models.")
add_bullet("Luiz et al. (2024). A deep learning approach for football match prediction.")
add_bullet("Atta Mills et al. (2024). Comparing machine learning models for football prediction.")

doc.save(OUT)
print(f"Saved: {OUT}")
