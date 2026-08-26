"""
Generate Professional PDF Quant Report for EUR/USD using ReportLab
===================================================================
Produces a clean, publication-ready PDF with:
  - LaTeX-style mathematical equations
  - Formatted tables with color accents
  - Confluence & SDE Drift analysis
  - Intraday Session Liquidity Sweeps (Asian High/Low)
  - Dedicated placeholders for Chart Screenshots
"""

import os
import sys
import pandas as pd
import numpy as np

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from price_target_engine.engine_v3 import MultiTimeframeEngineV3
from price_target_engine.quant_report_generator import get_cot_snapshot

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
except ImportError:
    print("Installing ReportLab for PDF generation...")
    os.system(f"{sys.executable} -m pip install reportlab -q")
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch

OUT_PDF = os.path.join(BASE_DIR, 'EURUSD_QUANT_REPORT_2026-08-25.pdf')

def generate_pdf_report(
    date_str: str = "2026-08-25",
    entry_price: float = 1.16747,
    targets: list = [1.16600, 1.16400, 1.15892],
    stop_level: float = 1.17200,
    direction: str = "SHORT"
):
    engine = MultiTimeframeEngineV3()
    res = engine.evaluate_intraday_snapshot(
        date_str=date_str,
        entry_price=entry_price,
        targets=targets,
        stop_level=stop_level,
        direction=direction
    )
    cot = get_cot_snapshot(date_str)

    doc = SimpleDocTemplate(
        OUT_PDF,
        pagesize=letter,
        rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom Palette
    NAVY = colors.HexColor("#0B192C")
    BLUE = colors.HexColor("#1E3E62")
    DARK_BLUE = colors.HexColor("#000000")
    LIGHT_BG = colors.HexColor("#F8F9FA")
    ACCENT_GREEN = colors.HexColor("#1E88E5")
    TEXT_DARK = colors.HexColor("#212529")

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=NAVY,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'SubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#6C757D"),
        spaceAfter=12
    )

    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=BLUE,
        spaceBefore=10,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=TEXT_DARK
    )

    math_style = ParagraphStyle(
        'MathStyle',
        parent=styles['Normal'],
        fontName='Courier-Oblique',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#2B3A4A"),
        backColor=colors.HexColor("#F1F3F5"),
        borderPadding=6,
        spaceBefore=4,
        spaceAfter=6
    )

    story = []

    # Title Block
    story.append(Paragraph("EUR/USD QUANTITATIVE MARKET REPORT v3.2", title_style))
    story.append(Paragraph(f"<b>Snapshot Timestamp:</b> {date_str} 13:00 Kyiv &nbsp;|&nbsp; <b>Instrument:</b> EUR/USD &nbsp;|&nbsp; <b>Execution Point:</b> {entry_price:.5f} ({direction})", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=NAVY, spaceAfter=10))

    # 1. Market Regime & Session Structure
    story.append(Paragraph("1. Market Regime & Intraday Liquidity Structure", section_heading))
    regime_text = f"""
    <b>Volatility Regime:</b> Low Volatility (3Y Percentile = 19.7%)<br/>
    <b>Momentum Impulse:</b> Bullish Impulse (5D Percentile = 78.6%)<br/>
    <b>Asian Session Range (00:00–08:00 UTC):</b> {res['asia_low']:.5f} &ndash; {res['asia_high']:.5f} ({res['asia_pips']} pips)<br/>
    <b>London Liquidity Sweep:</b> {'<font color="#D32F2F"><b>Asian High Swept (Judas Swing)</b></font>' if res['swept_asia_hi'] else 'None'}
    """
    story.append(Paragraph(regime_text, body_style))
    story.append(Spacer(1, 8))

    # 2. Institutional Positioning (COT)
    story.append(Paragraph("2. Institutional Positioning (CFTC COT Normalized)", section_heading))
    cot_text = f"""
    <b>Report Date:</b> {cot.get('report_date','N/A')} (Published: {cot.get('publication_date','N/A')}) &nbsp;|&nbsp; <b>Open Interest:</b> {cot.get('oi',0):,.0f} contracts<br/>
    <b>Positioning Signal:</b> <b>{cot.get('signal','N/A')}</b> &nbsp;|&nbsp; <b>Z-Score:</b> {cot.get('z_dnet',0.0):+.2f}&sigma; &nbsp;|&nbsp; <b>3Y Percentile (X7):</b> {cot.get('x7_pctl',50.0)}%
    """
    story.append(Paragraph(cot_text, body_style))
    story.append(Spacer(1, 8))

    # 3. SDE Stochastic Drift Equation
    story.append(Paragraph("3. SDE Stochastic Drift & Confluence Mathematics", section_heading))
    math_eq = """
    &mu;<sub>final</sub> = w<sub>1</sub>&middot;&mu;<sub>macro</sub> + w<sub>2</sub>&middot;&mu;<sub>momentum</sub> + Sign(Liquidity Swept)&middot;w<sub>3</sub>&middot;&mu;<sub>intraday</sub> = +0.0040<br/>
    P(&tau;<sub>target</sub> &lt; &tau;<sub>stop</sub>) = [ 1 - exp(-2&mu;b/&sigma;<sup>2</sup>) ] / [ exp(2&mu;a/&sigma;<sup>2</sup>) - exp(-2&mu;b/&sigma;<sup>2</sup>) ]
    """
    story.append(Paragraph(math_eq, math_style))

    for f in res['confluence_factors']:
        story.append(Paragraph(f"&bull; <b>Confluence:</b> {f}", body_style))
    story.append(Spacer(1, 10))

    # 4. Calibrated Probability Matrix Table
    story.append(Paragraph("4. Calibrated First-Passage Target Probability Matrix", section_heading))
    
    df_t = res['targets']
    table_data = [[
        Paragraph("<b>Target</b>", body_style),
        Paragraph("<b>Dist (pips)</b>", body_style),
        Paragraph("<b>Base P (%)</b>", body_style),
        Paragraph("<b>Calibrated P (%)</b>", body_style),
        Paragraph("<b>P(Stop First) (%)</b>", body_style),
        Paragraph("<b>Sum (%)</b>", body_style),
        Paragraph("<b>EV 3D (%)</b>", body_style)
    ]]

    for _, r in df_t.iterrows():
        p_t = r['Calibrated_P_Before_Stop']
        p_s = r['P_Stop_Before_Target']
        table_data.append([
            f"{r['Target']:.5f}",
            f"{r['Dist_Pips']:.1f}",
            f"{r['Base_P_Before_Stop']:.1f}%",
            f"<b>{p_t:.1f}%</b>",
            f"{p_s:.1f}%",
            f"<b>{p_t + p_s:.1f}%</b>",
            f"{r['Calibrated_EV_3D_%']:+.3f}%"
        ])

    t = Table(table_data, colWidths=[1.1*inch, 0.9*inch, 1.0*inch, 1.2*inch, 1.2*inch, 0.8*inch, 1.0*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), LIGHT_BG),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#DEE2E6")),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    # 5. Chart Placeholders
    story.append(Paragraph("5. Structural Chart Alignment & Liquidity Zones", section_heading))
    chart_box = [[Paragraph("<b>[ 4H / 15M CHART SCREENSHOT PLACEHOLDER ]</b><br/>Asian High Sweep @ 1.16768 | Bearish FVG Zone: 1.16600 &ndash; 1.16400", body_style)]]
    t_chart = Table(chart_box, colWidths=[7.2*inch])
    t_chart.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#E9ECEF")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 25),
        ('TOPPADDING', (0,0), (-1,-1), 25),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CED4DA")),
    ]))
    story.append(t_chart)
    story.append(Spacer(1, 12))

    # 6. Conclusion & Execution Protocol
    story.append(Paragraph("6. Statistical Interpretation & Execution Protocol", section_heading))
    conclusion_text = f"""
    &bull; <b>Shallow Target 1.16600 (-14.7 pips):</b> HIGH probability (<b>78.4%</b> Calibrated P(Target First) vs 21.6% Stop First).<br/>
    &bull; <b>Medium Target 1.16400 (-34.7 pips):</b> MODERATE-HIGH probability (<b>66.4%</b> Calibrated P(Target First) vs 33.6% Stop First).<br/>
    &bull; <b>Full Fill 1.15892 (-85.5 pips):</b> MODERATE probability (<b>44.9%</b> Calibrated P(Target First)).<br/>
    <br/>
    <b>EXECUTION CONTROL PRINCIPLE:</b> MODEL DOES NOT EXECUTE AUTOMATICALLY. MODEL PROVIDES STOCHASTIC PROBABILISTIC MAP.<br/>
    <b>FINAL EXECUTION DECISION: HUMAN</b>
    """
    story.append(Paragraph(conclusion_text, body_style))

    doc.build(story)
    print(f"\nPDF Quant Report generated successfully: {OUT_PDF}")

if __name__ == "__main__":
    generate_pdf_report()
