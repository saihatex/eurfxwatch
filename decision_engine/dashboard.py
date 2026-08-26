import os
import sys

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from decision_engine.engine import EURUSDDecisionEngine

def run_dashboard(date_str):
    engine = EURUSDDecisionEngine()
    res = engine.evaluate(date_str)

    print("="*60)
    print(f"EUR/USD MULTI-FACTOR DECISION ENGINE")
    print(f"Date: {res['date']} | EUR/USD Close: {res['eur_close']:.5f}")
    print("="*60)

    print("\nREGIME IDENTIFICATION (LAYER 1)")
    print(f"Volatility:          {res['regime']['volatility']}")
    print(f"Trend:               {res['regime']['trend']}")
    print(f"Momentum Percentile: {res['regime']['x1_mom_pctl']}%")
    print(f"Vol 3Y Percentile:   {res['regime']['x2_vol_pctl']}%")
    print(f"US2Y / DE2Y Spread:  {res['regime']['spread']:+.4f} (5D Chg X4: {res['regime']['x4_rates']:+.4f})")

    print("\n" + "-"*60)
    print("DIRECTIONAL EVIDENCE SCORES (LAYER 2)")
    print("-"*60)
    
    print(f"\nLONG FACTORS (Score: {res['scores']['long_score']}):")
    if res['scores']['long_factors']:
        for f in res['scores']['long_factors']:
            print(f"  + {f}")
    else:
        print("  - None")

    print(f"\nSHORT FACTORS (Score: {res['scores']['short_score']}):")
    if res['scores']['short_factors']:
        for f in res['scores']['short_factors']:
            print(f"  + {f}")
    else:
        print("  - None")

    print("\n" + "-"*60)
    print("CONDITIONAL DISTRIBUTIONS & EXPECTED RETURNS (LAYER 3)")
    print("-"*60)
    
    print("\nLONG THESIS:")
    print(f"  Expected 3D Return: {res['distributions']['long']['expected_3d']:+.2f}%")
    print(f"  Median 3D Return:   {res['distributions']['long']['median_3d']:+.2f}%")
    print(f"  P(Positive Return): {res['distributions']['long']['p_positive']:.1f}%")

    print("\nSHORT THESIS:")
    print(f"  Expected 3D Return: {res['distributions']['short']['expected_3d']:+.2f}%")
    print(f"  Median 3D Return:   {res['distributions']['short']['median_3d']:+.2f}%")
    print(f"  P(Negative Return): {res['distributions']['short']['p_negative']:.1f}%")

    print("\n" + "="*60)
    print(f"FINAL DECISION:       >>> {res['decision']['final_signal']} <<<")
    print(f"CONFIDENCE LEVEL:     {res['decision']['confidence']}")
    print("="*60 + "\n")

if __name__ == "__main__":
    test_date = sys.argv[1] if len(sys.argv) > 1 else "2026-08-18"
    run_dashboard(test_date)
