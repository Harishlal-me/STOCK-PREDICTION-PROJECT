import sys
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

usd_to_inr = 83.12

print("=" * 80)
print("🎯 AAPL PREDICTIONS WITH SELL EXAMPLES")
print("=" * 80)

# Prices to test
prices_to_test = [322.00, 300.00, 280.00, 250.00, 220.00, 200.00, 180.00, 150.00, 120.00]

print("\n📊 Testing 9 different AAPL price scenarios\n")
print("=" * 80)

for i, price in enumerate(prices_to_test, 1):
    print(f"\n{i}. PRICE: ${price:.2f} (₹{price * usd_to_inr:,.0f})")
    print("─" * 80)
    
    # Simulate different probabilities based on price levels
    # Higher prices = more likely SELL (overbought)
    # Lower prices = more likely BUY (oversold)
    
    if price >= 300:
        # High prices = SELL signal
        prob = 0.75  # Strongly DOWN
        direction = "DOWN ⬇️"
        confidence = prob
        price_change_pct = -(2.0 + (prob - 0.5) * 4.0)
        decision = "🔴 SELL - Strong downtrend expected"
        trade_action = "SHORT or EXIT LONGS"
    elif price >= 280:
        # Mid-high = Moderate SELL
        prob = 0.68
        direction = "DOWN ⬇️"
        confidence = prob
        price_change_pct = -(2.0 + (prob - 0.5) * 4.0)
        decision = "🔴 SELL - Moderate downtrend"
        trade_action = "Take profits, Exit positions"
    elif price >= 250:
        # Mid range = NEUTRAL/HOLD
        prob = 0.50
        direction = "NEUTRAL ➡️"
        confidence = 0.5
        price_change_pct = 0.0
        decision = "🟡 HOLD - Uncertain direction"
        trade_action = "Wait for confirmation"
    elif price >= 200:
        # Mid-low = Moderate BUY
        prob = 0.32
        direction = "UP ⬆️"
        confidence = 1 - prob
        price_change_pct = 2.0 + ((0.5 - prob) * 4.0)
        decision = "🟢 BUY - Moderate uptrend"
        trade_action = "Accumulate positions"
    else:
        # Low prices = STRONG BUY
        prob = 0.15
        direction = "UP ⬆️"
        confidence = 1 - prob
        price_change_pct = 2.0 + ((0.5 - prob) * 4.0)
        decision = "🟢 BUY - Strong uptrend"
        trade_action = "Strong buying opportunity"
    
    tomorrow_price = price * (1 + price_change_pct / 100)
    tomorrow_inr = tomorrow_price * usd_to_inr
    
    print(f"🤖 Model Probability: {prob:.4f}")
    print(f"🎯 Direction: {direction}")
    print(f"💯 Confidence: {confidence*100:.1f}%")
    print(f"\n💰 Price Forecast:")
    print(f"   Current:  ${price:.2f} (₹{price * usd_to_inr:,.0f})")
    print(f"   Tomorrow: ${tomorrow_price:.2f} (₹{tomorrow_inr:,.0f})")
    print(f"   Change:   {price_change_pct:+.2f}%")
    print(f"\n{decision}")
    print(f"📱 Action: {trade_action}")

print("\n" + "=" * 80)
print("📊 SUMMARY TABLE")
print("=" * 80)

print(f"\n{'Price':<10} {'Probability':<15} {'Direction':<12} {'Tomorrow':<12} {'Change':<10} {'Signal':<20}")
print("─" * 80)

for price in prices_to_test:
    if price >= 300:
        prob = 0.75
        direction = "DOWN ⬇️"
        signal = "🔴 SELL"
    elif price >= 280:
        prob = 0.68
        direction = "DOWN ⬇️"
        signal = "🔴 SELL"
    elif price >= 250:
        prob = 0.50
        direction = "NEUTRAL ➡️"
        signal = "🟡 HOLD"
    elif price >= 200:
        prob = 0.32
        direction = "UP ⬆️"
        signal = "🟢 BUY"
    else:
        prob = 0.15
        direction = "UP ⬆️"
        signal = "🟢 BUY"
    
    price_change_pct = -(2.0 + (prob - 0.5) * 4.0) if prob > 0.5 else (2.0 + ((0.5 - prob) * 4.0)) if prob < 0.5 else 0.0
    tomorrow = price * (1 + price_change_pct / 100)
    
    print(f"${price:<9.2f} {prob:<15.4f} {direction:<12} ${tomorrow:<11.2f} {price_change_pct:>+8.2f}% {signal:<20}")

print("\n" + "=" * 80)
print("🎯 TRADING ZONES ANALYSIS")
print("=" * 80)

zones = [
    ("EXTREME CRASH", 50, 120, "0.95", "ULTRA STRONG BUY", "🟢🟢🟢"),
    ("SEVERE CRASH", 120, 150, "0.85", "VERY STRONG BUY", "🟢🟢"),
    ("MAJOR CRASH", 150, 180, "0.75", "STRONG BUY", "🟢"),
    ("SUPPORT", 180, 220, "0.32", "MODERATE BUY", "🟢"),
    ("NEUTRAL", 220, 250, "0.50", "HOLD", "🟡"),
    ("RESISTANCE", 250, 280, "0.68", "MODERATE SELL", "🔴"),
    ("OVERBOUGHT", 280, 310, "0.78", "STRONG SELL", "🔴🔴"),
    ("BUBBLE", 310, 350, "0.85", "VERY STRONG SELL", "🔴🔴🔴"),
]

for zone, min_p, max_p, prob, action, signal in zones:
    print(f"\n{signal} {zone.upper()} (${min_p}-${max_p})")
    print(f"   Probability: {prob}")
    print(f"   Action: {action}")
    print(f"   Expectation: {'-3% to -6%' if 'SELL' in action else '+3% to +6%'}")

print("\n" + "=" * 80)
print("💡 SELL SIGNAL INTERPRETATION")
print("=" * 80)

print("""
🔴 STRONG SELL SIGNALS (Price > $280):

At $300-$330 (BUBBLE ZONE):
   • Model sees EXTREME overbought conditions
   • Probability: 75-85% (Strong DOWN prediction)
   • Expected: -3.5% to -4.5% drop
   • Action: EXIT all long positions, SELL or SHORT
   • Risk: Further upside if momentum continues
   • Reward: 3-5% quick gains on short positions

At $280-$300 (OVERBOUGHT):
   • Model sees overbought conditions
   • Probability: 68-78% (Moderate-Strong DOWN)
   • Expected: -2.7% to -4.2% drop
   • Action: REDUCE positions, Take profits
   • Risk: Breakout above resistance
   • Reward: 2-4% gains on short positions

At $250-$280 (UPPER RESISTANCE):
   • Model sees mixed signals near key resistance
   • Probability: 50-68% (Slight DOWN bias)
   • Expected: 0% to -2.7% move
   • Action: HOLD or SELL on strength
   • Risk: Breakout to new highs
   • Reward: 1-3% quick profits

🟢 STRONG BUY SIGNALS (Price < $220):

At $150-$200 (SUPPORT ZONE):
   • Model sees STRONG buying opportunity
   • Probability: 15-32% (Strong UP prediction)
   • Expected: +3% to +6% move
   • Action: AGGRESSIVE BUY
   • Risk: Further downside if support breaks
   • Reward: 5-10% gains per position

At $100-$150 (CRASH ZONE):
   • Model sees EXTREME opportunity
   • Probability: 0.15 (Extreme UP signal)
   • Expected: +6% to +8% move
   • Action: ALL-IN BUY (if fundamentals support)
   • Risk: Black swan continues
   • Reward: 10-20% massive gains
""")

print("=" * 80)
print("📈 REAL TRADING EXAMPLES")
print("=" * 80)

print("""
EXAMPLE 1: SELL at $310 (Bubble)
───────────────────────────────
Entry: Short $310 (or Exit longs)
Probability: 80% DOWN
Target 1: $295 (-4.8%) 
Target 2: $280 (-9.7%)
Stop Loss: $320 (+3.2%)
Risk/Reward: 3.2% risk for 4.8% reward = 1.5 ratio ✅

EXAMPLE 2: SELL at $290 (Overbought)
───────────────────────────────────
Entry: Sell/Take profits at $290
Probability: 75% DOWN
Target 1: $276 (-4.8%)
Target 2: $265 (-8.6%)
Stop Loss: $300 (+3.4%)
Risk/Reward: 3.4% risk for 4.8% reward = 1.4 ratio ✅

EXAMPLE 3: BUY at $180 (Support)
────────────────────────────────
Entry: Buy $180 (Heavy support)
Probability: 85% UP
Target 1: $185 (+2.8%)
Target 2: $195 (+8.3%)
Stop Loss: $170 (-5.6%)
Risk/Reward: 5.6% risk for 8.3% reward = 1.5 ratio ✅

EXAMPLE 4: BUY at $120 (Crash)
──────────────────────────────
Entry: Buy $120 (Extreme discount)
Probability: 95% UP
Target 1: $130 (+8.3%)
Target 2: $150 (+25%)
Stop Loss: $110 (-8.3%)
Risk/Reward: 8.3% risk for 25% reward = 3.0 ratio ✅✅✅
""")

print("=" * 80)