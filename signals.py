"""
signals.py
----------
Generates trading signals by combining price prediction + news sentiment.
"""

from dataclasses import dataclass


@dataclass
class TradeSignal:
    action:      str    # "BUY" | "SELL" | "HOLD"
    confidence:  str    # "HIGH" | "MEDIUM" | "LOW"
    change_pct:  float  # Predicted % move
    sentiment:   float  # Sentiment score [-1, 1]
    stop_loss:   float
    take_profit: float
    reason:      str


def generate_signal(
    current_price:     float,
    predicted_price:   float,
    sentiment_score:   float = 0.0,
    strong_threshold:  float = 0.03,
    weak_threshold:    float = 0.008,
    sentiment_confirm: float = 0.05,   # min sentiment to confirm direction
    stop_loss_pct:     float = 0.05,
    take_profit_pct:   float = 0.10,
) -> TradeSignal:
    """
    Generate a sentiment-aware trading signal.

    Signal logic:
      BUY  → price prediction UP  + sentiment not strongly negative
      SELL → price prediction DOWN + sentiment not strongly positive
      HOLD → conflicting signals or change within neutral band

    Args:
        current_price:     Latest BTC close in USD
        predicted_price:   Next-day predicted price in USD
        sentiment_score:   VADER compound score in [-1, 1]
        strong_threshold:  % change for HIGH confidence
        weak_threshold:    % change to trigger BUY/SELL (vs HOLD)
        sentiment_confirm: Sentiment must exceed this to avoid conflict
        stop_loss_pct:     % for stop-loss placement
        take_profit_pct:   % for take-profit placement

    Returns:
        TradeSignal dataclass
    """
    change_pct = (predicted_price - current_price) / current_price

    # Determine base action from price delta
    if change_pct > weak_threshold:
        # Sentiment conflict dampens to HOLD
        if sentiment_score < -sentiment_confirm:
            action = "HOLD"
            reason = (
                f"Model predicts +{change_pct*100:.2f}% but sentiment is "
                f"bearish ({sentiment_score:+.3f}). Signals conflict."
            )
        else:
            action = "BUY"
            reason = (
                f"Model predicts +{change_pct*100:.2f}% rise to "
                f"${predicted_price:,.2f}. "
                f"Sentiment: {sentiment_score:+.3f} (confirming)."
            )

    elif change_pct < -weak_threshold:
        if sentiment_score > sentiment_confirm:
            action = "HOLD"
            reason = (
                f"Model predicts {change_pct*100:.2f}% drop but sentiment is "
                f"bullish ({sentiment_score:+.3f}). Signals conflict."
            )
        else:
            action = "SELL"
            reason = (
                f"Model predicts {change_pct*100:.2f}% drop to "
                f"${predicted_price:,.2f}. "
                f"Sentiment: {sentiment_score:+.3f} (confirming)."
            )

    else:
        action = "HOLD"
        reason = (
            f"Predicted change ({change_pct*100:.2f}%) within neutral band "
            f"(±{weak_threshold*100:.1f}%). Sentiment: {sentiment_score:+.3f}."
        )

    # Stops
    if action == "BUY":
        stop_loss   = current_price * (1 - stop_loss_pct)
        take_profit = current_price * (1 + take_profit_pct)
    elif action == "SELL":
        stop_loss   = current_price * (1 + stop_loss_pct)
        take_profit = current_price * (1 - take_profit_pct)
    else:
        stop_loss   = current_price * (1 - stop_loss_pct)
        take_profit = current_price * (1 + take_profit_pct)

    # Confidence (boosted when price + sentiment agree)
    abs_change = abs(change_pct)
    sentiment_aligns = (
        (change_pct > 0 and sentiment_score > sentiment_confirm) or
        (change_pct < 0 and sentiment_score < -sentiment_confirm)
    )

    if abs_change >= strong_threshold and sentiment_aligns:
        confidence = "HIGH"
    elif abs_change >= strong_threshold or sentiment_aligns:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return TradeSignal(
        action=action,
        confidence=confidence,
        change_pct=round(change_pct * 100, 3),
        sentiment=sentiment_score,
        stop_loss=round(stop_loss, 2),
        take_profit=round(take_profit, 2),
        reason=reason,
    )
