"""
app.py — Trading Bot Dashboard Server
Run: python app.py
Then open: http://localhost:5000
"""

from flask import Flask, render_template, jsonify, request, session, redirect, url_for
from datetime import datetime
from functools import wraps
import os
from dotenv import load_dotenv
import database
from strategies import momentum_breakout as strategy
from strategies import mean_reversion as t2

load_dotenv()

app = Flask(__name__)

# ─── Auth ─────────────────────────────────────────────────────────────────────
app.secret_key = os.environ.get("SECRET_KEY", "tradebot-secret-change-this-2026")
DASHBOARD_USER = os.environ.get("DASHBOARD_USER", "admin")
DASHBOARD_PASS = os.environ.get("DASHBOARD_PASS", "changeme")
STARTING_CAPITAL = 20_000


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)

    return decorated


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        u = request.form.get("username", "").strip()
        p = request.form.get("password", "").strip()
        if u == DASHBOARD_USER and p == DASHBOARD_PASS:
            session["logged_in"] = True
            return redirect(url_for("dashboard"))
        error = "Invalid credentials"
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ─── Pages ────────────────────────────────────────────────────────────────────


@app.route("/")
@login_required
def dashboard():
    return render_template("dashboard.html")


# ─── API ──────────────────────────────────────────────────────────────────────


@app.route("/api/stock_count")
@login_required
def api_stock_count():
    """Returns the current number of stocks in the universe."""
    return jsonify({"count": len(strategy.STOCKS)})


@app.route("/api/macro")
@login_required
def api_macro():
    """Returns current macro environment snapshot."""
    from prediction import macro_sentiment

    return jsonify(macro_sentiment.compute())


@app.route("/api/score_breakdown/<symbol>")
@login_required
def api_score_breakdown(symbol):
    """Returns the full quality score breakdown for a given stock."""
    if not symbol.endswith(".NS"):
        symbol += ".NS"
    from prediction import score as scorer

    df = strategy._fetch(symbol, "3mo")
    sector_sym = strategy.SECTOR_INDEX.get(symbol, strategy.NIFTY)
    sector_df = strategy._fetch(sector_sym, "3mo")
    result = scorer.compute_score(symbol, df, sector_df)
    return jsonify(result)


@app.route("/api/smart_money/<symbol>")
@login_required
def api_smart_money(symbol):
    """Returns smart money breakdown for a stock."""
    if not symbol.endswith(".NS"):
        symbol += ".NS"
    from prediction import smart_money

    return jsonify(smart_money.compute(symbol))


@app.route("/api/summary")
@login_required
def api_summary():
    positions = database.get_open_positions()
    # Exclude paper parallel from summary — only show live positions
    positions = [p for p in positions if p.get("mode") != "PAPER_PARALLEL"]
    stats = database.get_stats()
    todays_pnl = database.get_todays_pnl()

    open_pnl = 0
    invested = 0
    for pos in positions:
        price = strategy.get_current_price(pos["symbol"])
        open_pnl += price * pos["quantity"] - pos["buy_price"] * pos["quantity"]
        invested += pos["buy_price"] * pos["quantity"]

    closed_pnl = stats["total_pnl"] or 0
    total_pnl = closed_pnl + open_pnl
    cash = STARTING_CAPITAL - invested + closed_pnl
    portfolio = STARTING_CAPITAL + total_pnl

    regime = strategy.get_market_regime()
    nifty = strategy.get_nifty_info()
    bot_paused = todays_pnl <= -1000

    regime_labels = {
        "BULL": ("Bull Market", "green"),
        "CHOP": ("Choppy Market", "yellow"),
        "BEAR": ("Bear Market — No New Buys", "red"),
    }
    reg_label, reg_color = regime_labels.get(regime, ("Unknown", "grey"))

    n = stats["n"] or 0
    wins = stats["wins"] or 0
    win_rate = round(wins / n * 100, 1) if n > 0 else 0

    return jsonify(
        {
            "portfolio_value": round(portfolio, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl / STARTING_CAPITAL * 100, 2),
            "open_pnl": round(open_pnl, 2),
            "closed_pnl": round(closed_pnl, 2),
            "cash_available": round(cash, 2),
            "invested": round(invested, 2),
            "open_positions": len(positions),
            "total_trades": n,
            "win_rate": win_rate,
            "wins": wins,
            "losses": stats["losses"] or 0,
            "bot_paused": bot_paused,
            "regime": regime,
            "regime_label": reg_label,
            "regime_color": reg_color,
            "nifty_price": nifty["price"],
            "nifty_change": nifty["change_pct"],
            "todays_pnl": round(todays_pnl, 2),
            "as_of": datetime.now().strftime("%d %b %Y, %I:%M %p"),
            "max_positions": strategy.CFG["max_positions"],
            "stock_count": len(strategy.STOCKS),
        }
    )


@app.route("/api/holdings")
@login_required
def api_holdings():
    positions = database.get_open_positions()
    positions = [p for p in positions if p.get("mode") != "PAPER_PARALLEL"]
    result = []
    for pos in positions:
        price = strategy.get_current_price(pos["symbol"])
        bp = pos["buy_price"]
        qty = pos["quantity"]
        cost = bp * qty
        cur_val = price * qty
        pnl = cur_val - cost
        pnl_pct = (pnl / cost * 100) if cost > 0 else 0
        sl = bp * 0.96
        tp = bp * 1.08
        progress = ((price - sl) / (tp - sl) * 100) if tp != sl else 50
        result.append(
            {
                "id": pos["id"],
                "symbol": pos["symbol"],
                "display_name": pos["symbol"].replace(".NS", ""),
                "quantity": qty,
                "buy_price": round(bp, 2),
                "current_price": round(price, 2),
                "invested": round(cost, 2),
                "current_value": round(cur_val, 2),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "stop_loss": round(sl, 2),
                "target": round(tp, 2),
                "progress": round(min(max(progress, 0), 100), 1),
                "days_held": pos["days_held"],
                "buy_date": pos["buy_date"],
                "mode": pos["mode"],
            }
        )
    return jsonify(result)


@app.route("/api/exits")
@login_required
def api_exits():
    return jsonify(strategy.check_exits())


@app.route("/api/signals")
@login_required
def api_signals():
    result = strategy.scan_for_signals()
    return jsonify(result)


@app.route("/api/signals/history")
@login_required
def api_signals_history():
    return jsonify(database.get_recent_signals(30))


@app.route("/api/trade_history")
@login_required
def api_trade_history():
    return jsonify(database.get_closed_trades())


@app.route("/api/portfolio_chart")
@login_required
def api_portfolio_chart():
    history = database.get_portfolio_history()
    return jsonify(history if history else [])


@app.route("/api/add_trade", methods=["POST"])
@login_required
def api_add_trade():
    data = request.json
    symbol = data.get("symbol", "").upper().strip()
    if not symbol.endswith(".NS"):
        symbol += ".NS"
    qty = int(data.get("quantity", 0))
    price = float(data.get("price", 0))
    mode = data.get("mode", "PAPER")
    if not symbol or qty < 1 or price <= 0:
        return jsonify({"error": "Invalid trade data"}), 400
    database.add_trade(symbol, qty, price, mode)
    summary = api_summary().get_json()
    database.save_portfolio_value(summary["portfolio_value"])
    return jsonify({"message": f"Trade recorded: {qty} x {symbol} @ Rs{price}"})


@app.route("/api/close_trade", methods=["POST"])
@login_required
def api_close_trade():
    data = request.json
    trade_id = int(data.get("trade_id", 0))
    price = float(data.get("price", 0))
    reason = data.get("reason", "manual")
    if not trade_id or price <= 0:
        return jsonify({"error": "Invalid data"}), 400
    pnl = database.close_trade(trade_id, price, reason)

    # Log outcome to sentiment validator
    try:
        from prediction.sentiment_validator import log_outcome as sv_log

        trade = database.get_trade_by_id(trade_id)
        if trade:
            outcome = "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "TIME"
            sv_log(trade["symbol"], trade["buy_date"], outcome, pnl)
    except Exception:
        pass

    # Log outcome to ML collector
    try:
        from ml.collector import log_trade_outcome

        trade = database.get_trade_by_id(trade_id)
        if trade:
            outcome = "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "TIME"
            pnl_pct = round((price - trade["buy_price"]) / trade["buy_price"] * 100, 2)
            log_trade_outcome(
                trade["symbol"],
                trade["buy_date"],
                outcome,
                pnl,
                pnl_pct,
                trade.get("days_held", 0),
                reason,
            )
    except Exception:
        pass

    summary = api_summary().get_json()
    database.save_portfolio_value(summary["portfolio_value"])
    return jsonify({"message": f"Trade closed. P&L: Rs{pnl:+.0f}", "pnl": pnl})


# ─── ML API endpoints ─────────────────────────────────────────────────────────


@app.route("/api/ml/status")
@login_required
def api_ml_status():
    """Returns ML data collection status and sample counts."""
    from ml.collector import get_sample_count

    return jsonify(get_sample_count())


@app.route("/api/ml/report")
@login_required
def api_ml_report():
    """Full ML evaluation report — validation + feature importance."""
    from ml.evaluator import full_report

    return jsonify(full_report())


@app.route("/api/ml/train", methods=["POST"])
@login_required
def api_ml_train():
    """Trigger model training manually."""
    from ml.trainer import train_and_validate, save_model

    result = train_and_validate()
    if result.get("deploy") and result.get("model"):
        save_model(result["model"], result.get("features", []), result)
        result.pop("model")  # don't serialize model object
        result["saved"] = True
    else:
        result.pop("model", None)
    return jsonify(result)


@app.route("/api/sentiment/report")
@login_required
def api_sentiment_report():
    """Monthly sentiment validation report."""
    from prediction.sentiment_validator import monthly_report

    return jsonify(monthly_report())


@app.route("/api/sentiment/logs")
@login_required
def api_sentiment_logs():
    """Recent sentiment validation log entries."""
    from prediction.sentiment_validator import get_recent_logs

    return jsonify(get_recent_logs(20))


@app.route("/api/snapshot", methods=["POST"])
@login_required
def api_snapshot():
    summary = api_summary().get_json()
    database.save_portfolio_value(summary["portfolio_value"])
    return jsonify(
        {"message": "Portfolio value saved.", "value": summary["portfolio_value"]}
    )


# ─── Mean Reversion (Track 2) API ─────────────────────────────────────────────


@app.route("/api/t2/summary")
@login_required
def api_t2_summary():
    return jsonify(t2.get_summary())


@app.route("/api/t2/holdings")
@login_required
def api_t2_holdings():
    positions = t2.get_open_positions()
    result = []
    for pos in positions:
        price = t2._current_price(pos["symbol"]) or pos["buy_price"]
        bp = pos["buy_price"]
        qty = pos["quantity"]
        pnl = round((price - bp) * qty, 2)
        pnl_pct = round((price - bp) / bp * 100, 2)
        sl = round(bp * (1 - t2.CFG["stop_loss_pct"]), 2)
        tp = round(bp * (1 + t2.CFG["profit_target_pct"]), 2)
        progress = round(min(max((price - sl) / (tp - sl) * 100, 0), 100), 1)
        result.append(
            {
                **pos,
                "display_name": pos["symbol"].replace(".NS", ""),
                "current_price": round(price, 2),
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "stop_loss": sl,
                "target": tp,
                "progress": progress,
            }
        )
    return jsonify(result)


@app.route("/api/t2/history")
@login_required
def api_t2_history():
    return jsonify(t2.get_closed_trades())


@app.route("/api/t2/chart")
@login_required
def api_t2_chart():
    return jsonify(t2.get_portfolio_history())


@app.route("/api/t2/run", methods=["POST"])
@login_required
def api_t2_run():
    result = t2.run_daily()
    return jsonify(result)


@app.route("/api/comparison")
@login_required
def api_comparison():
    positions1 = database.get_open_positions()
    positions1 = [p for p in positions1 if p.get("mode") != "PAPER_PARALLEL"]
    stats1 = database.get_stats()
    open_pnl1 = 0
    for pos in positions1:
        price = strategy.get_current_price(pos["symbol"])
        open_pnl1 += (price - pos["buy_price"]) * pos["quantity"]
    closed_pnl1 = stats1["total_pnl"] or 0
    total_pnl1 = closed_pnl1 + open_pnl1
    n1 = stats1["n"] or 0
    wr1 = round((stats1["wins"] or 0) / n1 * 100, 1) if n1 > 0 else 0

    t2_sum = t2.get_summary()
    hist1 = database.get_portfolio_history()
    hist2 = t2.get_portfolio_history()

    return jsonify(
        {
            "track1": {
                "label": "Momentum Breakout",
                "starting": 20_000,
                "portfolio_value": round(20_000 + total_pnl1, 2),
                "total_pnl": round(total_pnl1, 2),
                "total_pnl_pct": round(total_pnl1 / 20_000 * 100, 2),
                "total_trades": n1,
                "win_rate": wr1,
                "open_positions": len(positions1),
                "mode": "LIVE (manual orders)",
                "strategy": "Breakout + Volume + RSI + ADX + Sector filter",
                "stock_count": len(strategy.STOCKS),
            },
            "track2": {
                "label": "Mean Reversion",
                "starting": 50_000,
                "portfolio_value": t2_sum["portfolio_value"],
                "total_pnl": t2_sum["total_pnl"],
                "total_pnl_pct": t2_sum["total_pnl_pct"],
                "total_trades": t2_sum["total_trades"],
                "win_rate": t2_sum["win_rate"],
                "open_positions": t2_sum["open_positions"],
                "mode": "PAPER (automated)",
                "strategy": "Dip 10-25% + RSI<45 + Quality fundamentals",
            },
            "chart1": hist1,
            "chart2": hist2,
        }
    )


if __name__ == "__main__":
    database.init_db()
    t2.init_db()
    print("\n  Trading Bot — Momentum Breakout + Mean Reversion")
    print(f"  Universe: {len(strategy.STOCKS)} stocks")
    print("  Open: http://localhost:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
