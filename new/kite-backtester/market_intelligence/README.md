# Market Intelligence Engine

## Overview

The **Market Intelligence Engine** is a standalone analytical subsystem within the
ML Trading Engine project.

Its purpose is **not to predict price direction or execute trades**, but to:
- Understand market context and structure
- Aggregate quantitative and qualitative signals
- Use LLMs to reason about market bias and scenarios
- Deliver human-readable insights via Telegram

This engine complements (but does not interfere with) the existing:
- ML prediction loop
- Backtesting engine
- Rule-based strategies

## Folder Structure — market_intelligence/

market_intelligence/
│
├── README.md
│
├── config/
│   ├── instruments.yaml          # symbol metadata (index/stock/forex)
│   ├── market_sessions.yaml      # exchange timings, holidays
│   ├── llm_config.yaml           # model, temperature, max tokens, enable/disable
│   ├── telegram_config.yaml      # bot token, chat id, enable/disable
│
├── context/
│   ├── time_context.py            # current datetime, timezone, session state
│   ├── session_context.py         # pre-open / open / closed logic
│   ├── instrument_context.py      # index vs stock vs forex abstraction
│
├── data_capture/
│   ├── candle_snapshot.py         # multi-TF candles (15m, 1H, 1D)
│   ├── indicator_snapshot.py      # EMA, RSI, MACD, ADX, ATR, BBW etc.
│   ├── pdh_pdl.py                 # previous day high / low
│   ├── sr_snapshot.py             # 1D support/resistance zone extraction
│
├── news/
│   ├── news_fetcher.py            # fetch raw global & regional news
│   ├── news_filter.py             # instrument-specific relevance filtering
│
├── prompts/
│   ├── prompt_start.py            # Prompt 1 (pre-market / startup)
│   ├── prompt_update.py           # Prompt 2 (TF-based live updates)
│   ├── prompt_formatter.py        # JSON + text formatting helpers
│
├── llm/
│   ├── llm_client.py              # OpenAI client wrapper
│   ├── conversation_manager.py    # history + context memory
│   ├── cost_guard.py              # usage limits & safety checks
│
├── telegram/
│   ├── notifier.py                # Telegram message formatting & sending
│
├── runners/
│   ├── pre_market_runner.py       # runs once at script start
│   ├── live_update_runner.py      # runs at every TF candle close
│
├── tests/
│   └── test_market_intelligence_flow.py

## Key Design Principles

- **Strict separation from execution and prediction**
- **Asset-agnostic** (Index / Stock / Forex / Crypto-ready)
- **LLM is optional and cost-controlled**
- **Telegram-first output**
- **Deterministic data + probabilistic reasoning**
- **Reuses core project libraries (indicators, S/R, fetchers)**

---

## What This Engine Does

### 1. One-Time Pre-Market Context Build
Executed once at script start (or market open):

- Current datetime & market session state
- Instrument metadata (index / stock / forex)
- Latest relevant global & regional news
- Multi-timeframe candles (15m, 1H)
- Full indicator snapshot (EMA, RSI, MACD, ADX, ATR, BBW, etc.)
- Previous Day High / Low (PDH / PDL)
- Daily Support & Resistance zones (via swing + clustering engine)

This context is:
- Sent directly to Telegram
- Optionally sent to an LLM for structured reasoning

---

### 2. Live Market Updates (Timeframe-Based)
Executed at every candle close (e.g. 15m):

- Capture latest candle + indicators
- Detect interaction with HTF S/R zones
- Append incremental context
- Generate updated reasoning via LLM (optional)
- Push structured insights to Telegram

---

## Folder Responsibilities

### `context/`
Builds time, session, and instrument awareness.

### `data_capture/`
Collects and structures quantitative market data.

### `news/`
Fetches and filters relevant macro and instrument-specific news.

### `prompts/`
Defines Prompt 1 (startup) and Prompt 2 (incremental updates).

### `llm/`
Handles OpenAI integration, conversation memory, and cost safety.

### `telegram/`
Formats and sends insights to Telegram.

### `runners/`
Controls execution flow:
- `pre_market_runner.py` → runs once
- `live_update_runner.py` → runs per timeframe

---

## What This Engine Explicitly Does NOT Do

- ❌ Execute trades
- ❌ Predict next candle direction
- ❌ Manage risk or position sizing
- ❌ Override deterministic strategy logic

Those responsibilities belong to other systems in the project.

---

## Typical Use Case

> “At market open, summarize the full market context and key levels.  
Then, every 15 minutes, explain what changed and what matters now.”

---

## Status

🚧 **Under active development**

Initial focus:
- Context schema
- Prompt design
- Pre-market runner
- S/R integration

---

## Future Extensions

- Streamlit / Web dashboard
- LLM-assisted trade journaling
- Multi-asset correlation reasoning
- Event-driven alerts (zone break, rejection, volatility expansion)