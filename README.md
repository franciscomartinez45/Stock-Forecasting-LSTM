# Stock Forecasting with LSTM + Sentiment Analysis

A two-stage pipeline that combines financial news sentiment with historical price data to forecast next-day stock movements for **Apple (AAPL)** and **Microsoft (MSFT)**.

**Stage 1 (complete):** Build a daily sentiment dataset from financial news headlines using FinBERT.  
**Stage 2 (in progress):** Train an LSTM model on the combined sentiment + price features to predict next-day price direction.

---

## How It Works

```
Polygon.io News API ──► FinBERT (ProsusAI) ──► Daily Sentiment Scores ─┐
                                                                          ├──► LSTM Model ──► Price Direction
yfinance Stock Prices ──────────────────────────────────────────────────┘
```

For each trading day, FinBERT classifies every news headline about the stock as **positive**, **neutral**, or **negative**. These per-article scores are averaged to produce a daily sentiment profile, then joined with closing price data. The resulting dataset is what the LSTM will train on.

---

## Repository Structure

```
Stock-Forecasting-LSTM/
├── notebooks/
│   └── Sentiment_Analysis.ipynb       # Stage 1: news collection + FinBERT scoring pipeline
├── data/
│   ├── raw/
│   │   ├── aapl_news_2023.csv         # Raw AAPL headlines from Polygon.io (full year 2023)
│   │   └── msft_news_2023.csv         # Raw MSFT headlines from Polygon.io (full year 2023)
│   └── processed/
│       ├── aapl_daily_sentiment_2023.csv  # Daily sentiment + price features for AAPL
│       └── msft_daily_sentiment_2023.csv  # Daily sentiment + price features for MSFT
└── LICENSE
```

---

## Dataset Schema

`aapl_daily_sentiment_2023.csv` and `msft_daily_sentiment_2023.csv` share the same schema:

| Column | Description |
|---|---|
| `date` | Trading date (YYYY-MM-DD) |
| `n_articles` | Number of news articles that day |
| `avg_negative` | Mean FinBERT negative score across all articles |
| `avg_neutral` | Mean FinBERT neutral score across all articles |
| `avg_positive` | Mean FinBERT positive score across all articles |
| `close` | Closing price on this date |
| `next_close` | Closing price on the next trading day |
| `next_day_pct` | Percent change from `close` to `next_close` |
| `direction` | `UP` or `DOWN` — the classification target |

---

## Setup

### Prerequisites

- Python 3.9+
- A [Polygon.io](https://polygon.io) API key (free tier works, but rate-limited to 5 req/min)
- A [Hugging Face](https://huggingface.co/settings/tokens) token (for FinBERT inference)
- GPU recommended for FinBERT batch inference (the notebook uses `device=0`)

### Google Colab (recommended)

The notebook is designed to run in Google Colab with GPU runtime.

1. Open the notebook via the badge at the top of `Sentiment_Analysis.ipynb`.
2. Enable GPU: **Runtime → Change runtime type → T4 GPU**.
3. Add secrets in Colab (**Key icon → Secrets**):
   - `HF_TOKEN` — your Hugging Face token
   - `POLYGON_API_KEY` — your Polygon.io API key
4. Run all cells.

### Local Setup

```bash
git clone https://github.com/franciscomartinez45/Stock-Forecasting-LSTM.git
cd Stock-Forecasting-LSTM
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install transformers torch yfinance pandas requests
```

Set your API keys as environment variables before running:

```bash
export HF_TOKEN=your_token_here
export POLYGON_API_KEY=your_key_here
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `transformers` | FinBERT model and tokenizer |
| `torch` | GPU inference backend |
| `yfinance` | Historical stock price data |
| `pandas` | Data manipulation |
| `requests` | Polygon.io API calls |

---

## Data Sources

- **[Polygon.io News API](https://polygon.io/docs/stocks/get_v2_reference_news)** — financial news headlines, filtered by ticker and date
- **[yfinance](https://github.com/ranaroussi/yfinance)** — daily OHLCV price data from Yahoo Finance
- **[ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)** — FinBERT, a BERT model fine-tuned on financial text for 3-class sentiment classification

---

## Roadmap

- [x] Collect raw news headlines via Polygon.io (AAPL, MSFT — full year 2023)
- [x] Score headlines with FinBERT and aggregate to daily sentiment features
- [x] Join sentiment features with next-day price direction labels
- [ ] Build and train LSTM on combined sentiment + price features
- [ ] Evaluate model — accuracy, precision/recall on UP/DOWN classification
- [ ] Extend to additional tickers and time ranges

---

## License

MIT — see [LICENSE](LICENSE) for details.
