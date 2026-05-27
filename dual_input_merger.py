"""
Dual-Input Neural Network: Sentiment (FinBERT) + Price (LSTM) Merger
====================================================================

This module implements a complete pipeline for merging sentiment features (from FinBERT)
with historical price features (LSTM) for improved stock price prediction.

Key Components:
  1. SentimentPriceDataset: Aligns sentiment (daily) with price sequences (rolling windows)
  2. DualInputModel: Processes both inputs and merges via concatenation + fusion layer
  3. Training loop with early stopping
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA ALIGNMENT & DATASET
# ─────────────────────────────────────────────────────────────────────────────

def align_sentiment_and_prices(
    prices_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    price_col: str = "close",
    sentiment_col: str = "sentiment_score",
    date_col: str = "date"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aligns daily sentiment scores with daily prices.
    
    Args:
        prices_df: DataFrame with columns [date_col, price_col]
        sentiment_df: DataFrame with columns [date_col, sentiment_col]
        price_col: Name of price column
        sentiment_col: Name of sentiment column (e.g., from FinBERT: -1 to 1)
        date_col: Name of date column
    
    Returns:
        (prices, sentiments, dates) — aligned arrays of shape (T,)
    """
    # Ensure datetime type
    prices_df[date_col] = pd.to_datetime(prices_df[date_col])
    sentiment_df[date_col] = pd.to_datetime(sentiment_df[date_col])
    
    # Merge on date (inner join to keep only aligned dates)
    merged = prices_df.merge(
        sentiment_df[[date_col, sentiment_col]],
        on=date_col,
        how="left"
    )
    
    # Fill missing sentiment with 0 (neutral) or forward-fill
    merged[sentiment_col] = merged[sentiment_col].fillna(0.0)
    
    prices = merged[price_col].values.astype(np.float32)
    sentiments = merged[sentiment_col].values.astype(np.float32)
    dates = merged[date_col].values
    
    print(f"Aligned {len(prices)} trading days with sentiment scores")
    return prices, sentiments, dates


class SentimentPriceDataset(Dataset):
    """
    Dataset that provides:
      - Price sequence: (past_history,) — historical price windows
      - Sentiment sequence: (past_history,) — aligned sentiment for same window
      - Target: scalar close price for next day
    
    Shapes for a batch:
      - prices_seq: (batch, past_history, 1)
      - sentiment_seq: (batch, past_history, 1)
      - target: (batch,)
    """
    
    def __init__(
        self,
        prices: np.ndarray,
        sentiments: np.ndarray,
        past_history: int = 60,
        scaler_prices: StandardScaler = None,
        scaler_sentiments: StandardScaler = None,
    ):
        """
        Args:
            prices: Array of shape (T,) with daily close prices
            sentiments: Array of shape (T,) with daily sentiment scores (-1 to 1)
            past_history: Number of historical days per window
            scaler_prices: Pre-fit StandardScaler for prices (from training set)
            scaler_sentiments: Pre-fit StandardScaler for sentiments (from training set)
        """
        assert len(prices) == len(sentiments), "Prices and sentiments must have same length"
        
        self.past_history = past_history
        self.prices = prices
        self.sentiments = sentiments
        
        # Normalize using provided scalers or fit on full data (training only)
        if scaler_prices is None:
            scaler_prices = StandardScaler()
            scaler_prices.fit(prices.reshape(-1, 1))
        if scaler_sentiments is None:
            scaler_sentiments = StandardScaler()
            scaler_sentiments.fit(sentiments.reshape(-1, 1))
        
        self.scaler_prices = scaler_prices
        self.scaler_sentiments = scaler_sentiments
        
        # Normalize both arrays
        self.prices_scaled = scaler_prices.transform(prices.reshape(-1, 1)).flatten()
        self.sentiments_scaled = scaler_sentiments.transform(sentiments.reshape(-1, 1)).flatten()
    
    def __len__(self):
        return len(self.prices) - self.past_history
    
    def __getitem__(self, idx):
        # Historical window
        price_window = self.prices_scaled[idx : idx + self.past_history]
        sentiment_window = self.sentiments_scaled[idx : idx + self.past_history]
        
        # Target (next day's price)
        target_price = self.prices_scaled[idx + self.past_history]
        
        # Convert to tensors with shape (seq_len, 1)
        price_tensor = torch.tensor(price_window, dtype=torch.float32).unsqueeze(-1)
        sentiment_tensor = torch.tensor(sentiment_window, dtype=torch.float32).unsqueeze(-1)
        target_tensor = torch.tensor(target_price, dtype=torch.float32)
        
        return price_tensor, sentiment_tensor, target_tensor


# ─────────────────────────────────────────────────────────────────────────────
# 2. DUAL-INPUT ARCHITECTURES
# ─────────────────────────────────────────────────────────────────────────────

class DualInputLSTM_Concat(nn.Module):
    """
    Dual-input architecture: LSTM(prices) || Sentiment → Concat → FC → Output
    
    Simple merge via concatenation. Each input has its own LSTM encoder,
    outputs are concatenated and passed through a fusion layer.
    """
    
    def __init__(
        self,
        price_hidden_dim: int = 64,
        sentiment_hidden_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1,
        fusion_dim: int = 64,
    ):
        super().__init__()
        
        # Price encoder: LSTM
        self.price_lstm = nn.LSTM(
            input_size=1,
            hidden_size=price_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
        # Sentiment encoder: LSTM (lighter)
        self.sentiment_lstm = nn.LSTM(
            input_size=1,
            hidden_size=sentiment_hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        
        # Fusion layers
        concat_dim = price_hidden_dim + sentiment_hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(concat_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 1),
        )
    
    def forward(self, prices: torch.Tensor, sentiments: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prices: (batch, seq_len, 1)
            sentiments: (batch, seq_len, 1)
        
        Returns:
            predictions: (batch,)
        """
        # Price: extract last hidden state from LSTM
        _, (h_price, _) = self.price_lstm(prices)  # h_price: (num_layers, batch, hidden)
        h_price = h_price[-1]  # (batch, price_hidden_dim)
        
        # Sentiment: extract last hidden state from LSTM
        _, (h_sentiment, _) = self.sentiment_lstm(sentiments)  # (num_layers, batch, hidden)
        h_sentiment = h_sentiment[-1]  # (batch, sentiment_hidden_dim)
        
        # Concatenate and fuse
        fused = torch.cat([h_price, h_sentiment], dim=-1)  # (batch, concat_dim)
        output = self.fusion(fused).squeeze(-1)  # (batch,)
        
        return output


class DualInputLSTM_Attention(nn.Module):
    """
    Dual-input with attention mechanism: LSTM(prices) + Cross-Attention(sentiment)
    
    More sophisticated: sentiment acts as context for price via multi-head attention.
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        num_heads: int = 4,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Price encoder
        self.price_lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
        # Sentiment encoder
        self.sentiment_lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        
        # Cross-attention: price queries, sentiment as key/value
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        
        # Output projection
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
    
    def forward(self, prices: torch.Tensor, sentiments: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prices: (batch, seq_len, 1)
            sentiments: (batch, seq_len, 1)
        
        Returns:
            predictions: (batch,)
        """
        # Encode price sequence
        price_enc, _ = self.price_lstm(prices)  # (batch, seq_len, hidden_dim)
        
        # Encode sentiment sequence
        sentiment_enc, _ = self.sentiment_lstm(sentiments)  # (batch, seq_len, hidden_dim)
        
        # Cross-attention: use last price hidden state as query, sentiment as context
        attn_output, _ = self.cross_attn(
            price_enc[:, -1:, :],  # Query: (batch, 1, hidden_dim)
            sentiment_enc,         # Key/Value: (batch, seq_len, hidden_dim)
            sentiment_enc,
        )
        
        # Predict
        output = self.fc(attn_output[:, 0, :]).squeeze(-1)  # (batch,)
        
        return output


class DualInputLSTM_EarlyFusion(nn.Module):
    """
    Early fusion: Concatenate price + sentiment → Single LSTM
    
    Simplest approach: treat [price, sentiment] as 2-channel input.
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=2,  # [price, sentiment]
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, prices: torch.Tensor, sentiments: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prices: (batch, seq_len, 1)
            sentiments: (batch, seq_len, 1)
        
        Returns:
            predictions: (batch,)
        """
        # Concatenate along feature dimension
        combined = torch.cat([prices, sentiments], dim=-1)  # (batch, seq_len, 2)
        
        # LSTM
        out, (h, _) = self.lstm(combined)  # h: (num_layers, batch, hidden_dim)
        
        # Predict from last hidden state
        output = self.fc(h[-1]).squeeze(-1)  # (batch,)
        
        return output


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train_dual_input_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 15,
) -> Tuple[List[float], List[float]]:
    """
    Train a dual-input model.
    
    Args:
        model: DualInputLSTM_* instance
        train_loader: DataLoader yielding (prices, sentiments, targets)
        val_loader: DataLoader yielding (prices, sentiments, targets)
        device: torch.device
        epochs: Max epochs
        lr: Learning rate
        patience: Early stopping patience
    
    Returns:
        (train_losses, val_losses)
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_count = 0
    train_losses, val_losses = [], []
    
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        for prices, sentiments, targets in train_loader:
            prices = prices.to(device)
            sentiments = sentiments.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            preds = model(prices, sentiments)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for prices, sentiments, targets in val_loader:
                prices = prices.to(device)
                sentiments = sentiments.to(device)
                targets = targets.to(device)
                
                preds = model(prices, sentiments)
                loss = criterion(preds, targets)
                val_loss += loss.item()
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / max(len(val_loader), 1))
        
        # Early stopping
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Train: {train_losses[-1]:.4f} | Val: {val_losses[-1]:.4f}")
    
    return train_losses, val_losses


# ─────────────────────────────────────────────────────────────────────────────
# 4. EXAMPLE USAGE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Mock data (replace with your actual data)
    np.random.seed(42)
    n_days = 1000
    
    # Generate synthetic price data
    prices = np.cumsum(np.random.randn(n_days) * 0.01) + 100
    prices = prices.astype(np.float32)
    
    # Generate synthetic sentiment data (-1 to 1)
    sentiments = np.random.uniform(-1, 1, n_days).astype(np.float32)
    
    dates = pd.date_range("2021-01-01", periods=n_days)
    
    # Split
    train_size = int(0.8 * n_days)
    val_size = int(0.1 * n_days)
    
    train_prices, val_prices, test_prices = (
        prices[:train_size],
        prices[train_size : train_size + val_size],
        prices[train_size + val_size :],
    )
    train_sentiments, val_sentiments, test_sentiments = (
        sentiments[:train_size],
        sentiments[train_size : train_size + val_size],
        sentiments[train_size + val_size :],
    )
    
    # Fit scalers on training data only
    scaler_p = StandardScaler()
    scaler_s = StandardScaler()
    scaler_p.fit(train_prices.reshape(-1, 1))
    scaler_s.fit(train_sentiments.reshape(-1, 1))
    
    # Create datasets
    train_ds = SentimentPriceDataset(
        train_prices, train_sentiments, past_history=60,
        scaler_prices=scaler_p, scaler_sentiments=scaler_s
    )
    val_ds = SentimentPriceDataset(
        val_prices, val_sentiments, past_history=60,
        scaler_prices=scaler_p, scaler_sentiments=scaler_s
    )
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    # Choose architecture
    model = DualInputLSTM_Concat(
        price_hidden_dim=64,
        sentiment_hidden_dim=32,
        num_layers=1,
        dropout=0.1,
    )
    
    # Train
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_losses, val_losses = train_dual_input_model(
        model, train_loader, val_loader, device, epochs=50
    )
    
    print("Training complete!")
