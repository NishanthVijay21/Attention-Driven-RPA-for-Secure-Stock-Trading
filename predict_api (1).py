def predict():
    try:
        import torch
        import torch.nn as nn
        import numpy as np
        import yfinance as yf
        import pandas as pd
        import traceback
        import warnings

        warnings.filterwarnings("ignore")

        # -------- MODEL --------
        class SelfAttention(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.q = nn.Linear(dim, dim)
                self.k = nn.Linear(dim, dim)
                self.v = nn.Linear(dim, dim)
                self.scale = dim ** 0.5

            def forward(self, x):
                Q = self.q(x)
                K = self.k(x)
                V = self.v(x)
                attn_scores = torch.softmax((Q * K) / self.scale, dim=-1)
                return attn_scores * V

        class Actor(nn.Module):
            def __init__(self, state_dim, action_dim):
                super().__init__()
                self.fc1 = nn.Linear(state_dim, 256)
                self.attn = SelfAttention(256)
                self.fc2 = nn.Linear(256, 128)
                self.out = nn.Linear(128, action_dim)

            def forward(self, state):
                x = torch.relu(self.fc1(state))
                x = self.attn(x)
                x = torch.relu(self.fc2(x))
                return torch.tanh(self.out(x))

        # -------- LOAD MODEL --------
        try:
            model = Actor(67, 5)
            model.load_state_dict(torch.load(r"C:\DRL\with sentiment.pth", map_location='cpu'))
            model.eval()
        except Exception as e:
            return "MODEL LOAD ERROR:\n" + str(e) + "\n" + traceback.format_exc()

        # -------- ASSETS --------
        assets = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "ICICIBANK.NS", "ITC.NS"]

        state = []
        valid_assets = []

        # -------- LOOP --------
        for asset in assets:
            try:
                df = None

                try:
                    df = yf.download(
                        asset,
                        period="3mo",
                        auto_adjust=True,
                        progress=False,
                        threads=False
                    )
                except Exception:
                    continue

                if df is None or df.empty:
                    continue

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # -------- FEATURES --------
                df["Returns"] = df["Close"].pct_change()

                delta = df["Close"].diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                rs = gain.rolling(14).mean() / (loss.rolling(14).mean() + 1e-9)
                df["RSI"] = 100 - (100 / (1 + rs))

                ema12 = df["Close"].ewm(span=12).mean()
                ema26 = df["Close"].ewm(span=26).mean()
                df["MACD"] = ema12 - ema26
                df["MACD_signal"] = df["MACD"].ewm(span=9).mean()

                df["SMA20"] = df["Close"].rolling(20).mean()
                df["EMA20"] = df["Close"].ewm(span=20).mean()
                df["EMA50"] = df["Close"].ewm(span=50).mean()

                sma20 = df["Close"].rolling(20).mean()
                std20 = df["Close"].rolling(20).std()
                df["BB_upper"] = sma20 + 2 * std20
                df["BB_lower"] = sma20 - 2 * std20

                tr = np.maximum(
                    df["High"] - df["Low"],
                    np.maximum(
                        abs(df["High"] - df["Close"].shift()),
                        abs(df["Low"] - df["Close"].shift())
                    )
                )
                df["ATR"] = tr.rolling(14).mean()

                df["Momentum"] = df["Close"] - df["Close"].shift(10)
                df["OBV"] = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()

                df["Sentiment"] = 0.0

                df.dropna(inplace=True)

                if df.empty:
                    continue

                row = df.iloc[-1]

                features = [
                    float(row["Returns"]),
                    float(row["RSI"]) / 100.0,
                    float(row["MACD"]),
                    float(row["MACD_signal"]),
                    float(row["SMA20"]) / float(row["Close"]),
                    float(row["EMA20"]) / float(row["Close"]),
                    float(row["EMA50"]) / float(row["Close"]),
                    float(row["BB_upper"]) / float(row["Close"]),
                    float(row["BB_lower"]) / float(row["Close"]),
                    float(row["ATR"]) / float(row["Close"]),
                    float(row["Momentum"]) / float(row["Close"]),
                    float(row["OBV"]) / 1e7,
                    float(row["Sentiment"])
                ]

                state.extend(features)
                valid_assets.append(asset)

            except Exception:
                continue

        if len(state) == 0:
            return "DATA ERROR: No valid data fetched"

        state.append(100000)
        state.append(0)

        while len(state) < 67:
            state.append(0.0)

        if len(state) > 67:
            state = state[:67]

        try:
            state = np.array(state, dtype=np.float32)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            with torch.no_grad():
                action = model(state_tensor)

            action_np = action.numpy()[0]

        except Exception as e:
            return "MODEL INFERENCE ERROR:\n" + str(e) + "\n" + traceback.format_exc()

        results = []

        for i in range(len(valid_assets)):
            val = float(action_np[i])

            if val > 0.6:
                decision = "STRONG BUY"
            elif val > 0.2:
                decision = "BUY"
            elif val < -0.6:
                decision = "STRONG SELL"
            elif val < -0.2:
                decision = "SELL"
            else:
                decision = "HOLD"

            results.append(f"{valid_assets[i]} → {decision} (confidence: {round(abs(val),2)})")

        if len(results) == 0:
            return "No valid predictions generated"

        return "\n".join(results)

    except Exception as e:
        import traceback
        return "CRITICAL ERROR:\n" + str(e) + "\n" + traceback.format_exc()


# -------- WRITE OUTPUT FOR UIPATH --------
if __name__ == "__main__":
    output = predict()
    with open(r"C:\DRL\output.txt", "w", encoding="utf-8") as f:
        f.write(output)