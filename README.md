# Attention-Driven-RPA-for-Secure-Stock-Trading
This repository contains the code and implementation details for our research project on AI-driven algorithmic trading. The system integrates advanced Deep Reinforcement Learning (specifically the TD3 algorithm) with a custom Self-Attention mechanism and real-time financial news sentiment analysis (FinBERT) to generate actionable stock market predictions.

## 📊 Project Overview
Traditional algorithmic trading models often fail to capture the nuances of market sentiment and complex state dependencies. This project introduces a hybrid architecture that:
1. **Analyzes Market Sentiment:** Uses NLP to quantify daily news sentiment for major Indian stocks (RELIANCE, TCS, INFY, ICICIBANK, ITC).
2. **Learns Optimal Policies:** Employs a Twin Delayed DDPG (TD3) agent equipped with a self-attention layer to weigh the importance of different market features.
3. **Deploys Real-Time Predictions:** Provides an inference API that translates raw market states into discrete trading signals (Strong Buy, Buy, Hold, Sell, Strong Sell).

## 📂 Repository Structure

* **`sentiment_engine.ipynb`**: The NLP pipeline. It loads `final_news_sentiment_analysis.csv`, processes the text using a HuggingFace FinBERT sequence classification model, applies exponential moving average (EMA) smoothing, and outputs the daily sentiment to `sentiment_data.csv`.
* **`networks.ipynb`**: Contains the PyTorch neural network definitions. It defines the `SelfAttention` module, alongside the `Actor` and `Critic` classes required for the TD3 architecture.
* **`train.ipynb`**: The main reinforcement learning training loop. It instantiates the `SecureTradingEnv`, applies exploration noise, trains the TD3 agent to maximize episode profits, plots the training curve, and saves the final model weights (`td3_latest_trader_actor_final.pth`).
* **`predict_api.py`**: The production inference script. It reconstructs the `Actor` network, loads the pre-trained weights, pads incoming live market states to the required 67 dimensions, and generates threshold-based trading decisions.
* **`Output.jpeg`**: A screenshot demonstrating the final application output and trading signal generation interface.
