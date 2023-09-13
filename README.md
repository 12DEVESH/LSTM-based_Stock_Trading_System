# LSTM-based_Stock_Trading_System
This repository contains my code and instructions for building and testing a stock trading system using an LSTM (Long Short-Term Memory) model. My goal is to predict stock prices and make trading decisions based on those predictions. This README provides a comprehensive guide to understanding and using my project.

## Data Analysis
I began by familiarizing myself with the input data [sp500_tickers_A-D_1min_1pppix.zip](https://iitbacin-my.sharepoint.com/:u:/g/personal/asethi_iitb_ac_in/ER8Evo2yfdhPmuaw1rjIOywBvN2WyjIYbYopcQeMTL8z7g?e=DvXNbs).

I conducted data analysis as follows:

 *  I plotted the minute-by-minute closing price series of a few stocks.
 *  I plotted the day-by-day closing price series of a few stocks.
 *  I created a complete candlestick chart with volume on the secondary y-axis for a few stocks, choosing an appropriate time period.
 *  I documented my observations, including data issues, unexpected jumps, or missing data.
## Data Normalization
I explored at least two methods for normalizing the data. After experimenting, I selected one normalization method and provided a justification for my choice.
## Scenario Decisions
I made several scenario-related decisions:
 *  I determined my preferred trading strategy from high-frequency trading, intra-day swing trading, inter-day trading, or long-term trading.
 *  I assumed a buy-ask spread (inversely related to volume and directly related to price) and trading commissions based on quick market research. These assumptions are documented.
 *  I decided to do High Frequency Trading - traded a single stock.
## LSTM Model
I wrote a PyTorch module for defining an LSTM model. This module offers flexibility in adjusting input dimensions, the number of units, and the number of layers.
Data Loader
I created a flexible data loader tailored for training the LSTM model, especially when dealing with high-frequency data. The data loader includes open, close, high, low, and volume data for one or more stocks to aid in predicting the selected stock's price.
## Model Training
I trained or pre-trained the model to predict future prices (or changes in price if normalized). I ensured that the future prediction horizon remains adjustable, such as between one minute or ten minutes into the future. I reserved the last two years of data for testing purposes.
## Trading Module
I developed a trading module capable of making logical decisions to buy, hold, or sell stocks, with the flexibility to perform these actions in any order, considering shorting possibilities.
## Testing the Trading System
I tested the trading system on data from the latest years, which were not used during model training.

I addressed the following questions:
 *  Does the price prediction error increase as I test the system further from the last training date?
 *  Is it possible to trade profitably, accounting for bid-ask spreads and commissions?
 *  How does the system's profitability compare to a simple buy-and-hold strategy over the long term (e.g., one or two years)?
# Conclusion
In conclusion, this repository serves as my comprehensive framework for building and testing a stock trading system using an LSTM model. It encompasses data analysis, normalization, scenario decision-making, model implementation, data loading, training, trading logic, and testing. I encourage myself and others to experiment with various configurations to optimize trading strategies and evaluate profitability.
