#!/usr/bin/env python
# coding: utf-8

# ## 1) Familarizing with the data

# In[9]:


import os #File and Directory Operations:
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

data_directory = os.getcwd() # Directory where your data files are stored

data_frames = [] # Initialize an empty DataFrame to store the combined data
number = 10
count = 0

for filename in os.listdir(data_directory):
    count = count +1
    if filename.endswith(".txt"):  
        filepath = os.path.join(data_directory, filename) #The line of code filepath = os.path.join(data_directory, filename) is used to create a full file path by joining together two parts: data_directory and filename
        
        
        df = pd.read_csv(filepath, delimiter=',', header=None, # header = 0  indicates firls line of text file has columnn names but since we dont have it we will put None
                         names=['date_time', 'open', 'high', 'low', 'close', 'volume'])
        
        
        df['ticker'] = os.path.splitext(filename)[0]

        
        df['Timestamp'] = pd.to_datetime(df['date_time'])
        
        
        df.set_index(['Timestamp', 'ticker'], inplace=True)
        
        
        df.drop(['date_time'], axis=1, inplace=True)
       
        
        data_frames.append(df)
    if count == number:
        break

combined_data = pd.concat(data_frames)
combined_data.sort_index(inplace=True)

# You now have all the intraday stock data for S&P 500 stocks in the 'combined_data' DataFrame.
# You can perform various analyses or use this data as needed.


# In[10]:


print(combined_data.head(20))
print(combined_data['close'].dtype)


# In[11]:


combined_data.index.get_level_values('ticker').unique()


# In[12]:


# List of stock symbols  to plot
symbols_to_plot = ['AAPL_1min', 'ABMD_1min', 'ABT_1min', 'ABC_1min', 'AAP_1min',
       'ACN_1min']


# ### a) - plotting minute-by-minute

# In[13]:


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

for symbol in symbols_to_plot:
    symbol_data = combined_data.loc[combined_data.index.get_level_values('ticker') == symbol]
    plt.plot(symbol_data.index.get_level_values('Timestamp'), symbol_data['close'], label=symbol)

plt.title('Minute-by-Minute Closing Prices of Selected Symbols')
plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.legend()
plt.grid(True)

plt.show()


# In the code daily_data = combined_data['close'].resample('D').last(), 'D' is a string that represents the resampling frequency. In this case, 'D' stands for "daily."
# 
# When you resample time-series data, you can specify different frequencies to aggregate or transform the data based on your needs. Here are some common frequency strings used in Pandas:
# 
# 'D': Daily frequency, which aggregates the data to a daily interval. In this case, .last() is used to take the last value of each day's data, typically the closing price.
# 
# 'W': Weekly frequency, which aggregates the data to a weekly interval.
# 
# 'M': Monthly frequency, which aggregates the data to a monthly interval.
# 
# 'Q': Quarterly frequency, which aggregates the data to a quarterly interval.
# 
# 'A': Annual frequency, which aggregates the data to an annual interval.

# ##### About the function :
# Here's a breakdown of what .resample('D').last() does:
# 
# .resample('D'): This method resamples the time-series data, in this case, to a daily ('D') frequency. It groups the data into daily intervals.
# 
# .last(): Within each daily interval created by the resampling, this function selects the last data point (the last row) as the representative value for that day. It assumes that the data points within each daily interval are sorted in chronological order, with the last data point representing the closing value for that day.
# 
# For example, if you have intraday data with timestamps for each minute and you use .resample('D').last(), it will give you a DataFrame where each row represents a day, and the values for 'open', 'high', 'low', 'close', and 'volume' are taken from the last minute of that day.
# 
# This method is commonly used in financial time series analysis to create daily candlestick charts, where the last data point of the day represents the closing price, and the 'high' and 'low' values for the day are the highest and lowest prices reached during the day.
# 
# ##### other methods
#  Assuming 'df' is your DataFrame and 'date_time' is your timestamp column\
# daily_data = df.resample('D').agg({\
#     'open': 'first',\
#     'high': 'max',\
#     'low': 'min',\
#     'close': 'last',\
#     'volume': 'sum'\
# })
# 
# 

# ### b) - plotting Day-by-Day

# In[14]:


# Only valid with DatetimeIndex, TimedeltaIndex or PeriodIndex, but got an instance of 'MultiIndex'

# Reset the index to make it a single-level index
combined_data_reset = combined_data.reset_index()

# Resampling data to daily frequency and calculate the daily closing price
daily_data = combined_data_reset.groupby(['ticker', pd.Grouper(key='Timestamp', freq='D')])['close'].last() # pd.grouper is op 

plt.figure(figsize=(12, 6))

for symbol in symbols_to_plot:
    symbol_data = daily_data.loc[symbol]
    plt.plot(symbol_data.index, symbol_data, label=symbol)

plt.title('Day-by-Day Closing Prices of Selected Symbols')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.grid(True)

plt.show()



# ### c)- Candle stick plot
# To create a candlestick chart from your combined multi-index DataFrame (combined_data), you can follow these steps:
# 
# Prepare the Data for Plotting: You'll need to resample the data to the desired time frame (e.g., daily) since candlestick charts are typically used to represent daily price movements. You can also choose a specific stock ticker to plot.
# 
# Plot the Candlestick Chart: Use a plotting library like mplfinance or plotly to create the candlestick chart.

# In[15]:


def generate_daily_candlesticks(data, tickers, start_date,end_date):
    daily_candlesticks = {}  # Create a dictionary to store data for each ticker
    
    for ticker in tickers:
        # Filter data for the current ticker
        ticker_data = data.xs(key=ticker, level='ticker')
        ticker_data = ticker_data.loc[start_date:end_date]
        
        # Resample to daily frequency
        daily_data = ticker_data.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Store the daily data for the current ticker in the dictionary
        daily_candlesticks[ticker] = daily_data
    
    return daily_candlesticks


# In[16]:


# Specify the time period you want to plot
start_date = '2021-10-02'
end_date = '2022-01-06'
daily_candlestick = generate_daily_candlesticks(combined_data, symbols_to_plot,start_date,end_date)


# In[18]:


import mplfinance as mpf
style = mpf.make_mpf_style(base_mpf_style='binance', gridstyle='-')

# Plot the candlestick chart with volume on a secondary Y-axis
for ticker in daily_candlestick:
    mpf.plot(daily_candlestick[ticker], type='candle', style=style, title=f'Candlestick Chart for {ticker}',
         ylabel='Price', volume=True, ylabel_lower='Volume', figratio=(10, 6))


# ### d)- Looking at the data one can see that there are issues like :
#         * presence of data outside of trading hours
#         * missing days (most likely Saturdays & Sundays) when market is closed 

# # Q2: Normalization 
# Here, Normalization is important because , prices of various stocks changes overtime which can bring high volatality, and extreme values can distort analysis. Normalization can help stabilize the data and reduce the impact of outliers.
# 
# **Min-Max Scaling:**
# 
# Use Case: Min-Max scaling is useful when you want to maintain the relative differences between data points and constrain them to a specific range (e.g., [0, 1]).\
# When to Use: It's often used when you need to compare data points in terms of their proportions or when you're working with models that require input data to be within a certain range.\
# 
# **Z-Score Normalization (Standardization):**
# 
# Use Case: Z-score normalization is suitable when you want to compare data points in terms of their distance from the mean and are not concerned about preserving the original range.\
# When to Use: It's useful for identifying outliers and comparing data points in terms of standard deviations from the mean.\
# 

# ##### I have used Min-Max scaling as it scales the data to a specific range, typically between 0 and 1. This preserves the interpretability of the scaled data. Since the data doesnot conatain any outliers we can use MinMax scaler. Standard Scaler scales the data between [-1,1]. Standadscaler is more like find the deviation from the mean meanwhile MinMax scaling is more of scaling the data.

# In[19]:


print(combined_data)


# we have to normalize each stock individually 

# In[20]:


from sklearn.preprocessing import MinMaxScaler

# Specify the stock ticker you want to scale
target_ticker = 'AAPL_1min'
columns_to_normalize = ['open', 'high', 'low', 'close', 'volume']

filter_data = combined_data.loc[pd.IndexSlice[:, target_ticker], columns_to_normalize]
print(filter_data)
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(filter_data)

normalized_combined_data = combined_data.copy()
normalized_combined_data.loc[pd.IndexSlice[:, target_ticker], columns_to_normalize] = normalized_data



# In[21]:


print(normalized_combined_data.loc[pd.IndexSlice[:, target_ticker], columns_to_normalize])


# In[ ]:


print(normalized_combined_data.shape)


# In[22]:


from sklearn.preprocessing import StandardScaler

# Specify the stock ticker you want to scale
target_ticker = 'AAPL_1min'
columns_to_normalize = ['open', 'high', 'low', 'close', 'volume']

filtered_data = combined_data.loc[pd.IndexSlice[:, target_ticker], columns_to_normalize]
scaler = StandardScaler()
normalized_data = scaler.fit_transform(filtered_data)

normalized_combined_data1 = combined_data.copy()
normalized_combined_data1.loc[pd.IndexSlice[:, target_ticker], columns_to_normalize] = normalized_data



# In[ ]:


print(normalized_combined_data1)


# # Data preprocessing
#  * Remove all data which are out of trading hours(discard all data before 9:30 and after 15:59)
#  
#  

# In[23]:


start_time = pd.Timestamp("09:30:00")
end_time = pd.Timestamp("16:00:00")  # 4:00 PM

#a custom filtering function
def filter_trading_hours(group):
    trading_hours_mask = ((group.index.get_level_values('Timestamp').time >= start_time.time()) & 
                         (group.index.get_level_values('Timestamp').time <= end_time.time()))
    return group[trading_hours_mask]


filtered_data = normalized_combined_data.groupby('ticker',as_index=False).apply(filter_trading_hours) # this is a pandas series

# The 'filtered_data' DataFrame will now contain data only within the specified trading hours on each trading day for each ticker.


# In[ ]:


print(type(filtered_data))


# In[24]:


print(filtered_data)


# In[25]:


aapl_data = filtered_data.xs(target_ticker, level='ticker')
print(aapl_data)


# ## 3.
# 1. I would opt for high-frequency trading, which means making a lot of quick trades in a very short time, here minutes. For frequent, high-volume traders, a slightly wider spread with lower commission fees may be more cost-effective.</br>
# 
# 2. To keep my high-frequency trading costs low, I'm concentrating on two things: buy-ask spreads and trading commissions.
# I'm picking assets that are easy to buy and sell and setting lower commission rates.</br>
# 
# 3. In high-frequency trading, I'd choose to trade one stock. Focusing on a single stock allows us to specialize and become an expert in the behavior of that particular asset. Then the knowledge of this stock can be applied to other stocks in same industry.

# ## 4) LSTM to predict minute by minute closing price 

# In[26]:


#hyperparameters
input_size = 5  # Number of input features
hidden_size = 64  # Number of LSTM units
num_layers = 2  # Number of LSTM layers
output_size = 1  # Number of output units (predicting next minute's closing price)
sequence_length = 20  # Length of input sequences


# In[27]:


import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=hidden_size, num_layers=num_layers, output_size=1, dropout=0.0):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.linear_1 = nn.Linear(input_size, hidden_layer_size)
        # self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_layer_size, hidden_size=self.hidden_layer_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(num_layers*hidden_layer_size, output_size)
        

    def forward(self, x):
        batchsize = x.shape[0]

        # layer 1
        x = self.linear_1(x)
       
        # LSTM layer
        lstm_out, (h_n, c_n) = self.lstm(x)

        # reshape output from hidden cell into [batch, features] for `linear_2`
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1) 
        
        # layer 2
        x = self.dropout(x)
        predictions = self.linear_2(x)
        return predictions[:,-1]


# ## 5) -Flexible Datloader
# 

# In[28]:


import pandas as pd
# Specify the desired ticker and date
selected_ticker = target_ticker # to change this make change in minmax scaler part 
start_day = pd.to_datetime('2021-01-04')  # problem : i have to look if the date is present in the data
end_day = pd.to_datetime('2021-06-23')   
ticker_data  = filtered_data[(filtered_data.index.get_level_values('Timestamp').date >= start_day.date()) & 
                             (filtered_data.index.get_level_values('Timestamp').date <= end_day.date()) & 
                             (filtered_data.index.get_level_values('ticker') == selected_ticker)]


# In[29]:


print(ticker_data)


# ### making a dataframe with target close price 

# In[30]:


import pandas as pd
ticker_data_copy = ticker_data.copy()
ticker_data_copy['target_close'] = ticker_data_copy['close'].shift(-1)# the shifting is done to make the next minute closing price as the taget price for the previous minute
ticker_data_copy = ticker_data_copy.dropna()

# Define input features and target variable
input_features = ['open', 'high', 'low', 'close', 'volume']
target_variable = 'target_close'

# Split the data into input (X) and target (y) arrays
X = ticker_data_copy[input_features].values
y = ticker_data_copy[target_variable].values


# In[31]:


print(ticker_data_copy)


# In[ ]:


print(type(X))
print(X.shape)


# #### Defined a class to make tensors of required sequence length 

# In[32]:


from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

# Define a custom dataset
class StockDataset(Dataset):
    def __init__(self, X, y, sequence_length):
        self.X = X
        self.y = y
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.X) - self.sequence_length

    def __getitem__(self, indx):
        # Get a sequence of data with the specified length
        X_seq = self.X[indx:indx + self.sequence_length]
        y_target = self.y[indx + self.sequence_length - 1]  # The target is the last value in the sequence
        return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y_target, dtype=torch.float32)


# ## 6)- Training the Model

# In[33]:


# Create datasets and dataloaders for training and testing
train_ratio = 0.74885  # Adjust this ratio based on your train/test split
train_size = int(train_ratio * len(X))
print(train_size+1)

train_dataset = StockDataset(X[:train_size+1], y[:train_size+1], sequence_length)
test_dataset = StockDataset(X[train_size+1:], y[train_size+1:], sequence_length)

batch_size = 32# Adjust as needed
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size,drop_last=True)


device = "cuda" if torch.cuda.is_available() else "cpu"


# In[34]:


# Iterate through the DataLoader and print the contents of each batch
for batch_idx, (x, y) in enumerate(train_dataloader):
    print(f"Batch {batch_idx + 1} - x:")
    print(x)
    print(f"Batch {batch_idx + 1} - y:")
    print(y)


# In[36]:


model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout=0.0)
model = model.to(device)


# In[37]:


criterion = nn.MSELoss()
lr = 0.0002
optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.98), eps=1e-9)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# ## Training
# 

# In[38]:


num_epochs = 100
train_loss = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss= 0.0
    
    for idx, (x, y) in enumerate(train_dataloader):
        optimizer.zero_grad()
        batchsize = x.shape[0]
        #print(x.shape[0])
        #print(y)
        x = x.to(device)
        y = y.to(device)
        out = model(x)
        loss = criterion(out.contiguous(), y.contiguous())
        loss.backward()
        optimizer.step()
        
        epoch_loss += (loss.detach().item() / batchsize)
    # lr = scheduler.get_last_lr()[0] 
    average_epoch_loss = np.mean(epoch_loss)
    train_loss.append(average_epoch_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}] | Loss: {loss.item()} | lr:{lr}')
    # scheduler.step()
     


# In[39]:


# Plot the loss 
plt.plot(range(1, num_epochs+1), train_loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.grid(True)
plt.show()


# In[41]:


actual_values = []  # List to store actual values
predicted_values = []  # List to store predicted values

model.eval()
epoch_loss2= 0.0

with torch.no_grad():
    for x, y in train_dataloader:
        batchsize = x.shape[0]

        x = x.to('cpu')
        y = y.to('cpu')
        out = model(x)
        loss = criterion(out.contiguous(), y.contiguous())
        epoch_loss2 += (loss.detach().item() / batchsize)

        actual_values.extend(y.detach().numpy())
        predicted_values.extend(out.detach().numpy())

    


# In[ ]:


print(predicted_values)


# In[42]:


plt.figure(figsize=(10, 6))
plt.plot(actual_values, label='Actual', linestyle='-')
plt.plot(predicted_values, label='Predicted', linestyle='-')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.title('Actual vs. Predicted Values')
plt.legend()
plt.show()


# ## Testing

# In[54]:


test_loss=[] #to store test loss
test_actual_values = []  # List to store actual values
test_predicted_values = []  # List to store predicted values

model.eval()
epoch_loss2= 0.0

with torch.no_grad():
    for x, y in test_dataloader:
        batchsize = x.shape[0]

        x = x.to('cpu')
        y = y.to('cpu')
        out = model(x)
        loss = criterion(out.contiguous(), y.contiguous())
        epoch_loss2 += (loss.detach().item() / batchsize)

        test_actual_values.extend(y.detach().numpy())
        test_predicted_values.extend(out.detach().numpy())
        
for i in range(0,len(test_predicted_values)):
    test_loss.append(abs(test_actual_values[i]-test_predicted_values[i]))
    


# In[63]:


print(len(test_actual_values))


# In[55]:


plt.figure(figsize=(10, 6))
plt.plot(test_actual_values, label='Actual', linestyle='-')
plt.plot(test_predicted_values, label='Predicted', linestyle='-')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.title('Actual vs. Predicted Values')
plt.legend()
plt.show()


# ## 7) -

# In[65]:





# ### Trading Module

# In[83]:


print(filter_data)


# In[87]:


filter_data = combined_data.loc[pd.IndexSlice[:, target_ticker], columns_to_normalize]
scaler = MinMaxScaler()
normalized_data_1 = scaler.fit_transform(filter_data['close'].to_numpy().reshape(-1,1))


# In[93]:


unscaled_predicted_test_data=scaler.inverse_transform(np.array(test_predicted_values).reshape(-1, 1)) #applying inverse to unnormalize the predicted data
print(unscaled_predicted_test_data)
unscaled_actual_test_data=scaler.inverse_transform(np.array(test_actual_values).reshape(-1, 1)) #applying inverse to unnormalize the predicted data
print(unscaled_actual_test_data)


# In[95]:


print(unscaled_predicted_test_data.shape)
print(unscaled_actual_test_data.shape)


# In[120]:


class TradingPlatform:
    def __init__(self, initial_balance=100000, commission_rate=0.002, bid_ask_spread=0.01):
        self.balance = initial_balance
        self.portfolio = {}
        self.commission_rate = commission_rate
        self.trade_history = []

    def place_order(self, asset, action, quantity, price):
        if action == 'buy':
            total_cost = price * quantity
            commission = total_cost * self.commission_rate
            if total_cost + commission <= self.balance:
                # Incorporate bid-ask spread for buying
#                 total_cost += total_cost * self.bid_ask_spread
                self.balance -= (total_cost + commission)
                self.portfolio[asset] = self.portfolio.get(asset, 0) + quantity
                self.trade_history.append({'asset': asset, 'action': action, 'quantity': quantity, 'price': price, 'commission': commission})
                return price
        elif action == 'sell':
            if asset in self.portfolio and self.portfolio[asset] >= quantity:
                total_proceeds = price * quantity
                commission = total_proceeds * self.commission_rate
                # Incorporate bid-ask spread for selling
#                 total_proceeds -= total_proceeds * self.bid_ask_spread
                self.balance += (total_proceeds - commission)
                self.portfolio[asset] -= quantity
                self.trade_history.append({'asset': asset, 'action': action, 'quantity': quantity, 'price': price, 'commission': commission})
                return price
        return None

    def get_balance(self):
        return self.balance

    def get_portfolio(self):
        return self.portfolio

    def get_trade_history(self):
        return self.trade_history

Trading_module = TradingPlatform()

for i in range(len(test_actual_values)-1):
    if(unscaled_predicted_test_data[i+1]>unscaled_actual_test_data[i]):
        Trading_module.place_order('AAPL', 'buy', 10, unscaled_actual_test_data[i])
    else: Trading_module.place_order('AAPL', 'sell', 10, unscaled_actual_test_data[i])
    
print(Trading_module.get_balance())


# ## 8)-
# 
# a) - The price prediction error increase as you go further from the last time it was trained 

# In[56]:


plt.figure(figsize=(14, 6))
plt.plot(range(1, len(test_loss) + 1),test_loss, marker='o', linestyle='-', color='b', label='Training MSE')
plt.xlabel('Time')
plt.ylabel('MSE')
plt.title('Test Error')
plt.legend()
plt.grid(True)
plt.show()


# c) - as we can see from test graph , it is merely a chance that a simple buy and hold strategy works better . It also depends on the duration in which we are looking . For the duration I have chosen the buy and hold works better as i would have bought low and sell high. but there are other parts of the graphs where we might have bought high and sell low.

# ## References
# 
# https://stackoverflow.com/questions/75648914/trying-to-understand-lstm-parameter-hidden-size-in-pytorch - for hidden size
# 
# 
# there is good article on how to select hidden layers in LSTM model :
# https://ai.stackexchange.com/questions/3156/how-to-select-number-of-hidden-layers-and-number-of-memory-cells-in-an-lstm
# 
# Specifically for LSTMs, see this Reddit discussion Does the number of layers in an LSTM network affect its ability to remember long patterns? https://www.reddit.com/r/MachineLearning/comments/4behuh/does_the_number_of_layers_in_an_lstm_network/?rdt=50446
# 
# The main point is that there is usually no rule for the number of hidden nodes you should use, it is something you have to figure out for each case by trial and error.
# 
# If you are also interested in feedforward networks, see the question How to choose the number of hidden layers and nodes in a feedforward neural network? at Stats SE. Specifically, this answer was helpful https://stats.stackexchange.com/q/181/82135

# 
