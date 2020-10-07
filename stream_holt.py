# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 09:33:43 2020

@author: Nick
"""


import warnings
import pandas as pd
from statsmodels.tsa.holtwinters import Holt
from sklearn.metrics import r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot


# choose a city to model
city = "Chicago" # Chicago, Mumbai, Beijing, Auckland, San Diego

# define training parameters
start_size = 200 # number of data points to start modeling with
horizon = 7 # number of data points to predict ahead

# In[1]: Prepare the Data

# read in the data
data = pd.read_csv("Weather.csv")

# define output
outputs = ["avg_temp"]

# pick a city and split the data into training
data_copy = data.copy()
data = data.loc[data["city"] == city].reset_index(drop=True)
train = data[:start_size].copy()

# In[2]: Stream the Data

# define a function for streaming data
def incoming_data(df, idx):
    return df.iloc[[idx], :].copy()

predictions = pd.DataFrame()
actuals = pd.DataFrame()
time = []
for i in range(train.shape[0], data.shape[0] - horizon):
    # track time
    time.append(data.index[i])

    # get output (Y)
    Y = train[[outputs[0]]].copy() # only single stream output allowed

    # train and forecast
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = Holt(Y)
        model_fit = model.fit()
    forecast = model_fit.forecast(horizon)
    predictions = pd.concat([predictions, 
                             pd.DataFrame(forecast).reset_index(drop=True).T], axis=0) 

    # get the horizon
    Y_new = data.iloc[i:(i + horizon)][outputs[0]].tolist()
    actuals = pd.concat([actuals, pd.DataFrame(Y_new).T], axis=0) 

    # stream data
    train = pd.concat([train, incoming_data(data, i)], axis=0)

    # report on one step ahead prediction
    pred = forecast.tolist()[0]
    true = Y_new[0]
    print('1-step ahead: predicted=%f, expected=%f' % (pred, true))

# In[2]: Evaluate the Model

# compute R2
R2 = r2_score(actuals, predictions, multioutput="raw_values")
R2 = pd.DataFrame({"R2": R2})
R2["Step Ahead"] = R2.index + 1
print(R2)

# pick a step ahead to evaluate
step_ahead = 1
df = pd.concat([actuals.iloc[:,step_ahead - 1],
                predictions.iloc[:,step_ahead - 1]], axis=1)
df.columns = ["Actual", "Predict"]
df["index"] = time

# plot the prediction series
fig = px.scatter(df, x="index", y="Predict")
fig.add_trace(go.Scatter(x=df["index"], y=df["Actual"], mode="lines", showlegend=False, name="Actual"))
fig.update_layout(font=dict(size=16))
plot(fig, filename="Series Predictions.html")

# draw a parity plot
fig1 = px.scatter(df, x="Actual", y="Predict")
fig1.add_trace(go.Scatter(x=df["Actual"], y=df["Actual"], mode="lines", showlegend=False, name="Actual"))
fig1.update_layout(font=dict(size=16))
plot(fig1, filename="Parity Plot.html")
