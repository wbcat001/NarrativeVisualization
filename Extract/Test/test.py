import plotly.express as px
import pandas as pd


df = pd.read_csv("data/harrypotter/harry1_df.csv")

px.line(df["SentimentScore"].rolling(window=30, min_periods=1).mean()).show()