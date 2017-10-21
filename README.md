# Stock-Market-Prediction-Challenge
Following repo is an attempt at Stock Market Prediction using Neural Networks and Sentiment Analysis

What it does:
This program aims to predict the Stock Prices using financial data from the past acquired online, and uses sentiments on twitter about that product as a parameter. This specific case looks into Facebook's stock and checks the users sentiments on Twitter to see if there is a positive sentiment or a negative sentiment. If there is a positive sentiment, then the prices are predicted to go up. 

This work was originally sourced from https://github.com/Avhirup/Stock-Market-Prediction-Challenge. Then modded to fit our needs. 

Update on the work:
My accuracy is quite low, but I was able to use a model which supported this theory with high accuracy. 

How to run:

1)
The program is run using Python. It also uses the following dependancies to run effectively. 

import sys
import tweepy
import csv
import pydot
import graphviz
import numpy as np
from textblob import TextBlob
from keras.models import Sequential
from keras.layers import Dense,LSTM
from keras.utils import plot_model
import matplotlib.pyplot as plt

2)
To run the file, Change directory on the terminal to you working directory where these files are saved. 

3)
type in python3 predicting_stock_prices.py fb.csv
 or python predicting_stock_prices.py fb.csv

 4)
 When the graph pops up, close it to continue the compilation of the code. 

 5)
 After the code compiles, you will see the acuracy all the way in the bottom. 


