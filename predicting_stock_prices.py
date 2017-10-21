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
np.random.seed(7)
import matplotlib.pyplot as plt

datafile = sys.argv[1]


#Step 1 - Insert your API keys
consumer_key= 'plPNvWhlzzlOqgrT50P1i9wGj'
consumer_secret= 'NDDtlTwwFYIwbtX5Vru8SBDF0A6fnvmEnOsJzn68P565rvLy4j'
access_token='792294428891152385-pjVppYt7GMvUyFWYugjzfoDZZW74PUH'
access_token_secret='JuJYVEnH6XP6oWcinuyL8WCUtXrWxFQoXKO2zR57VeM75'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

#Step 2 - Search for your company name on Twitter
public_tweets = api.search('Facebook')
print (public_tweets[2].text)

#Step 3 - Define a threshold for each sentiment to classify each 
#as positive or negative. If the majority of tweets you've collected are positive
#then use your neural network to predict a future price
threshold=0
pos_sent_tweet=0
neg_sent_tweet=0
for tweet in public_tweets:    
    analysis = TextBlob(tweet.text)
    if analysis.sentiment.polarity>=threshold:
        pos_sent_tweet=pos_sent_tweet+1
    else:
        neg_sent_tweet=neg_sent_tweet+1
if pos_sent_tweet>neg_sent_tweet:
    print("Overall Positive")
else:
    print("Overall Negative")
    

#data collection
dates = []
prices = []
def get_data(filename):
	with open(filename, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader)
		for row in csvFileReader:
			dates.append(int(row[0].split('-')[0]))
			prices.append(float(row[1]))
	return

#Step 5 reference your CSV file here
get_data(datafile)
plt.plot(prices)

plt.show()

#Step 6 In this function, build your neural network model using Keras, train it, then have it predict the price 
#on a given day. We'll later print the price out to terminal.
def create_datasets(dates,prices):
    train_size=int(0.80*len(dates))
    TrainX,TrainY=[],[]
    TestX,TestY=[],[]
    cntr=0
    for date in dates:
        if cntr<train_size:
            TrainX.append(date)
        else:
            TestX.append(date)    
    for price in prices:
        if cntr<train_size:
            TrainY.append(price)
        else:
            TestY.append(price)
            
    return TrainX,TrainY,TestX,TestY

def predict_prices(dates, prices, x):
    TrainX,TrainY,TestX,TestY=create_datasets(dates,prices)

    TrainX=np.reshape(TrainX,(len(TrainX),1))
    TrainY=np.reshape(TrainY,(len(TrainY),1))
    TestX=np.reshape(TestX,(len(TestX),1))
    TestY=np.reshape(TestY,(len(TestY),1))
    
   # for i in range(251):
    #    print (TrainX[i],TrainY[i],'\n')
    
    
    model=Sequential()
    model.add(Dense(32,input_dim=1,init='uniform',activation='relu'))
    model.add(Dense(32,input_dim=1,init='uniform',activation='relu'))
    model.add(Dense(16,init='uniform',activation='relu'))
    
    model.add(Dense(1,init='uniform',activation='relu'))
    model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
    trainData = model.fit(TrainX,TrainY,nb_epoch=100,batch_size=3,verbose=1)
    score = model.evaluate(TrainX,TrainY,batch_size=3, verbose=1, sample_weight=None)
    print('\n','Test score:', score[0])
    print('Test accuracy:', score[1]) 

    
    #score,acc = model.evaluate(TrainX,TrainY,batch_size=3,show_accuracy=true)
    #return score,acc
predict_prices(dates,prices,2)
#print(predicted_price)