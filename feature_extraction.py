from cmath import nan
from email.utils import parsedate_to_datetime
import pandas as pd
import numpy as np
from textblob import TextBlob



def getsentiment(text,polar):
    
    if isinstance(text, str,):
        
        blob = TextBlob(text)
        n = len(blob.sentences)
        score = blob.sentiment.polarity
        score2 = blob.sentiment.subjectivity
        if(polar==1):
            return score
        elif(polar==0):
            return score2
    else:
        return 0



df = pd.read_csv("A2_Data.csv",index_col=[0],parse_dates = ['date'])
df['Month'] = df['date'].dt.month
df['Year'] = df['date'].dt.year
df['Hour'] = df['date'].dt.hour
df.dropna()
df['Helpfulness'] = df['user_favourites'] / df['user_followers'] 



df['TextScore'] = df['text'].apply(lambda x:getsentiment(x,polar=1))
df['Subjectivity'] = df['text'].apply(lambda x:getsentiment(x,polar=0))


df.to_csv("./data/X_train.csv")