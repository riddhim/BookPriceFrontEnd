import pandas as pd
import numpy as np 
from sklearn import preprocessing
import pickle

df=pd.read_csv('C:\\Users\\User\Desktop\BookPricePrediction\data (2).csv')

df.head()
cat_cols=df.select_dtypes('object')


print(cat_cols)
encode=preprocessing.LabelEncoder()
for col in cat_cols:
    df[col]=encode.fit_transform((df[col]))
    
df.shape

from sklearn.linear_model import LinearRegression
df=df.drop('Unnamed: 0',axis=1)
X=df[['Name','Author','Genre']]
y=df['Price']

lr=LinearRegression()
lr.fit(X,y)
pickle.dump(lr,open('model.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))
