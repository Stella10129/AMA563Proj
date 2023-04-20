# -- coding:utf-8 -*-
# Author : Wrs
# Data : 2022/10/23 下午2:20


import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split



class Voice:
    dt='./voice.csv'

    @classmethod
    def info(cls,dt=dt):
        df=pd.read_csv(dt)
        print('Data columns:{}'.format(df.columns))
        print('Data shape:{}'.format(df.shape))
        print('Data info:\n{}'.format(df.info()))
        print('Dara null count:\n{}'.format(df.isnull().sum()))
        print('***Number of male: {}***'.format(df[df.label=='male'].shape[0]))
        print('***Number of female: {}***'.format(df[df.label=='female'].shape[0]))

    @classmethod
    def prepro(self,test_size,dt=dt):
        df=pd.read_csv(dt)
        X,y=df.iloc[:,:-1],df.iloc[:,-1]
        y=LabelEncoder().fit_transform(y)
        scaler=StandardScaler().fit(X)
        X=scaler.transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
        return df,X,y,X_train,X_test,y_train,y_test