# -- coding:utf-8 -*-
# Author : Wrs
# Data : 2022/10/23 下午2:36


from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import cross_val_score,learning_curve,GridSearchCV
import numpy as np


#svm\xgboost\randomforest\

class VoiceClassifier:

    def __init__(self,*args):
        self.df,self.X,self.y,self.X_train,self.X_test,self.y_train,self.y_test=args

    def __call__(self,*args,**kgs):
        return self.__init__(*args,**kgs)

    def svm_classifier(self,cv,scoring):
        kernels,res=["linear", "poly", "rbf", "sigmoid"],{}
        while kernels:
            kernel=kernels.pop()
            svc=SVC(kernel=kernel)
            svc.fit(self.X_train,self.y_train)
            y_pred = svc.predict(self.X_test)
            print('Accuracy score(kernel={}):{}'.format(kernel,metrics.accuracy_score(self.y_test,y_pred)))
            scores = cross_val_score(svc,self.X,self.y,cv=cv,scoring=scoring)
            print('{}-fold cv accuracy scores:{}\nMean score:{}\n'.format(cv,scores,scores.mean()))
            res[kernel]=scores
        return res


    def learning_curve(self,**kwgs):
        res={}
        if kwgs.__contains__('estimator_add_info'):
            for k,v in kwgs['estimator_add_info'].items():
                while v:
                    estimator=f"{kwgs['estimator']}({k}='{v.pop()}')"
                    train_sizes,train_scores,test_scores=learning_curve(estimator=eval(estimator),X=self.X_train,y=self.y_train,
                                                                train_sizes=kwgs['optinal_train_range'],cv=kwgs['cv'])
                    res[estimator]=[train_sizes,train_scores,test_scores,np.mean(train_scores, axis=1),np.std(train_scores, axis=1),\
                                                np.mean(test_scores, axis=1),np.std(test_scores, axis=1)]
        else:
            train_sizes, train_scores, test_scores = learning_curve(estimator=eval(f"{kwgs['estimator']}()"), X=self.X_train,y=self.y_train,
                                                                    train_sizes=kwgs['optinal_train_range'],cv=kwgs['cv'])
            res[kwgs['estimator']] = [train_sizes, train_scores, test_scores, np.mean(train_scores, axis=1),
                              np.std(train_scores, axis=1), \
                              np.mean(test_scores, axis=1), np.std(test_scores, axis=1)]
        return res


    def svm_param_choose(self,param_range,keneral,cv):
        acc_score=[]
        if all(list(map(lambda x:isinstance(x,int),param_range))) and keneral=='poly' :
            for d in param_range:
                svc = SVC(kernel=keneral, degree=d)
                scores = cross_val_score(svc, self.X,self.y,cv=cv, scoring='accuracy')
                acc_score.append(scores.mean())
        else:
            for v in param_range:
                if keneral=='linear':
                    svc=SVC(kernel=keneral, C=v)
                elif keneral=='rbf':
                    svc=SVC(kernel=keneral, gamma=v)
                else:
                    raise  Exception('Unexpected keneral')
                scores = cross_val_score(svc, self.X,self.y,cv=cv, scoring='accuracy')
                acc_score.append(scores.mean())
        return acc_score

    def grid_search(self,**kwgs):
        model_svm = GridSearchCV(SVC(),kwgs,cv=10, scoring='accuracy')
        model_svm.fit(self.X_train, self.y_train)
        y_pred = model_svm.predict(self.X_test)
        print('Best params:{}'.format(model_svm.best_params_))
        print('Training best_score:{}'.format(model_svm.best_score_))
        print('Accuracy of model:{}'.format(metrics.accuracy_score(y_pred, self.y_test)))





