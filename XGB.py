# -- coding:utf-8 -*-
# Author : Wrs
# Data : 2022/11/9 下午12:00

from data_prepro import Voice
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import cross_val_score,learning_curve,GridSearchCV
import numpy as np
import warnings
warnings.filterwarnings("ignore")

df=pd.read_csv('./voice.csv')
print(df)
plt.figure(figsize = (15,15))
corr = df.corr()
seaborn.heatmap(corr,annot=True)
plt.show()


def prepro(df, test_size):
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    y = LabelEncoder().fit_transform(y)
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
    return df, X, y, X_train, X_test, y_train, y_test

res = prepro(df, test_size=0.2)

for i in range(res[3].shape[-1]):
    plt.figure()
    seaborn.kdeplot(res[3][:,i],shade = True)
    plt.twinx()
    seaborn.kdeplot(res[4][:,i],shade = True,color = "y")
    plt.show()

import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

xgb.set_config(verbosity=0)
res=Voice.prepro(test_size=0.2)

class XGBModel():

    def __init__(self, configs, *args):
        self.configs = configs
        self.df, self.X, self.Y, self.X_train, self.X_test, self.Y_train, self.Y_test = args

    def build_model(self):
        configs = self.configs
        self.model = XGBClassifier(learning_rate=configs["learning_rate"],
                                   max_depth=configs["max_depth"],
                                   gamma=configs["gamma"],
                                   objective=configs["objective"],
                                   num_class=configs["num_class"],
                                   random_state=configs["random_state"],
                                   eval_metric=configs["eval_metric"], )
        return

    def hyp_tuning(self, hyp_ranges):
        for hyp in hyp_ranges:
            validation_mean = []
            validation_std = []
            for i in hyp_ranges[hyp]:
                configs = self.configs
                configs[hyp] = i
                val = xgb.cv(configs, xgb.DMatrix(self.X, self.Y), nfold=10, seed=27, verbose_eval=False,
                             metrics=['auc'])
                validation_mean.append(val.iloc[-1, -2])
                validation_std.append(val.iloc[-1, -1,])
            plt.errorbar(hyp_ranges[hyp], validation_mean, yerr=validation_std, fmt='o-', ecolor='r', color='g',
                         capsize=4)

            plt.xlabel('Value of {}'.format(hyp))
            plt.ylabel('Cross-Validated auc')
            plt.show()
        return

    def LC_auc(self):
        configs = self.configs
        hyp_res = xgb.cv(configs, xgb.DMatrix(self.X, self.Y), nfold=10, seed=27, verbose_eval=False, metrics=['auc'])

        plt.plot(hyp_res.index * 10, hyp_res["train-auc-mean"], color='blue', marker='o', label="train_auc")
        plt.fill_between(hyp_res.index * 10, hyp_res["train-auc-mean"] + hyp_res["train-auc-std"],
                         hyp_res["train-auc-mean"] - hyp_res["train-auc-std"], alpha=0.1, color='red')
        plt.plot(hyp_res.index * 10, hyp_res["test-auc-mean"], color='grey', linestyle='--', marker='s',
                 label="test_auc")
        plt.fill_between(hyp_res.index * 10, hyp_res["test-auc-mean"] + hyp_res["test-auc-std"],
                         hyp_res["test-auc-mean"] - hyp_res["test-auc-std"], alpha=0.1, color='blue')
        plt.xlabel('percent of training samples %')
        plt.ylabel("auc")
        plt.legend()
        plt.show()
        return

    def train(self):
        print("\n" + "-" * 30, )
        print('Model Training Started')

        self.model.fit(self.X_train, self.Y_train,
                       eval_set=[(self.X_test, self.Y_test)], verbose=False)
        self.model.get_booster().feature_names = self.df.drop(columns='label').columns.values.tolist()

        print('Model Training Completed.')
        print("-" * 30, "\n")
        return

    def eval(self):
        predictions = self.model.predict(self.X_test)
        cm = confusion_matrix(self.Y_test, predictions)
        seaborn.heatmap(cm, annot=True)
        print('confusion_matrix:')
        plt.show()
        print('classification_report:\n')
        print(classification_report(self.Y_test, predictions))

    def imp_plot(self, importance_type):
        print('feature_importances: ({})'.format(importance_type))
        xgb.plot_importance(self.model, importance_type=importance_type)
        plt.show()
        return

    def grid_search(self, hyp_ranges):
        GS = GridSearchCV(self.model, param_grid=hyp_ranges, cv=10, scoring='roc_auc', )
        GS.fit(self.X_train, self.Y_train)
        return GS


XGB_configs = {"booster":"gbtree",
              "learning_rate":0.1,
              "max_depth":6,
              "gamma":0.,
               "min_child_weight": 3,
               'reg_alpha':0.,
               'reg_lambda':0.,
               "tree_method":"exact",
              "objective":'multi:softmax',
              "num_class": 2,
              "random_state":27 ,
              "eval_metric":"mlogloss"}

hyp_ranges = {"booster":["gbtree","gblinear","dart"]}
XGB = XGBModel(XGB_configs,*res)
XGB.hyp_tuning(hyp_ranges)
print("-"*30,"\n")

print("gbtree粗调整")
XGB_configs["booster"]="gbtree"
hyp_ranges = {"learning_rate":np.arange(0.01, 0.5, 0.01,),
             "max_depth":range( 3,10),
             "gamma":[0.1,1,10,100],
             'min_child_weight':range(1,6),
             'reg_alpha':[1e-5,+1e-2,+0.1,+1,+100],
             'colsample_bytree':[i/100.0 for i in range(75,90,5)],
             'subsample':[i/100.0 for i in range(75,90,5)],}
XGB = XGBModel(XGB_configs,*res)
XGB.hyp_tuning(hyp_ranges)
print("-"*30,"\n")

print("dart粗调整")
XGB_configs["booster"]="dart"
hyp_ranges = {"learning_rate":np.arange(0.01, 0.5, 0.01,),
             "max_depth":range( 3,10),
             "gamma":[0.1,1,10,100],
             'min_child_weight':range(1,6),
             'reg_alpha':[1e-5,+1e-2,+0.1,+1,+100],
             'colsample_bytree':[i/100.0 for i in range(75,90,5)],
             'subsample':[i/100.0 for i in range(75,90,5)],}
XGB = XGBModel(XGB_configs,*res)
XGB.hyp_tuning(hyp_ranges)
print("-"*30,"\n")
XGB_configs = {"booster":"gbtree",
              "learning_rate":0.1,
              "max_depth":6,
              "gamma":0.,
               "min_child_weight": 3,
               'reg_alpha':0.,
               'reg_lambda':0.,
               "tree_method":"exact",
              "objective":'multi:softmax',
              "num_class": 2,
              "random_state":27 ,
              "eval_metric":"mlogloss"}
print("gbtree细调整")
XGB_configs["booster"]="gbtree"
hyp_ranges = {"learning_rate":np.arange(0.3, 0.35, 0.001,),
    "gamma":[0.0001,0.001,0.01,0.02,0.03,0.04,0.05,0.1]}
XGB = XGBModel(XGB_configs,*res)
XGB.hyp_tuning(hyp_ranges)


XGB_configs = {"booster":"gbtree",
              "learning_rate":0.1,
              "max_depth":6,
              "gamma":0.05,
               "min_child_weight": 3,
               'reg_alpha':0.,
               'reg_lambda':0.,
               "tree_method":"exact",
              "objective":'multi:softmax',
              "num_class": 2,
              "random_state":27 ,
              "eval_metric":"mlogloss"}

hyp_ranges = {"max_depth":range( 3,10),
"learning_rate":np.arange(0.3, 0.33, 0.01,),
'subsample':[i/100.0 for i in range(90-5,90+5,1,)]}
XGB = XGBModel(XGB_configs,*res)
XGB.build_model()
GS = XGB.grid_search(hyp_ranges)
print(GS.best_params_)



XGB_configs = {"booster":"gbtree",
              'learning_rate': 0.31,
              'max_depth': 4,
              "gamma":0.05,
               "min_child_weight": 3,
               'reg_alpha':0.,
               'reg_lambda':0.,
               "tree_method":"exact",
              "objective":'multi:softmax',
              "num_class": 2,
              "random_state":27 ,
              "eval_metric":"mlogloss",'subsample': 0.89}

XGB = XGBModel(XGB_configs,*res)
XGB.build_model()
XGB.train()
XGB.LC_auc()
XGB.eval()
XGB.imp_plot(importance_type = "weight")
XGB.imp_plot(importance_type = "gain")
XGB.imp_plot(importance_type = "cover") # “weight”, “”, or “cover”

# Ockham's Razor 去除线性相关的特征，
df=df.drop(columns = ["dfrange","sfm","IQR","kurt","sd","meanfreq","centroid","meandom"])
res=prepro(df,test_size=0.2)
seaborn.heatmap(df.corr())
plt.show()

XGB_configs = {"booster":"gbtree",
              'learning_rate': 0.31,
              'max_depth': 4,
              "gamma":0.05,
               "min_child_weight": 3,
               'reg_alpha':0.,
               'reg_lambda':0.,
               "tree_method":"exact",
              "objective":'multi:softmax',
              "num_class": 2,
              "random_state":27 ,
              "eval_metric":"mlogloss",'subsample': 0.89}

XGB = XGBModel(XGB_configs,*res)
XGB.build_model()
XGB.train()
XGB.LC_auc()
XGB.eval()
XGB.imp_plot(importance_type = "weight")
XGB.imp_plot(importance_type = "gain")
XGB.imp_plot(importance_type = "cover")








