# -- coding:utf-8 -*-
# Author : Wrs
# Data : 2022/10/23 下午2:36

from data_prepro import Voice
from data_mining import VoiceClassifier
import matplotlib.pyplot as plt
import seaborn
import numpy as np
import warnings
warnings.filterwarnings("ignore")

##Info and preprocessing
Voice.info(dt='./voice.csv')
print('*'*60)
res=Voice.prepro(test_size=0.2)
df=res[0]
df.plot(kind='scatter',x='meandom',y='dfrange')
df.plot(kind='kde',y='meandom')
plt.show()
seaborn.pairplot(df[['meandom','mode','sfm','skew','centroid','label']],hue='label',size=2)
plt.show()

######################################SVM classifier######################################
# The first classifier is svc, a kind of svm used for classification.
# First,We find the predictor by comparing different kernels and cvs.
ClassifierInfo=VoiceClassifier(*res)
kinds_scores=ClassifierInfo.svm_classifier(cv=10,scoring='accuracy')

#Compare the cv scores of different kernels
for k,s in kinds_scores.items():
    plt.title('Cross validation scores(kernel={})'.format(k))
    plt.scatter(np.arange(len(s)),s)
    plt.axhline(y=np.mean(s), color='b')
    plt.show()

# The default value of 0.2 is used to divide training set and test set, but what kind of training scale can fit kernel and cv better?
estimator_add_info={'kernel':["linear", "poly", "rbf", "sigmoid"]}
res=ClassifierInfo.learning_curve(estimator='SVC',estimator_add_info=estimator_add_info,
                                                                      optinal_train_range=np.linspace(0.1,1.0,10),cv=10)
for k,v in res.items():
    train_sizes,train_scores,test_scores,train_mean,train_std,test_mean,test_std=v
    plt.plot(train_sizes, train_mean, color='blue', marker='o', label='Training Accuracy')
    plt.fill_between(train_sizes,train_mean+train_std,train_mean-train_std,alpha=0.1, color='red')
    plt.plot(train_sizes, test_mean, color='grey', linestyle='--', marker='s',label='Test Accuracy')
    plt.fill_between(train_sizes,test_mean+test_std,test_mean-test_std,alpha=0.1, color='blue')
    plt.title(k)
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# To further optimize the classifier, we choose to add key parameters to different kernels.(C，degree and gamma).

# The C parameter tells the SVM optimization how much you want to avoid misclassifying each training example.
# For large values of C, the optimization will choose a smaller-margin hyperplane
# if that hyperplane does a better job of getting all the training points classified correctly.
# Conversely, a very small value of C will cause the optimizer to look for a larger-margin separating hyperplane,
# even if that hyperplane misclassifies more points.
# Thus for a very large values we can cause overfitting of the model and for a very small value of C we can cause underfitting.
# In conclusion,the value of C must be chosen in such a manner that it generalised the unseen data well.
C_ranges=[list(range(1, 26)),list(np.arange(0.1,6,0.1))]
while C_ranges:
    C_range=C_ranges.pop()
    scores=ClassifierInfo.svm_param_choose(C_range,'linear',cv=10)
    print("Accuracy of linear kernel with param c and 10-fold cv:\n{}".format(scores))
    plt.plot(C_range,scores)
    plt.xlabel('Value of C for SVC')
    plt.ylabel('Cross-Validated Accuracy')
    plt.show()
# From the above plot we can see that accuracy has been close to 97% for C=1 and C=6 and then it drops around 96.8% and remains constant.

# Technically, the gamma parameter is the inverse of the standard deviation of the RBF kernel (Gaussian function),
# which is used as similarity measure between two points. Intuitively, a small gamma value define a Gaussian function
# with a large variance. In this case, two points can be considered similar even if are far from each other.
# In the other hand, a large gamma value means define a Gaussian function with a small variance and in this case,
# two points are considered similar just if they are close to each other.
Gamma_ranges=[[0.0001,0.001,0.01,0.1,1,10,100],[0.0001,0.001,0.01,0.1],[0.01,0.02,0.03,0.04,0.05]]
while Gamma_ranges:
    Gamma_range=Gamma_ranges.pop()
    scores=ClassifierInfo.svm_param_choose(Gamma_range,'rbf',cv=10)
    print("Accuracy of rbf kernel with param gamma and 10-fold cv:\n{}".format(scores))
    plt.plot(Gamma_range,scores)
    plt.xlabel('Value of gamma for SVC')
    plt.ylabel('Cross-Validated Accuracy')
    plt.show()
##Amongst three figure, gamma should be around 0.03

#Taking polynomial kernel with different degree
degrees=[2,3,4,5,6]
scores=ClassifierInfo.svm_param_choose(degrees,'poly',cv=10)
print("Accuracy of poly kernel with param degree and 10-fold cv:\n{}".format(scores))
plt.plot(degrees,scores,color='r')
plt.xlabel('Value of degree for SVC')
plt.ylabel('Cross-Validated Accuracy')
plt.show()
# Score is high for third degree polynomial and then there is drop in the accuracy score as degree of polynomial increases.
# Thus increase in polynomial degree results in high complexity of the model and thus causes overfitting.

# The parameters of the last step are optimized through grid search.
# The kernel function selects 'poly' and 'rbf',the selection range of' C 'is 1 to 6,the gamma value is 0.02-0.032,
# the degree is 2-4 and the training set proportion is 0.8
kernel=('poly','rbf')
C=list(range(1,7))
gamma=[round(x,3)for x in list(np.arange(0.02,0.033,0.001))]#float精度溢出
degree=list(range(2,5))
ClassifierInfo.grid_search(kernel=kernel,C=C,gamma=gamma,degree=degree)




















