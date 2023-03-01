from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataset=pd.read_csv('C:/Users\hp\Downloads\Iris_new.csv')
dataset.head()
dataset.shape
x=dataset.iloc[:,0:4].values
y=dataset.iloc[:,-1].values

print(x)
print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,
                    test_size=0.20,random_state=0)

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

classifier=GaussianNB()
#classifier= MultinomialNB
#classifier=BernoulliNB

classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)
print(y_pred)


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
acc=(sum(np.diag(cm))/len(y_test))
print(acc)

from sklearn import metrics
acc1=metrics.accuracy_score(y_test,y_pred)
print(acc1)

from sklearn.preprocessing import LabelEncoder
labelen=LabelEncoder()
yy=labelen.fit_transfrom(y)









