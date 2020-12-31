import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle
df=pd.read_csv("banknote_authentication.csv")
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
print(X.shape)
print(y.shape)
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
clf=RandomForestClassifier()
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
score=accuracy_score(y_test,y_pred)
print(score)
pickle_out=open("classifier.pkl","wb")
pickle.dump(clf,pickle_out)
pickle_out.close()
print(clf.predict([[2,3,4,1]]))