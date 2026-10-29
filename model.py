import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

student = pd.read_csv('student-data (1).csv')
student


x = student.drop('passed',axis=1)
y = student['passed']



cate_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus','Mjob','Fjob','reason','guardian','schoolsup', 'famsup', 'paid', 'activities', 'nursery',
       'higher', 'internet', 'romantic']
num_cols=['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel',
       'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']



x_train,x_test,y_train,y_test =train_test_split(
    x,y,test_size=0.2,random_state=42
)

x_train = pd.get_dummies(
    x_train, columns=cate_cols, dtype=int
)


x_test = pd.get_dummies(
    x_test, columns=cate_cols, dtype=int
)

model = LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)


y_pred = model.predict(x_test)
print('Accuracy = ' ,accuracy_score(y_test,y_pred))



#Save the model
pickle.dump(model,open('model.pkl','wb'))

print('Model training completed')