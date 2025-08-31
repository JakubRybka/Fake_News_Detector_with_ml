import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import numpy as np

# READ FEATURES
df = pd.read_csv("features2.csv")
df['Label'] = df['Label'].map(lambda x: 1-x)
df.drop(columns=['Index'],inplace=True)

# TRAIN
xtrain,xtest,ytrain,ytest = train_test_split( df.drop("Label", axis=1),df["Label"], test_size=0.2, random_state=42)
model = RandomForestClassifier()#LogisticRegression(max_iter=50000)#
model.fit(xtrain,ytrain)
# TEST
prediction1 = model.predict(xtest)
accuracy1 = np.where(prediction1 == ytest, 1, 0).sum()/len(prediction1)
print(f"Test F1 score: {f1_score(prediction1,ytest)}")
print(f"Test Accuracy: {accuracy1}")
