import pandas as pd
from train import preprocess
from keras.models import load_model
from math import isnan

dataframe = pd.read_csv("data/test.csv")
p_ids = dataframe["PassengerId"]
dataframe["Fare"] = dataframe["Fare"].apply(lambda x: dataframe["Fare"].mean() if isnan(x) else x)
data = preprocess(dataframe)
model = load_model('titanic.h5')
prediction = model.predict(data)

with open('result.csv', 'w') as file:
    file.write("PassengerId,Survived\n")
    for p_id, pred in zip(p_ids, prediction):
        rounded_pred = int(round(pred[0]))
        file.write(str(p_id) + "," + str(rounded_pred) + "\n")
