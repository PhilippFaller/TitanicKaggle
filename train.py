import pandas as pd
from selection import preprocess
from selection import build_model
from keras.callbacks import EarlyStopping

if __name__ == "__main__":
    dataframe = pd.read_csv("data/train.csv")
    target = dataframe["Survived"]
    data = dataframe.drop("Survived", axis=1)
    data = preprocess(data)
    model = build_model(data)
    model.fit(x=data, y=target, epochs=1000, callbacks=[EarlyStopping(monitor="loss", patience=10)])
    model.save("titanic.h5")

