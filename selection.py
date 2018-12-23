import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.utils import to_categorical
from keras.regularizers import l1_l2
from math import isnan
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import re

def preprocess(data):
    #Title info from names
    data["Mr"] = data["Name"].apply(lambda name: 1 if "Mr" in name and not "Mrs" in name else 0)
    data["Miss"] = data["Name"].apply(lambda name: 1 if "Miss" in name else 0)
    data["Mrs"] = data["Name"].apply(lambda name: 1 if "Mrs" in name else 0)
    #data["Dr"] = data["Name"].apply(lambda name: 1 if "Dr" in name else 0)
    #data["Master"] = data["Name"].apply(lambda name: 1 if "Master" in name else 0)
    #data["Rev"] = data["Name"].apply(lambda name: 1 if "Rev" in name else 0)
    #data["Soldier"] = data["Name"].apply(lambda name: 1 if ("Major" in name or "Capt" in name or "Col" in name) else 0)
    data["Title"] = data["Name"].apply(lambda name: 1 if any([t in name for t in ["Major","Capt","Col","Rev","Master","Dr"]]) else 0)

    #Fix format
    data["Sex"] = data["Sex"].apply(lambda sex: 0 if sex=="male" else 1)

    #useless noise
    data["Cabin"] = data["Cabin"].apply(lambda c: c if isinstance(c, str) else "0")
    data["Cabin Letter"] = data["Cabin"].apply(lambda c: max([i if letter in c else 3 for i, letter in enumerate("ABCDDEFG")]) -3 )
    data["Cabin Letter Valid"] = data["Cabin Letter"].apply(lambda l: 0 if l==0 else 1)
    #data["Cabin Num"] = data["Cabin"].apply(lambda c: int(next(iter(re.findall("\d+", c)), "0")))
    #data["Cabin Num Valid"] = data["Cabin Num"].apply(lambda n: 0 if n==0 else 1)
    data["Multiple Cabins"] = data["Cabin"].apply(lambda c: 1 if len(c.split()) > 1 else 0)

    #Drop useless
    data = data.drop("Ticket", axis=1)
    data = data.drop("Cabin", axis=1)
    data = data.drop("PassengerId", axis=1)
    data = data.drop("Name", axis=1)

    #Categorize
    data["1st"] = data["Pclass"].apply(lambda t: 1 if t == 1 else 0)
    data["2nd"] = data["Pclass"].apply(lambda t: 1 if t == 2 else 0)
    data["3rd"] = data["Pclass"].apply(lambda t: 1 if t == 3 else 0)
    data = data.drop("Pclass", axis=1)
    #Just noise
    #data["Cherbourg"] = data["Embarked"].apply(lambda s: 1 if s == "C" else 0)
    #data["Queenstown"] = data["Embarked"].apply(lambda s: 1 if s == "Q" else 0)
    #data["Southampton"] = data["Embarked"].apply(lambda s: 1 if s == "S" else 0)
    data = data.drop("Embarked", axis=1)
    #Sanitize
    data["Age_valid"] = data["Age"].apply(lambda a: 0 if isnan(a) else 1)
    data["Age"] = data["Age"].apply(lambda a: -1 if isnan(a) else a)    
    data["Fare"] = data["Fare"].apply(lambda x: data["Fare"].mean() if isnan(x) else x)

    #Just noise
    #data["Traveled Free"] = data["Fare"].apply(lambda x: 1 if x==0 else 0)

    #Truncate Outliers
    data["SibSp"] = data["SibSp"].apply(lambda s: min(4, s))
    data["Parch"] = data["Parch"].apply(lambda s: min(3, s))
    data["Fare"] = data["Fare"].apply(lambda s: np.log(s+1))

    #Scale
    s = MinMaxScaler()
    keys = data.keys()
    data[data.keys()] = s.fit_transform(data[data.keys()])

    #Cross features

    """
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=42)
    for x in keys:
        for y in keys:
            cross_feature = data[x]*data[y]
            c, _ = pearsonr(cross_feature, target) 
            if c >= 0.4:
                data[x + " x " + y] = cross_feature
                print(x + " x " + y)
    """

    #Cross features just noise?
    """
    data["Sex x Age"] = data["Sex"]*data["Age"]
    data["Sex x Age_valid"] = data["Sex"]*data["Age_valid"]
    data["Sex x Fare"] = data["Sex"]*data["Fare"]
    data["Sex x 1st"] = data["Sex"]*data["1st"]
    data["Sex x 2nd"] = data["Sex"]*data["2nd"]
    """
    """
    for k in data.keys():
        data[k].hist()
        plt.title(k)
        plt.show()
    """
    return data

def build_model(data):
    samples, num_features = data.shape

    reg_const = 0.0005
    inp = Input(shape=(num_features,))
    hidden = Dense(num_features, activation="relu", kernel_regularizer=l1_l2(l1=reg_const, l2=reg_const))(inp)
    out = Dense(1, activation="sigmoid")(hidden)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def train_evaluate(n, i, model, X_train, y_train, X_test, y_test): 
    history = model.fit(x=X_train, y=y_train, epochs=200
        , validation_data=(X_test, y_test) , verbose=0
        )    
    #Plot
    # summarize history for accuracy
    plt.subplot(n, 3, 3*i-2)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # summarize history for loss
    plt.subplot(n, 3, 3*i-1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #detect outliers
    #plt.subplot(n, 3, 3*i)
    #plt.scatter(model.predict(X_train), y_train)
    return history


if __name__ == "__main__":
    dataframe = pd.read_csv("data/train.csv")
    target = dataframe["Survived"]
    data = dataframe.drop("Survived", axis=1)
    data = preprocess(data)
    
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)

    cvacc = []
    cvvalacc = []
    for i, (train, test) in enumerate(skf.split(data, target)):
        print("Running Fold " + str(i+1) + "/" + str(n_folds))
        model = None # Clearing the NN.
        model = build_model(data)
        h = train_evaluate(n_folds, i+1, model, data.reindex(train), target.reindex(train), data.reindex(test), target.reindex(test))
        cvacc.append(h.history["acc"][-1])
        cvvalacc.append(h.history["val_acc"][-1])

    print("Mean acc: " + str(np.mean(cvacc)) +" -/+ " + str(np.std(cvacc)))
    print("Mean val_acc: " + str(np.mean(cvvalacc)) +" -/+ " + str(np.std(cvvalacc)))
    plt.show()

