import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

label_encoder = preprocessing.LabelEncoder()
sns.set_style("darkgrid")
col_names = ['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
data = pd.read_csv("titanic.csv",skiprows=1, header= 0,names=col_names)
data.dropna(subset=["Embarked","Age"], inplace=True)
data.drop("Name",axis=1,inplace=True)
data.drop("SibSp",axis=1,inplace=True)
data.drop("Parch",axis=1,inplace=True)
data.drop("Ticket",axis=1,inplace=True)
data.drop("Cabin",axis=1,inplace=True)
data.drop("PassengerId",axis=1,inplace=True)

data["Sex"]=label_encoder.fit_transform(data["Sex"])
data["Embarked"]=label_encoder.fit_transform(data["Embarked"])
data.head(10)



def calculate_prior(df, Y):
    classes = sorted(list(df[Y].unique()))
    prior = []
    for i in classes:
        prior.append(len(df[df[Y]==i])/len(df))
         
    return prior

def calculate_likelihood_gaussian(df, feat_name, feat_val, Y, label):
    df = df[df[Y]==label]
    mean, std = df[feat_name].mean(), df[feat_name].std()
    p_x_given_y = (1 / (np.sqrt(2 * np.pi) * std)) *  np.exp(-((feat_val-mean)**2 / (2 * std**2 )))
    
    return p_x_given_y

def naive_bayes_gaussian(df, X, Y):
    # get feature names
    features = list(df.columns)[1:]

    # calculate prior
    prior = calculate_prior(df, Y)
    
    Y_pred = []
    # loop over every data sample
    for x in X:
        # calculate likelihood
        labels = sorted(list(df[Y].unique()))
        likelihood = [1]*len(labels)
        for j in range(len(labels)):
            for i in range(len(features)):
                likelihood[j] *= calculate_likelihood_gaussian(df, features[i], x[i], Y, labels[j])

        # calculate posterior probability (numerator only)
        post_prob = [1]*len(labels)
        for j in range(len(labels)):
            post_prob[j] = likelihood[j] * prior[j]

        Y_pred.append(np.argmax(post_prob))

    return np.array(Y_pred) 
train, test = train_test_split(data, test_size=.3, random_state=41)

X_test = test.iloc[:,1:].values
Y_test = test.iloc[:,0].values


Y_pred = naive_bayes_gaussian(train, X=X_test, Y="Survived")


print(confusion_matrix(Y_test, Y_pred))
print(f1_score(Y_test, Y_pred, average='macro'))

sns.heatmap(confusion_matrix(Y_test, Y_pred),center=0,cmap="Blues");
