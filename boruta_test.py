import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

if __name__=="__main__":

    df = pd.read_csv("/home/fidel/mltest/auto-miles-per-gallon.csv")
    print df.describe()
    X = df.iloc[ : , 1:-1].values
    print X
    y = df.iloc[:, 0].values
    print y
    y = y.ravel()


    # defining random forest vclassifier

    rf = RandomForestClassifier(n_jobs=-1, class_weight='auto', max_depth=4)

    # feature selection method for boruta

    feature_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)

    # finding all relevant feature
    feature_selector.fit(X, y)

    # check the selected feature
    feature_selector.support_

    # check the ranking feature

    feature_selector.ranking_
