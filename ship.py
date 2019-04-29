#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 13:03:45 2019

@author: fidel
"""

import numpy as np
import pandas as pd

def Pred():

    df = pd.read_csv('/home/fidel/Downloads/shiptrain.csv')
    df_test = pd.read_csv('/home/fidel/Downloads/shiptest.csv')

    df.isna().sum()

    drop_list = ["Report.Year",'Index',"vstr_subtype_name",'WEATHER_SEA_STATE']
    df.drop(drop_list, inplace = True, axis=1)
    df_test.drop(drop_list, inplace=True, axis=1)


    df_corr=df.corr()
    print(df_corr)


    X=df.iloc[:,[1,3,4,5,6,7,8,9,10,11,12]].values
    y = df.iloc[:,2:3].values

    X_test = df_test.iloc[:, [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]].values
    y_test = df_test.iloc[:, 2:3].values

    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder_X = LabelEncoder()


    X[:,7] = labelencoder_X.fit_transform(X[:,7])
    X[:,8] = labelencoder_X.fit_transform(X[:,8])
    X[:,9] = labelencoder_X.fit_transform(X[:,9])
    X[:,10] = labelencoder_X.fit_transform(X[:,10])
    #X[:,11] = labelencoder_X.fit_transform(X[:,11])



    X_test[:,7] = labelencoder_X.fit_transform(X_test[:,7])
    X_test[:,8] = labelencoder_X.fit_transform(X_test[:,8])
    X_test[:,9] = labelencoder_X.fit_transform(X_test[:,9])
    X_test[:,10] = labelencoder_X.fit_transform(X_test[:,10])


    onehotencoder = OneHotEncoder(categorical_features=[7,8,9,10])

    X = onehotencoder.fit_transform(X).toarray()
    X_test = onehotencoder.fit_transform(X_test).toarray()


    # feature scaling

    from  sklearn.preprocessing import StandardScaler

    # sc_X = StandardScaler()
    # sc_y= StandardScaler()
    # X_feature = sc_X.fit(X).transform(X)
    # y_label = sc_y.fit(y).transform(y)


    # applying linear model

    from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
    regressor = LinearRegression()
    regressor_ridge = RidgeCV(alphas=[0.1, 0.5,0.05,1], cv=10)
    regressor_lasso = LassoCV(alphas=[0.1, 0.5,0.05,1], cv=3)

    # linear
    lr = regressor.fit(X,y)
    r2 = regressor.score(X,y)
    r2_test = regressor.score(X_test, y_test)
    print('linear regression output:-', r2_test)

    print('linear regression output:-',r2)

    pred = lr.predict(X_test)


    # ridge
    regressor_ridge.fit(X, y)
    ridge_r2 = regressor_ridge.score(X, y)
    ridge_r2_test = regressor_ridge.score(X_test, y_test)

    print("ridge regression output:-" ,ridge_r2)

    # lasso
    regressor_lasso.fit(X, y)
    lasso_r2 = regressor_lasso.score(X, y)
    lasso_r2_test = regressor_lasso.score(X_test, y_test)
    print("lasso regression output:-" ,lasso_r2)
    ###################################################
    # decision tree regression

    from sklearn.tree import DecisionTreeRegressor
    regressor_decision = DecisionTreeRegressor(random_state=0)
    regressor_decision.fit(X, y)

    # Predicting a new result
    y_pred_dec = regressor_decision.predict(X_test)
    #
    # from sklearn.metrics import classification_report, confusion_matrix
    # print(confusion_matrix(y_test, y_pred_dec))
    # print(classification_report(y_test, y_pred_dec))


    # pred vs actual data

    # df_decison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_dec})
    # print(df_decison)

    from sklearn import metrics
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_dec))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_dec))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_dec)))



    # Fitting Random Forest Regression to the dataset
    from sklearn.ensemble import RandomForestRegressor
    regressor_random = RandomForestRegressor(n_estimators=100, random_state=0)
    random_lr = regressor_random.fit(X, y)


    # Predicting a new result
    y_pred_random = random_lr.predict(X_test)

    # absolute error
    errors = abs(y_pred_random - y_test)
    print("errors: ",errors)

    # mean absolute error

    print("mean absolute error :  ", round(np.mean(errors), 2), 'degrees.')


    # calculating the accuracy

    mape = 100* (errors/y_test)

    # calculate the display accureacy
    accuracy = 100 - np.mean(mape)
    print("accuracy :", round(accuracy, 2), "%")

    # Get numerical feature importances
    importances = list(regressor_random.feature_importances_)
    print(importances)
    # List of tuples with variable and importance
    # feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    #
    # # Sort the feature importances by most important first
    # feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    #
    # # Print out the feature and importances
    # [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

    # print(y_pred_random)
    # print(y_test)
    random_score = random_lr.score(X, y)
    random_score_test = random_lr.score(X_test, y_test)
    print("Random forest regressor output:-", random_score)
    print("Random forest regressor output:-", random_score_test)

    # on training data the accuacy of the model is
    print("\n\nAccuracy on training data_________\n")
    print('linear regression output:-', r2)
    print("ridge regression output:-", ridge_r2)
    print("lasso regression output:-", lasso_r2)
    print("Random forest regressor output:-", random_score,"\n")

    # on test data the accuracy of model is
    print("Accuracy on test data_________\n")
    print('linear regression output:-', r2_test)
    print("ridge regression output:-", ridge_r2_test)
    print("lasso regression output:-", lasso_r2_test)
    print("Random forest regressor output:-", random_score_test,"\n")

    # applying svr model

    # from sklearn.svm import SVR
    # svr_regressor = SVR(kernel='rbf')
    # svr_regressor.fit(X, y)
    #
    #
    # svr_r2 = svr_regressor.score(X_feature,y_label)
    # print(svr_r2)


    # with statsmodels
    import statsmodels.api as sm
    # X_stats = sm.add_constant(X) # adding a constant

    model = sm.OLS(y, X).fit()
    # predictions = model.predict(X)

    print_model = model.summary()
    print(print_model)





if __name__=="__main__":
    Pred()
