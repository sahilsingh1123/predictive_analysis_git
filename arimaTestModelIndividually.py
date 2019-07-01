import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from pyramid.arima import auto_arima
from pandas.plotting import register_matplotlib_converters

import numpy as np
class ArimaForecasting():
    def arimaForecasting(self):
        from pandas import datetime
        def parser(x):
            return datetime.strptime(x, '%Y-%m-%d')
        indexedDataset = pd.read_csv("/home/fidel/Downloads/bankClientTwoMonthDataset.csv", date_parser=parser, parse_dates=[0],index_col=0)
        print (indexedDataset)
        #, date_parser=parser, parse_dates=[0],index_col=0
        # indexedDatasetFreq = pd.date_range(indexedDataset,freq=30)
        # print datasetExtracted.dtypes
        # print datasetExtracted
        # indexColm = "Month"
        # targetColm = "#Passengers"
        # salesCarsDatasetDateUpdated.csv(Month,Sales,y-m-d),/Downloads/AirPassengers.csv(Month,#Passengers),/Downloads/aTenDastaset(date,value)



        indexColm = "Date"
        targetColm = "Collections"
        # indexedDatasetDate = pd.to_datetime(indexedDataset, infer_datetime_format=True)
        # indexedDatasetDate.set_index(indexColm,inplace=True)

        indexedDataset = indexedDataset.filter([indexColm,targetColm])
        import pandas.plotting._converter as pandascnv
        pandascnv.register()
        # indexedDataset.to_csv("/home/fidel/Downloads/BIDataset.csv")
        # plt.plot(indexedDataset)
        # plt.show()


        # plt.plot(indexedDataset)
        # plt.show()
        modelAuto = auto_arima(indexedDataset,m=12,max_p=3,max_q=3 ,trace=True, error_action="ignore", suppress_warnings=True,seasonal=True,stepwise=True)
        # modelAutoFit = modelAuto.fit(indexedDataset.values)
        # forecasting
        newDatesLength = 5
        modelAutoPredict,confint = modelAuto.predict(n_periods=newDatesLength,return_conf_int=True)
        indexOfFc = pd.date_range(indexedDataset.index[-1],periods=newDatesLength,freq="MS")
        # modelAutoPredict = pd.DataFrame(modelAutoPredict, index=indexedDataset.index, columns=["Prediction"])

        # making series for plotting
        modelAutoPredictSeries = pd.Series(modelAutoPredict,index=indexOfFc)
        lowerSeries = pd.Series(confint[:,0],index=indexOfFc)
        upperSeries = pd.Series(confint[:,1],index=indexOfFc)

        #plotting the forecasting model
        plt.plot(indexedDataset)
        plt.plot(modelAutoPredictSeries,color="darkgreen")
        plt.fill_between(lowerSeries.index,lowerSeries,upperSeries,color='k',alpha=.15)
        plt.title("arima model forecasting")
        plt.show()

        # plt.plot(modelAutoPredict, label="Prediction")
        # plt.plot(indexedDataset)
        # plt.show()
        from math import sqrt
        from sklearn.metrics import mean_squared_error

        rms = sqrt(mean_squared_error(indexedDataset, modelAutoPredict))
        print(rms)



        rolmean = indexedDataset.rolling(window=12).mean()
        rolstddev = indexedDataset.rolling(window=12).std()

        print (rolmean, rolstddev)
        from statsmodels.tsa.stattools import adfuller
        datasetTest = adfuller(indexedDataset[targetColm], autolag="AIC")
        datasetOutput = pd.Series(datasetTest[0:4],
                                  index=["Test Statistic", "p-value", "#Lags Used", "Number of Observations Used"])
        for key, value in datasetTest[4].items():
            datasetOutput[key] = value
        print (datasetOutput)

        indexedDatasetDiff = indexedDataset.diff(periods=1)
        indexedDatasetDiff = indexedDatasetDiff[1:]


        # order_selection = sm.tsa.arma_order_select_ic(indexedDataset, max_ar=12, max_ma=2, ic="aic")
        # P = order_selection.aic_min_order[0]
        # Q = order_selection.aic_min_order[1]
        D=1
        P=12
        Q=0
        print (P,Q)
        model = ARIMA(indexedDataset[targetColm], order=(P, D, Q))
        modelFit = model.fit()
        prediction = modelFit.forecast(len(indexedDataset))[0]
        plt.plot(prediction, color='red')
        plt.plot(indexedDataset)
        plt.show()


if __name__=="__main__":
    ArimaForecasting().arimaForecasting()

