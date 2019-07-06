import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
class ArimaForecasting():
    def arimaForecasting(self):
        dataset = pd.read_csv("/home/fidel/Downloads/AirPassengers.csv", infer_datetime_format=True, header=0, delimiter=",")
        # print (dataset.columns['Profit'])
        # print dataset.dtypes

        # print datasetExtracted.dtypes
        # print datasetExtracted
        indexColm="Month"
        targetColm="#Passengers"
        dataset = dataset.filter([indexColm,targetColm])
        indexedDataset = dataset.set_index([indexColm])
        print (indexedDataset)

        # plt.xlabel("Date")
        # plt.ylabel("Freight")
        # plt.plot(indexedDataset)
        # plt.show()

        import datetime

        # calculating the rolling mean and std
        rolmean = indexedDataset.rolling(window=12).mean()
        rolstddev = indexedDataset.rolling(window=12).std()

        print (rolmean, rolstddev)
        # originalDataset = plt.plot(indexedDataset, color="blue", label = "Original")
        # meanData = plt.plot(rolmean, color="red", label = "mean")
        # stdDevData = plt.plot(rolstddev, color="green", label="stddev")
        # plt.show()

        # dickey fuller test
        from statsmodels.tsa.stattools import adfuller
        datasetTest = adfuller(indexedDataset[targetColm], autolag="AIC")
        datasetOutput = pd.Series(datasetTest[0:4], index=["Test Statistic", "p-value", "#Lags Used", "Number of Observations Used"])
        for key,value in datasetTest[4].items():
            datasetOutput[key] = value
        print (datasetOutput)

        # Estimating trend
        indexedDataset_logScale = np.log(indexedDataset)
        # plt.plot(indexedDataset_logScale)
        # plt.show()

        movingAverage = indexedDataset_logScale.rolling(window=12).mean()
        movingSTD = indexedDataset_logScale.rolling(window=12).std()
        # plt.plot(indexedDataset_logScale)
        # plt.plot(movingAverage, color='red')
        # plt.show()


        # Get the difference between the moving average and the actual number of passengers
        datasetLogScaleMinusMovingAverage = indexedDataset_logScale - movingAverage
        datasetLogScaleMinusMovingAverage.head(12)
        # Remove Nan Values
        datasetLogScaleMinusMovingAverage.dropna(inplace=True)
        datasetLogScaleMinusMovingAverage.head(10)

        from statsmodels.tsa.stattools import adfuller
        def test_stationarity(timeseries):

            # Determing rolling statistics
            movingAverage = timeseries.rolling(window=12).mean()
            movingSTD = timeseries.rolling(window=12).std()

            # Plot rolling statistics:
            # orig = plt.plot(timeseries, color='blue', label='Original')
            # mean = plt.plot(movingAverage, color='red', label='Rolling Mean')
            # std = plt.plot(movingSTD, color='black', label='Rolling Std')
            # plt.legend(loc='best')
            # plt.title('Rolling Mean & Standard Deviation')
            # plt.show(block=False)

            # Perform Dickey-Fuller test:
            print('Results of Dickey-Fuller Test:')
            dftest = adfuller(timeseries[targetColm], autolag='AIC')
            dfoutput = pd.Series(dftest[0:4],
                                 index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
            for key, value in dftest[4].items():
                dfoutput['Critical Value (%s)' % key] = value
            print(dfoutput)

        test_stationarity(datasetLogScaleMinusMovingAverage)
        #
        # exponentialDecayWeightedAverage = indexedDataset_logScale.ewm(halflife=12, min_periods=0, adjust=True).mean()
        # plt.plot(indexedDataset_logScale)
        # plt.plot(exponentialDecayWeightedAverage, color='red')
        #
        # datasetLogScaleMinusMovingExponentialDecayAverage = indexedDataset_logScale - exponentialDecayWeightedAverage
        # test_stationarity(datasetLogScaleMinusMovingExponentialDecayAverage)
        #
        datasetLogDiffShifting = indexedDataset_logScale - indexedDataset_logScale.shift()
        # plt.plot(datasetLogDiffShifting)
        #
        # datasetLogDiffShifting.dropna(inplace=True)
        # test_stationarity(datasetLogDiffShifting)
        #
        # from statsmodels.tsa.seasonal import seasonal_decompose
        # decomposition = seasonal_decompose(indexedDataset_logScale)
        #
        # trend = decomposition.trend
        # seasonal = decomposition.seasonal
        # residual = decomposition.resid
        #
        # plt.subplot(411)
        # plt.plot(indexedDataset_logScale, label='Original')
        # plt.legend(loc='best')
        # plt.subplot(412)
        # plt.plot(trend, label='Trend')
        # plt.legend(loc='best')
        # plt.subplot(413)
        # plt.plot(seasonal, label='Seasonality')
        # plt.legend(loc='best')
        # plt.subplot(414)
        # plt.plot(residual, label='Residuals')
        # plt.legend(loc='best')
        # plt.tight_layout()
        #
        # decomposedLogData = residual
        # decomposedLogData.dropna(inplace=True)
        # test_stationarity(decomposedLogData)
        #
        # # ACF and PACF plots:
        # from statsmodels.tsa.stattools import acf, pacf
        #
        # lag_acf = acf(datasetLogDiffShifting, nlags=20)
        # lag_pacf = pacf(datasetLogDiffShifting, nlags=20, method='ols')
        #
        # # Plot ACF:
        # plt.subplot(121)
        # plt.plot(lag_acf)
        # plt.axhline(y=0, linestyle='--', color='gray')
        # plt.axhline(y=-1.96 / np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='gray')
        # plt.axhline(y=1.96 / np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='gray')
        # plt.title('Autocorrelation Function')
        #
        # # Plot PACF:
        # plt.subplot(122)
        # plt.plot(lag_pacf)
        # plt.axhline(y=0, linestyle='--', color='gray')
        # plt.axhline(y=-1.96 / np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='gray')
        # plt.axhline(y=1.96 / np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='gray')
        # plt.title('Partial Autocorrelation Function')
        # plt.tight_layout()
        #
        # from statsmodels.tsa.arima_model import ARIMA
        #
        # # AR MODEL
        # model = ARIMA(indexedDataset_logScale, order=(2, 1, 0))
        # results_AR = model.fit(disp=-1)
        # plt.plot(datasetLogDiffShifting)
        # plt.plot(results_AR.fittedvalues, color='red')
        # plt.title('RSS: %.4f' % sum((results_AR.fittedvalues - datasetLogDiffShifting["#Passengers"]) ** 2))
        # print('Plotting AR model')
        #
        # # MA MODEL
        # model = ARIMA(indexedDataset_logScale, order=(0, 1, 2))
        # results_MA = model.fit(disp=-1)
        # plt.plot(datasetLogDiffShifting)
        # plt.plot(results_MA.fittedvalues, color='red')
        # plt.title('RSS: %.4f' % sum((results_MA.fittedvalues - datasetLogDiffShifting["#Passengers"]) ** 2))
        # print('Plotting AR model')
        #
        from statsmodels.tsa.arima_model import ARIMA
        model = ARIMA(indexedDataset_logScale, order=(2, 1, 2))
        results_ARIMA = model.fit(disp=-1)
        # plt.plot(datasetLogDiffShifting)
        # plt.plot(results_ARIMA.fittedvalues, color='red')
        # plt.title('RSS: %.4f' % sum((results_ARIMA.fittedvalues - datasetLogDiffShifting[targetColm]) ** 2))
        # plt.show()

        predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
        print (predictions_ARIMA_diff.head())

        # Convert to cumulative sum
        predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
        print (predictions_ARIMA_diff_cumsum.head())

        predictions_ARIMA_log = pd.Series(indexedDataset_logScale[targetColm].ix[0],
                                          index=indexedDataset_logScale[targetColm].index)
        predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
        predictions_ARIMA_log.head()

        predictions_ARIMA = np.exp(predictions_ARIMA_log)
        plt.plot(indexedDataset)
        plt.plot(predictions_ARIMA)
        plt.title('RMSE: %.4f' % np.sqrt(
            sum((predictions_ARIMA - indexedDataset["#Passengers"]) ** 2) / len(indexedDataset["#Passengers"])))
        plt.show()



if __name__=="__main__":
    ArimaForecasting().arimaForecasting()


