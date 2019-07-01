import json
import numpy as np
import pandas as pd
import time
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from pyramid.arima import auto_arima

class ForecastingModel():
    def perform_forecasting(data, count, len_type, model_type, trendType, seasonType, forecastAlgorithm, P, Q, D,
                            arima_model_type):
        start_time = time.time()
        print("count = " + str(count))
        print("len_type = " + str(len_type))
        print("model_type = " + str(model_type))
        print("trendType = " + str(trendType))
        print("seasonType = " + str(seasonType))
        trendType = None if trendType == 'null' else trendType
        if model_type == 'Automatic':
            seasonType = 'multiplicative'
        elif seasonType == 'null':
            seasonType = None
        df = pd.DataFrame(data)
        df = df[df[len(df.columns) - 1] != 'null']
        df[df.columns[-1]] = pd.to_datetime(df[df.columns[-1]], infer_datetime_format=True)
        # dfColm = pd.DataFrame(data, columns=['target', 'indexed'])
        # dfColm['indexed'] = pd.to_datetime(dfColm['indexed'], infer_datetime_format=True)
        # print(dfColm.dtypes)
        # dfColmInd = dfColm.set_index('indexed')
        newdates = []
        calculatedPDQ = []
        if len_type == 'SecondaryYear':
            newdates = pd.date_range(df.iloc[-1, -1] + relativedelta(years=1), periods=count, freq='YS')
            seasonalPeriods = 1 if seasonType else None
            mValue = 1
        elif len_type == 'SecondaryMonth':
            newdates = pd.date_range(df.iloc[-1, -1] + relativedelta(months=1), periods=count, freq='MS')
            mValue = 12
            seasonalPeriods = 12 if seasonType else None
        elif len_type == 'SecondaryQuarter':
            newdates = pd.date_range(df.iloc[-1, -1] + relativedelta(months=3), periods=count, freq='QS')
            seasonalPeriods = 4 if seasonType else None
            mValue = 4
        elif len_type == 'SecondaryWeekNumber':
            newdates = pd.date_range(df.iloc[-1, -1] + relativedelta(weeks=1), periods=count, freq='W')
            seasonalPeriods = 52 if seasonType else None
            mValue = 12
        elif len_type == 'SecondaryDay':
            newdates = pd.date_range(df.iloc[-1, -1] + relativedelta(days=1), periods=count, freq='D')
            seasonalPeriods = 365 if seasonType else None
            mValue = 12
        newdates = pd.to_datetime(newdates)
        newdates = pd.DataFrame(newdates)
        resultdf = []
        # predictedListAutoArima = ForecastingModel.arimaForecasting(dataList=data, newDateList=newdates,mValuePara=mValue)

        if forecastAlgorithm == 'arima':
            if arima_model_type == 'arimaAutomatic':

                for x in range(0, (len(df.columns) - 1)):
                    resultdf.append(
                        ForecastingModel.applyforecasting(pd.DataFrame(df.ix[:, x]), newdates, seasonalPeriods, None, seasonType,
                                         forecastAlgorithm, int(P), int(Q), int(D), arima_model_type,dataList=data, newDateList=newdates,mValuePara=mValue))
            elif arima_model_type == 'arimaCustom':
                for x in range(0, (len(df.columns) - 1)):
                    resultdf.append(
                        ForecastingModel.applyforecasting(pd.DataFrame(df.ix[:, x]), newdates, seasonalPeriods, None, seasonType,
                                         forecastAlgorithm, int(P), int(Q), int(D), arima_model_type))


        if forecastAlgorithm == "holtw" or forecastAlgorithm == "holtW":
            if model_type == 'Automatic':
                for x in range(0, (len(df.columns) - 1)):
                    resultdf.append(
                        ForecastingModel.applyforecasting(pd.DataFrame(df.ix[:, x]), newdates, seasonalPeriods, None, seasonType,
                                         forecastAlgorithm, 0, 0, 0, arima_model_type=None))
            elif model_type == 'Custom':
                for x in range(0, (len(df.columns) - 1)):
                    resultdf.append(
                        ForecastingModel.applyforecasting(pd.DataFrame(df.ix[:, x]), newdates, seasonalPeriods, trendType, seasonType,
                                         forecastAlgorithm, 0, 0, 0, arima_model_type=None))
        newDatesList = []
        for date in newdates[0].tolist():
            newDatesList.append(str(date))
        print(resultdf)
        # predictedListAutoArimaMod = [predictedListAutoArima]
        # print(predictedListAutoArimaMod)
        json_response = {'run_status': 'success', 'pred': resultdf, 'foredate': newDatesList,
                         'arimaParams': calculatedPDQ, 'execution_time': time.time() - start_time}
        # return str(json.dumps(json_response)).encode('utf-8')
        return json_response

    def applyforecasting(dataset, newdates, seasonalPeriods, trendType, seasonType, forecastingAlgorithm, P, Q, D,
                         arima_model_type,dataList,newDateList,mValuePara):
        # Year Time Series may be short to perform forecasting
        tempdf = pd.DataFrame().reindex_like(newdates)
        tempdf.insert(0, "pred", 0)
        if forecastingAlgorithm == "holtW" or forecastingAlgorithm == "holtw":
            shouldDampTrendComponent = trendType in ['additive', 'multiplicative'] and seasonType is None
            fit1 = ExponentialSmoothing(np.asarray(dataset.iloc[:, 0].values), seasonal_periods=seasonalPeriods,
                                        damped=shouldDampTrendComponent, trend=trendType, seasonal=seasonType).fit()
            tempdf['pred'] = fit1.forecast(len(newdates))
        if forecastingAlgorithm == "arima" and arima_model_type == "arimaCustom":
            arimaRes = ForecastingModel.startARIMAForecasting(np.asarray(dataset.iloc[:, 0].values).tolist(), int(P), int(D), int(Q),
                                             newdates)
            tempdf['pred'] = arimaRes
        if forecastingAlgorithm == "arima" and arima_model_type == "arimaAutomatic":
            arimaRes = ForecastingModel.arimaForecasting(dataList,newDateList,mValuePara)
            tempdf['pred'] = arimaRes

        resultds = tempdf['pred'].tolist()
        if np.isnan(resultds).any():
            resultds = []
        return resultds

    def startARIMAForecasting(dataset, P, D, Q, newdates):
        model = ARIMA(dataset, order=(P, D, Q))

        model_fit = model.fit(disp=0)
        prediction = model_fit.forecast(len(newdates))[0]
        return prediction

    def arimaForecasting(dataList, newDateList,mValuePara):
        dfColm = pd.DataFrame(dataList, columns=['target', 'indexed'])
        dfColm['indexed'] = pd.to_datetime(dfColm['indexed'], infer_datetime_format=True)
        dfColmInd = dfColm.set_index('indexed')

        modelAuto = auto_arima(dfColmInd, m=mValuePara, max_p=3, max_q=3, trace=True,
                               error_action="ignore",
                               suppress_warnings=True, seasonal=True, stepwise=True, max_order=10, maxiter=5)
        modelAutoPredict, confint = modelAuto.predict(n_periods=len(newDateList), return_conf_int=True)
        predictedList = []
        for value in modelAutoPredict:
            predictedList.append(value)
        return predictedList
