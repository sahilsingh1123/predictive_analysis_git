
import json
import numpy as np
import pandas as pd
import time
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.arima_model import ARIMA


def perform_forecasting(data, count, len_type, model_type, trendType, seasonType, forecastAlgorithm):
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
    newdates = []
    if len_type == 'SecondaryYear':
        newdates = pd.date_range(df.iloc[-1, -1] + relativedelta(years=1), periods=count, freq='YS')
    seasonalPeriods = 1 if seasonType else None
    elif len_type == 'SecondaryMonth':
    newdates = pd.date_range(df.iloc[-1, -1] + relativedelta(months=1), periods=count, freq='MS')
    seasonalPeriods = 12 if seasonType else None
    elif len_type == 'SecondaryQuarter':
    newdates = pd.date_range(df.iloc[-1, -1] + relativedelta(months=3), periods=count, freq='QS')
    seasonalPeriods = 4 if seasonType else None
    elif len_type == 'SecondaryWeek':
    newdates = pd.date_range(df.iloc[-1, -1] + relativedelta(weeks=1), periods=count, freq='W')
    seasonalPeriods = 52 if seasonType else None
    elif len_type == 'SecondaryDay':
    newdates = pd.date_range(df.iloc[-1, -1] + relativedelta(days=1), periods=count, freq='D')
    seasonalPeriods = 365 if seasonType else None
    newdates = pd.to_datetime(newdates)
    newdates = pd.DataFrame(newdates)
    resultdf = []
    if forecastAlgorithm == 'arima':
        for x in range(0, (len(df.columns) - 1)):
            resultdf.append(smoothingforecasting(pd.DataFrame(df.ix[:, x]), newdates, seasonalPeriods, None, seasonType,
                                                 forecastAlgorithm))
    if forecastAlgorithm == "holtw" or forecastAlgorithm == "holtW":
        if model_type == 'Automatic':
            for x in range(0, (len(df.columns) - 1)):
            resultdf.append(smoothingforecasting(pd.DataFrame(df.ix[:, x]), newdates, seasonalPeriods, None, seasonType,
                                                 forecastAlgorithm))
    elif model_type == 'Custom':
        for x in range(0, (len(df.columns) - 1)):
            resultdf.append(
                smoothingforecasting(pd.DataFrame(df.ix[:, x]), newdates, seasonalPeriods, trendType, seasonType,
                                     forecastAlgorithm))
    newDatesList = []
    for date in newdates[0].tolist():
        newDatesList.append(str(date))
    print(resultdf)
    json_response = {'run_status': 'success', 'pred': resultdf, 'foredate': newDatesList,
                     'execution_time': time.time() - start_time}
    return str(json.dumps(json_response)).encode('utf-8')


def smoothingforecasting(dataset, newdates, seasonalPeriods, trendType, seasonType, forecastingAlgorithm):


    # Year Time Series may be short to perform forecasting
    tempdf = pd.DataFrame().reindex_like(newdates)
    tempdf.insert(0, "pred", 0)
    if forecastingAlgorithm == "holtW" or forecastingAlgorithm == "holtw":
        shouldDampTrendComponent = trendType in ['additive', 'multiplicative'] and seasonType is None
    fit1 = ExponentialSmoothing(np.asarray(dataset.iloc[:, 0].values), seasonal_periods=seasonalPeriods,
                                damped=shouldDampTrendComponent, trend=trendType, seasonal=seasonType).fit()
    tempdf['pred'] = fit1.forecast(len(newdates))
    if forecastingAlgorithm == "arima":
        arimaRes = startARIMAForecasting(np.asarray(dataset.iloc[:, 0].values).tolist(), 1, 0, 0, newdates)
    tempdf['pred'] = arimaRes

    resultds = tempdf['pred'].tolist()
    if np.isnan(resultds).any():
        resultds = []
    return resultds


def startARIMAForecasting(Actual, P, D, Q, newdates):
    model = ARIMA(Actual, order=(P, D, Q))


    model_fit = model.fit(disp=0)
    prediction = model_fit.forecast(len(newdates))[0]
    return prediction



