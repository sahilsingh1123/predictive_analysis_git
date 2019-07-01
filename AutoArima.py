import pandas as pd
from pyramid.arima import auto_arima

class ArimaForecasting():
    def arimaForecasting(dataList,newDateList):
        dfColm = pd.DataFrame(dataList, columns=['target', 'indexed'])
        dfColm['indexed'] = pd.to_datetime(dfColm['indexed'], infer_datetime_format=True)
        print(dfColm.dtypes)
        dfColmInd = dfColm.set_index('indexed')

        modelAuto = auto_arima(dfColmInd, m=12, max_p=2, max_q=2, trace=True, error_action="ignore",
                               suppress_warnings=True, seasonal=True, stepwise=True)
        modelAutoPredict, confint = modelAuto.predict(n_periods=len(newDateList), return_conf_int=True)
        predictedList = []
        for value in modelAutoPredict:
            predictedList.append(value)
        return predictedList
