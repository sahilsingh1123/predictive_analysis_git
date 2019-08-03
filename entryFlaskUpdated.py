import json

import matplotlib as mpl
from flask import Flask
from flask import Response
from flask import jsonify
from flask import request

from PredictionAlgorithms import chi_sqr_original
from PredictionAlgorithms import pearson_corr_original
from PredictionAlgorithms.FeaturesSelection import FeaturesSelection
from PredictionAlgorithms.LassoRegression import LassoRegressionModel
from PredictionAlgorithms.LinearRegression import LinearRegressionModel
from PredictionAlgorithms.RidgeRegression import RidgeRegressionModel
from PredictionAlgorithms.ml_server_components import FPGrowth
from PredictionAlgorithms.ml_server_components import Forecasting
from PredictionAlgorithms.ml_server_components import KMeans
from PredictionAlgorithms.ml_server_components import SentimentAnalysis

mpl.use("TkAgg")

app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def root():
    response = Response(content_type="application/json")
    requestString = request.data.decode("utf-8")
    requestData = json.loads(requestString)
    fileLocation = requestData.get("fileLocation")
    feature_colm_req = requestData.get("features_column")
    label_colm_req = requestData.get("label_column")
    algo_name = requestData.get("algorithm_name")
    relation_list = requestData.get("relationship_list")
    relation = requestData.get("relationship")
    trainDataPercentage = requestData.get('trainDataPercentage')
    modelId = requestData.get('modelUUID')
    responseData = ''
    locationAddress='hdfs://fidel-Latitude-E5570:9000/dev/dmxdeepinsight/datasets/'

    try:
        if algo_name == "linear_reg":
            responseData = LinearRegressionModel(trainDataPercentage).linearReg(dataset_add=fileLocation,
                                                                                               feature_colm=feature_colm_req,
                                                                                               label_colm=label_colm_req,
                                                                                               relation_list=relation_list,
                                                                                               relation=relation,
                                                                                               userId=modelId,locationAddress=locationAddress)
        elif algo_name == 'pearson_test':
            responseData = pearson_corr_original.Correlation(dataset_add=fileLocation, feature_colm=feature_colm_req,
                                                             label_colm=label_colm_req)
        elif algo_name == 'chi_square_test':
            responseData = chi_sqr_original.Chi_sqr(dataset_add=fileLocation, feature_colm=feature_colm_req,
                                                    label_colm=label_colm_req)
        # elif algo_name == 'random_classifier':
        #     responseData = RandomClassifierModel.randomClassifier(dataset_add=fileLocation,
        #                                                           feature_colm=feature_colm_req,
        #                                                           label_colm=label_colm_req,
        #                                                           relation_list=relation_list, relation=relation,
        #                                                           userId=modelId)
        elif algo_name == 'random_regressor' or algo_name=="random_classifier":
            # responseData = RandomRegressionModel.randomRegressor(dataset_add=fileLocation,
            #                                                      feature_colm=feature_colm_req,
            #                                                      label_colm=label_colm_req, relation_list=relation_list,
            #                                                      relation=relation, userId=modelId)
            featureSelectionObj=FeaturesSelection()
            responseData=featureSelectionObj.featuresSelection(dataset_add=fileLocation,
                                                                  feature_colm=feature_colm_req,
                                                                  label_colm=label_colm_req,
                                                                  relation_list=relation_list, relation=relation,
                                                                  userId=modelId,algoName=algo_name)
        elif algo_name == 'lasso_reg':
            responseData = LassoRegressionModel(trainDataRatio=trainDataPercentage).lassoRegression(
                dataset_add=fileLocation, feature_colm=feature_colm_req, label_colm=label_colm_req,
                relation_list=relation_list, relation=relation, userId=modelId)
        elif algo_name == 'ridge_reg':
            responseData = RidgeRegressionModel().ridgeRegression(dataset_add=fileLocation,
                                                                  feature_colm=feature_colm_req,
                                                                  label_colm=label_colm_req,
                                                                  relation_list=relation_list, relation=relation,
                                                                  userId=modelId)
    except Exception as e:
        print('exception is = ' + str(e))
        responseData = str(json.dumps({'run_status ': 'request not processed '})).encode('utf-8')
    print(responseData)
    return jsonify(success='success', message='it was a success', data=responseData)

@app.route("/forecasting", methods=["POST", "GET"])
def forecasting():
    response = Response(content_type="application/json")
    requestString = request.data.decode("utf-8")
    requestData = json.loads(requestString)
    print("Request data ", requestData)


    j = json.loads(requestString)
    algorithm = j['algorithm']
    data = j['data']
    response_data = ''
    print(algorithm)
    try:
        if algorithm == 'kmeans':
            response_data = KMeans.perform_k_means(data=data, no_of_clusters=j['number_of_clusters'])
        elif algorithm == 'fp-growth':
            print('This is a FP-Growth request!!!!')
            response_data = FPGrowth.perform_fp_growth(data=data)
            print('Sending FP-Growth Response!!!')
        elif algorithm == 'sentimentAnalysis':
            response_data = SentimentAnalysis.perform_sentiment_analysis(data=data)
        elif algorithm == 'forecasting':
            forecastingAlgorithm = j['forecastingAlgorithm']
            alpha = j.get("alpha")
            beta = j.get("beta")
            gamma = j.get("gamma")
            isTrending = j.get("isTrend")
            isSeasonal = j.get("isSeason")
            seasonalPeriodsManual = j.get("seasonality")
            seasonalP = j.get("seasonalP")
            seasonalD = j.get("seasonalD")
            seasonalQ = j.get("seasonalQ")
            data = data
            count = j.get('count')
            len_type = j.get('len_type')
            model_type = j.get('model_type')
            trendType = j.get('trendType')
            seasonType = j.get('seasonType')
            forecastAlgorithm = j.get('forecastingAlgorithm')
            P = j.get('P')
            Q = j.get('Q')
            D = j.get('D')
            arima_model_type = j.get('arima_model_type')
            iterations = j.get('iterations')
            confIntPara = '0.95'

            forecastClass = Forecasting.ForecastingModel(alpha=alpha, beta=beta, gamma=gamma, isTrending=isTrending,
                                                         isSeasonal=isSeasonal,
                                                         seasonalPeriodsManual=seasonalPeriodsManual,
                                                         seasonalP=seasonalP, seasonalD=seasonalD, seasonalQ=seasonalQ,confIntPara=confIntPara)
            response_data=forecastClass.perform_forecasting(data=data, count=count, len_type=len_type,model_type=model_type,trendType=trendType,seasonType=seasonType,forecastAlgorithm=forecastAlgorithm,P=P,Q=Q,D=Q,arima_model_type=arima_model_type,iterations=iterations)

    except Exception as e:
        print('exception = ' + str(e))
        response_data = str(json.dumps({'run_status': 'sorry! unable to process your request'})).encode('utf-8')
    status = '200 OK'

    return jsonify(success='success', message='ml_server_response', data=response_data)




if (__name__=='__main__'):

    app.run(host='0.0.0.0', port = 3334, debug=False)



