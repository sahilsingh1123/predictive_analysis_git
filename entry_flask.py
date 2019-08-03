import json
from flask import Flask
from flask import Response
from flask import jsonify
from flask import request
from ml_server_components import FPGrowth
from ml_server_components import Forecasting
from ml_server_components import KMeans
from ml_server_components import SentimentAnalysis

import chi_sqr_original
import linear_reg_original
import pearson_corr_original
import random_forest_classifier_test
import random_forest_regression_test

app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def root():
    response = Response(content_type="application/json")
    requestString = request.data.decode("utf-8")
    requestData = json.loads(requestString)
    print("Request data ", requestData)


    hdfsLocation = ''

    responseData = {}

    fileLocation = requestData.get("fileLocation")
    print("file Location", fileLocation)
    feature_colm_req = requestData.get("features_column")
    print("features colm", feature_colm_req)
    label_colm_req = requestData.get("label_column")
    print("label colm ", label_colm_req)
    algo_name = requestData.get("algorithm_name")
    print ("algo name ", algo_name)
    relation_list = requestData.get("relationship_list")
    print(relation_list)
    relation = requestData.get("relationship")
    print(relation)

    responseData = ''

    try:
        if algo_name == "linear_reg":
            responseData = linear_reg_original.Linear_reg(dataset_add=fileLocation, feature_colm=feature_colm_req, label_colm=label_colm_req, relation_list= relation_list, relation=relation)
        elif algo_name == 'pearson_test':
            responseData = pearson_corr_original.Correlation(dataset_add=fileLocation, feature_colm=feature_colm_req,label_colm=label_colm_req)
        elif algo_name == 'chi_square_test':
            responseData = chi_sqr_original.Chi_sqr(dataset_add=fileLocation, feature_colm=feature_colm_req, label_colm=label_colm_req)
        elif algo_name == 'random_classifier':
            responseData = random_forest_classifier_test.randomClassifier(dataset_add=fileLocation, feature_colm=feature_colm_req,
                                                                           label_colm=label_colm_req)
        elif algo_name == 'random_regressor':
            responseData = random_forest_regression_test.randomClassifier(dataset_add=fileLocation, feature_colm=feature_colm_req,
                                                                           label_colm=label_colm_req)

    except Exception as e:
        print ('exception is = ' + str(e))
        responseData = str(json.dumps({'run_status ' : 'request not processed '})).encode('utf-8')


    # return iter([responseData])

    print(responseData)

    return jsonify(success='success', message = 'it was a success',data= responseData)

@app.route("/forecasting", methods=["POST", "GET"])
def forcasting():
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

            if forecastingAlgorithm == 'arima':
                arima_model_type= j['arima_model_type']
                response_data = Forecasting.perform_forecasting(data=data, count=j['count'], len_type=j['len_type'], model_type=j['model_type'], trendType=j['trendType'], seasonType=j['seasonType'] , forecastAlgorithm= j['forecastingAlgorithm'] , P=j['P'],Q=j['Q'],D=j['D'], arima_model_type=arima_model_type,iterations=j['iterations'])
            else:
                response_data = Forecasting.perform_forecasting(data=data, count=j['count'], len_type=j['len_type'], model_type=j['model_type'], trendType=j['trendType'], seasonType=j['seasonType'] , forecastAlgorithm= j['forecastingAlgorithm'] , P=None,Q=None,D=None, arima_model_type=None,iterations=None)
    except Exception as e:
        print('exception = ' + str(e))
        #response_data = str(json.dumps({'run_status': 'sorry! unable to process your request'})).encode('utf-8')
        response_data = {'run_status': 'sorry! unable to process your request'}
    status = '200 OK'

    return jsonify(success='success', message='ml_server_response', data=response_data)




if (__name__=='__main__'):

    app.run(host='0.0.0.0', port = 3334, debug=False)


