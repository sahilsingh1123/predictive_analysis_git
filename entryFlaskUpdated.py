import json

from flask import Flask
from flask import Response
from flask import jsonify
from flask import request

from pyspark.sql import SparkSession
from PredictionAlgorithms.PredictiveExceptionHandling import PredictiveExceptionHandling
from PredictionAlgorithms.PredictiveClassificationModel import PredictiveClassificationModel
from PredictionAlgorithms.PredictiveFeaturesSelection import PredictiveFeaturesSelection
from PredictionAlgorithms.PredictivePrediction import PredictivePrediction
from PredictionAlgorithms.PredictiveRegressionModel import PredictiveRegressionModel
from PredictionAlgorithms.PredictiveConstants import PredictiveConstants
from PredictionAlgorithms.ml_server_components import FPGrowth
from PredictionAlgorithms.ml_server_components import Forecasting
from PredictionAlgorithms.ml_server_components import KMeans
from PredictionAlgorithms.ml_server_components import SentimentAnalysis

# used in ETL--- ----------
# spark = SparkSession.builder.master(sparkURL).appName("DMXDeepInsightpy").getOrCreate()


spark = \
    SparkSession.builder.appName('predictive_Analysis').master('local[*]').getOrCreate()
spark.sparkContext.setLogLevel('ERROR')


app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def root():
    response = Response(content_type="application/json")
    requestString = request.data.decode("utf-8")
    requestData = json.loads(requestString)
    fileLocation = requestData.get(PredictiveConstants.FILELOCATION)
    feature_colm_req = requestData.get(PredictiveConstants.PREDICTOR)
    label_colm_req = requestData.get(PredictiveConstants.TARGET)
    algo_name = requestData.get(PredictiveConstants.ALGORITHMNAME)
    relation_list = requestData.get(PredictiveConstants.RELATIONSHIP_LIST)
    relation = requestData.get(PredictiveConstants.RELATIONSHIP)
    trainDataPercentage = requestData.get(PredictiveConstants.TRAINDATAPERCENTAGE)
    modelId = requestData.get(PredictiveConstants.MODELUUID)
    requestType = requestData.get(PredictiveConstants.REQUESTTYPE)
    modelStorageLocation = requestData.get(PredictiveConstants.MODELSTORAGELOCATION)
    locationAddress = requestData.get(PredictiveConstants.LOCATIONADDRESS)
    datasetName = requestData.get(PredictiveConstants.DATASETNAME)
    modelSheetName = requestData.get(PredictiveConstants.MODELSHEETNAME)
    responseData = {}
    # locationAddress='hdfs://10.171.0.151:9000/dev/dmxdeepinsight/datasets/'

    # locationAddress is where writing of parquet dataset take place
    # fileLocation is where dataframe is stored

    # waiting for the change from UI end
    regParam=0.05

    try:
        if (algo_name == PredictiveConstants.LINEAR_REG or algo_name == PredictiveConstants.LASSO_REG
            or algo_name == PredictiveConstants.RIDGE_REG
                or algo_name == "RandomForestAlgo" or algo_name == "GradientBoostAlgo") \
                and requestType == None:
            predictiveRegressionModelObj = \
                PredictiveRegressionModel(trainDataRatio=trainDataPercentage,
                                          dataset_add=fileLocation,
                                          feature_colm=feature_colm_req,
                                          label_colm=label_colm_req,
                                          relation_list=relation_list,
                                          relation=relation,
                                          userId=modelId,
                                          locationAddress=locationAddress,
                                          algoName=algo_name,
                                          modelSheetName=modelSheetName,
                                          spark = spark
                                          )
            if algo_name==PredictiveConstants.LINEAR_REG:
                responseData = \
                    predictiveRegressionModelObj.linearModel()
            if algo_name == PredictiveConstants.RIDGE_REG or algo_name == PredictiveConstants.LASSO_REG:
                responseData = \
                    predictiveRegressionModelObj.ridgeLassoModel(regParam=regParam)
            if algo_name == "RandomForestAlgo":
                responseData = predictiveRegressionModelObj.randomForestRegressorModel()
            if algo_name == "GradientBoostAlgo":
                            responseData = predictiveRegressionModelObj.gradientBoostRegressorModel()

        if (algo_name == PredictiveConstants.LOGISTIC_REG):
            PredictiveClassificationModelObj = \
                PredictiveClassificationModel(trainDataRatio=trainDataPercentage,
                                          dataset_add=fileLocation,
                                          feature_colm=feature_colm_req,
                                          label_colm=label_colm_req,
                                          relation_list=relation_list,
                                          relation=relation,
                                          userId=modelId,
                                          locationAddress=locationAddress,
                                          algoName=algo_name)
            responseData = PredictiveClassificationModelObj.logisticRegression()

        #for features selection only
        #not for predictive models--------------------------------------
        if (algo_name == PredictiveConstants.RANDOMREGRESSOR or algo_name == PredictiveConstants.RANDOMCLASSIFIER) \
                and requestType == None:
            PredictiveFeaturesSelectionObj = PredictiveFeaturesSelection(spark=spark)
            responseData = \
                PredictiveFeaturesSelectionObj.featuresSelection(dataset_add=fileLocation,
                                                      feature_colm=feature_colm_req,
                                                      label_colm=label_colm_req,
                                                      relation_list=relation_list, relation=relation,
                                                      userId=modelId,algoName=algo_name,
                                                      locationAddress=locationAddress)

        if requestType == PredictiveConstants.PREDICTION:
            predictivePredictionObj = PredictivePrediction(dataset_add=fileLocation,
                                                           feature_colm=feature_colm_req,
                                                           label_colm=None,
                                                           relation_list=relation_list,
                                                           relation=relation,
                                                           trainDataRatio=None,
                                                           userId=modelId,
                                                           locationAddress=locationAddress,
                                                           algoName=algo_name,
                                                           modelStorageLocation=modelStorageLocation,
                                                           modelSheetName=modelSheetName,
                                                           datasetName = datasetName,
                                                           spark=spark)
            responseData = \
                predictivePredictionObj.loadModel()


        responseData["run_status"] = "success"



    except Exception as e:
        print('exception is = ' + str(e))
        # responseData = str(json.dumps({'run_status ': 'request not processed '})).encode('utf-8')
        responseData = PredictiveExceptionHandling.exceptionHandling(e)
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
            locationAddress = j.get("locationAddress")
            modelName = j.get("modelName")
            columnsNameList = j.get("columnsNameList")
            sheetId = j.get("worksheetID")

            #for UI changes
            confIntPara = '0.95'

            forecastClass = \
                Forecasting.ForecastingModel(alpha=alpha, beta=beta, gamma=gamma, isTrending=isTrending,
                                                         isSeasonal=isSeasonal,
                                                         seasonalPeriodsManual=seasonalPeriodsManual,
                                                         seasonalP=seasonalP, seasonalD=seasonalD,
                                                         seasonalQ=seasonalQ,confIntPara=confIntPara)
            response_data = \
                forecastClass.forecastingTimeSeries(data=data, count=count, len_type=len_type,
                                                    model_type=model_type, trendType=trendType,
                                                    seasonType=seasonType, forecastAlgorithm=forecastAlgorithm,
                                                    P=P, Q=Q, D=D, arima_model_type=arima_model_type,
                                                    iterations=iterations,
                                                    columnsNameList=columnsNameList,
                                                    locationAddress=locationAddress,
                                                    modelName=modelName,
                                                    sheetId=sheetId)

    except Exception as e:
        print('exception = ' + str(e))
        response_data = str(json.dumps({'run_status': 'sorry! unable to process your request'})).encode('utf-8')
    status = '200 OK'

    return jsonify(success='success', message='ml_server_response', data=response_data)




if (__name__=='__main__'):

    app.run(host='0.0.0.0', port = 3334, debug=False)



