from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.sql import SparkSession
from pyspark.sql.types import *

from PredictionAlgorithms.PredictiveUtilities import PredictiveUtilities
from PredictionAlgorithms.PredictiveEvaluation import PredictiveEvaluation
from PredictionAlgorithms.PredictiveConstants import PredictiveConstants

spark = \
    SparkSession.builder.appName('predictive_Analysis').master('local[*]').getOrCreate()
spark.sparkContext.setLogLevel('ERROR')
'''
trainDataRation = ration of training data, take input from the user
learningRate = learning rate applied on the model, take input from the user
dataset_add = dataset address, from the user
feature_colm = column as a features is taken from the user
relation_list = relationship list of each column needed to be applied
relation = whether it is linear relation or non linear taken from the user end
'''


class PredictiveRegressionModel(PredictiveEvaluation):
    def __init__(self, trainDataRatio, dataset_add, feature_colm, label_colm, relation_list,
                 relation, userId, locationAddress, algoName, modelSheetName,spark):
        self.trainDataRatio = trainDataRatio
        self.datasetAdd = dataset_add
        self.featuresColmList = feature_colm
        self.labelColmList = label_colm
        self.relationshipList = relation_list
        self.relation = relation
        self.userId = userId
        self.locationAddress = locationAddress
        self.algoName = algoName
        self.modelSheetName = PredictiveConstants.PREDICTION_ + modelSheetName
        # self.spark = spark


        # only for etlpart of the dataset
        # PredictiveUtilities = PredictiveUtilities()

        ETLOnDatasetStats = \
            PredictiveUtilities.ETLOnDataset(datasetAdd=self.datasetAdd,
                                                     featuresColmList=self.featuresColmList,
                                                     labelColmList=self.labelColmList,
                                                     relationshipList=self.relationshipList,
                                                     relation=self.relation,
                                                     trainDataRatio=self.trainDataRatio,
                                                     spark=spark,
                                                     userId=userId)
        self.dataset = ETLOnDatasetStats.get(PredictiveConstants.DATASET)
        self.featuresColm = ETLOnDatasetStats.get(PredictiveConstants.FEATURESCOLM)
        self.labelColm = ETLOnDatasetStats.get(PredictiveConstants.LABELCOLM)
        self.trainData = ETLOnDatasetStats.get(PredictiveConstants.TRAINDATA)
        self.testData = ETLOnDatasetStats.get(PredictiveConstants.TESTDATA)
        self.idNameFeaturesOrdered = ETLOnDatasetStats.get(PredictiveConstants.IDNAMEFEATURESORDERED)


    def linearModel(self):
        linearRegressionModelfit = \
            LinearRegression(featuresCol=self.featuresColm, labelCol=self.labelColm,
                             predictionCol=self.modelSheetName)
        regressor = linearRegressionModelfit.fit(self.trainData)
        regressionStat = self.regressionModelEvaluation(regressor=regressor, spark=spark)

        # persisting the model
        modelName = "linearRegressionModel"
        extention = ".parquet"
        modelStorageLocation = self.locationAddress + self.userId.upper() + modelName.upper() + extention
        regressor.write().overwrite().save(modelStorageLocation)

        regressionStat["modelPersistLocation"] = {"modelName": modelName,
                                                  PredictiveConstants.MODELSTORAGELOCATION: modelStorageLocation}

        return regressionStat

    def ridgeLassoModel(self, regParam):
        regParam = 0.05 if regParam == None else float(regParam)
        elasticNetPara = 1 if self.algoName == PredictiveConstants.LASSO_REG else 0
        ridgeLassoModelFit = \
            LinearRegression(featuresCol=self.featuresColm,
                             labelCol=self.labelColm,
                             elasticNetParam=elasticNetPara,
                             regParam=regParam,
                             predictionCol=self.modelSheetName)
        regressor = ridgeLassoModelFit.fit(self.trainData)
        regressionStat = self.regressionModelEvaluation(regressor=regressor, spark=spark)

        # persisting model
        modelName = "lassoRegressionModel" if self.algoName == PredictiveConstants.LASSO_REG \
            else "ridgeRegressionModel"
        extention = ".parquet"
        modelStorageLocation = self.locationAddress + self.userId.upper() + modelName.upper() + extention
        regressor.write().overwrite().save(modelStorageLocation)

        regressionStat["modelPersistLocation"] = {"modelName": modelName,
                                                  "modelStorageLocation": modelStorageLocation}

        return regressionStat

    # for future
    def randomForestRegressorModel(self):
        randomForestRegressorModelFit = \
            RandomForestRegressor(labelCol=self.labelColm,
                                  featuresCol=self.featuresColm,
                                  numTrees=10,predictionCol=self.modelSheetName)
        regressor = randomForestRegressorModelFit.fit(self.trainData)
        # predictionData = regressor.transform(self.testData)

        regressionStat = self.randomGradientRegressionModelEvaluation(regressor=regressor)

        # persisting model
        modelName = "randomForestModel"
        extention = ".parquet"
        modelStorageLocation = self.locationAddress + self.userId.upper() + modelName.upper() + extention
        regressor.write().overwrite().save(modelStorageLocation)

        regressionStat["modelPersistLocation"] = {"modelName": modelName,
                                                  "modelStorageLocation": modelStorageLocation}

        return regressionStat

    def gradientBoostRegressorModel(self):
        gradientBoostRegressorModelFit = \
            GBTRegressor(labelCol=self.labelColm,
                         featuresCol=self.featuresColm,
                         predictionCol=self.modelSheetName)
        regressor = gradientBoostRegressorModelFit.fit(self.trainData)
        # predictionData = regressor.transform(self.testData)

        regressionStat = self.randomGradientRegressionModelEvaluation(regressor=regressor)

        # persisting model
        modelName = "gradientBoostModel"
        extention = ".parquet"
        modelStorageLocation = self.locationAddress + self.userId.upper() + modelName.upper() + extention
        regressor.write().overwrite().save(modelStorageLocation)

        regressionStat["modelPersistLocation"] = {"modelName": modelName,
                                                  "modelStorageLocation": modelStorageLocation}

        return regressionStat



        # reference for the future development.
        """
        randomForestModelFit = randomForestModel.fit(dataset)
        predictionData = randomForestModelFit.transform(dataset)
        numericalFeatures.append("prediction")
        predictionData.select(numericalFeatures).show()
        from pyspark.ml.evaluation import RegressionEvaluator
        metricsList = ['r2', 'rmse', 'mse', 'mae']
        testDataMetrics = {}
        for metric in metricsList:
            evaluator = RegressionEvaluator(labelCol=label, predictionCol="prediction", metricName=metric)
            metricValue = evaluator.evaluate(predictionData)
            testDataMetrics[metric] = metricValue
        print('testDataMetrics :', testDataMetrics)
        
        from pyspark.ml.regression import GBTRegressor
        gradientBoostingmodel = GBTRegressor(labelCol=label, featuresCol='features', maxIter=10)
        gradientBoostFittingTrainingData = gradientBoostingmodel.fit(dataset)
        gBPredictionData = gradientBoostFittingTrainingData.transform(dataset)
        numericalFeatures.append(label)
        gBPredictionData.select(numericalFeatures).show()
        metricsList = ['r2', 'rmse', 'mse', 'mae']
        testDataMetrics = {}
        for metric in metricsList:
            evaluator = RegressionEvaluator(labelCol=label, predictionCol="prediction", metricName=metric)
            metricValue = evaluator.evaluate(gBPredictionData)
            testDataMetrics[metric] = metricValue
        print('testDataMetrics :', testDataMetrics)
        
        predictionData.select(numericalFeatures).write.csv("/home/fidel/Downloads/randomForestPredictionArushiDataFinal.csv",header=True)
        
        gBPredictionData.select(numericalFeatures).write.csv("/home/fidel/Downloads/gradientBoostingArushiDataset.csv",header=True)

        """

        # return metric
