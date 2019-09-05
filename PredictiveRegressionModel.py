from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.sql import SparkSession
from pyspark.sql.types import *

from PredictionAlgorithms.PredictiveUtilities import PredictiveUtilities

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


class PredictiveRegressionModel():
    def __init__(self, trainDataRatio,dataset_add, feature_colm, label_colm, relation_list,
                        relation, userId, locationAddress, algoName):
        self.trainDataRatio = trainDataRatio
        self.datasetAdd = dataset_add
        self.featuresColmList = feature_colm
        self.labelColmList = label_colm
        self.relationshipList = relation_list
        self.relation = relation
        self.userId = userId
        self.locationAddress = locationAddress
        self.algoName = algoName

        #only for etlpart of the dataset
        self.predictiveUtilitiesObj = PredictiveUtilities()

        ETLOnDatasetStats = \
            self.predictiveUtilitiesObj.ETLOnDataset(datasetAdd=self.datasetAdd,
                                                     featuresColmList=self.featuresColmList,
                                                     labelColmList=self.labelColmList,
                                                     relationshipList=self.relationshipList,
                                                     relation=self.relation,
                                                     trainDataRatio=self.trainDataRatio,
                                                     spark=spark)
        self.dataset = ETLOnDatasetStats.get("dataset")
        self.featuresColm=ETLOnDatasetStats.get("featuresColm")
        self.labelColm=ETLOnDatasetStats.get("labelColm")
        self.trainData=ETLOnDatasetStats.get("trainData")
        self.testData=ETLOnDatasetStats.get("testData")
        self.idNameFeaturesOrdered=ETLOnDatasetStats.get("idNameFeaturesOrdered")



    def regressionModelStat(self,regressor):

        #getting stats for regressor model (ridge,lasso and linear)
        import builtins
        round = getattr(builtins, 'round')


        try:
            coefficientStdErrorList = regressor.summary.coefficientStandardErrors
            coefficientStdErrorDict = {}
            for coeffErr in coefficientStdErrorList:
                coefficientStdErrorDict[coefficientStdErrorList.index(coeffErr)] = round(coeffErr, 4)
            coefficientStdErrorDictWithName = \
                self.predictiveUtilitiesObj.summaryTable(featuresName=self.idNameFeaturesOrdered,
                                                    featuresStat=coefficientStdErrorDict)

            pValuesList = regressor.summary.pValues
            pValuesDict = {}
            for pVal in pValuesList:
                pValuesDict[pValuesList.index(pVal)] = round(pVal, 4)
            pValuesDictWithName = \
                self.predictiveUtilitiesObj.summaryTable(featuresName=self.idNameFeaturesOrdered,
                                                    featuresStat=pValuesDict)

            tValuesList = regressor.summary.tValues
            tValuesDict = {}
            for tVal in tValuesList:
                tValuesDict[tValuesList.index(tVal)] = round(tVal, 4)
            tValuesDictWithName = \
                self.predictiveUtilitiesObj.summaryTable(featuresName=self.idNameFeaturesOrdered,
                                                    featuresStat=tValuesDict)

            significanceDict = {}
            for pVal in pValuesDict.values():
                if (0 <= pVal < 0.001):
                    significanceDict[list(pValuesDict.values()).index(pVal)] = '***'
                if (0.001 <= pVal < 0.01):
                    significanceDict[list(pValuesDict.values()).index(pVal)] = '**'
                if (0.01 <= pVal < 0.05):
                    significanceDict[list(pValuesDict.values()).index(pVal)] = '*'
                if (0.05 <= pVal < 0.1):
                    significanceDict[list(pValuesDict.values()).index(pVal)] = '.'
                if (0.1 <= pVal < 1):
                    significanceDict[list(pValuesDict.values()).index(pVal)] = '-'
            significanceDictWithName = \
                self.predictiveUtilitiesObj.summaryTable(featuresName=self.idNameFeaturesOrdered,
                                                    featuresStat=significanceDict)
        except:
            coefficientStdErrorDictWithName={}
            pValuesDictWithName={}
            tValuesDictWithName={}
            significanceDictWithName={}

        coefficientList = list(map(float, list(regressor.coefficients)))
        coefficientDict = {}
        for coeff in coefficientList:
            coefficientDict[coefficientList.index(coeff)] = round(coeff, 4)
        coefficientDictWithName = \
            self.predictiveUtilitiesObj.summaryTable(featuresName=self.idNameFeaturesOrdered,
                                                     featuresStat=coefficientDict)

        #creating the table chart data
        summaryTableChartList = []
        for (keyOne, valueOne), valueTwo, valueThree, valueFour,valueFive in \
                zip(coefficientStdErrorDictWithName.items(),coefficientDictWithName.values(), pValuesDictWithName.values(),
                    tValuesDictWithName.values(), significanceDictWithName.values()):
            chartList = [keyOne, valueOne, valueTwo, valueThree, valueFour,valueFive]
            summaryTableChartList.append(chartList)
        schemaSummaryTable = StructType([StructField("Column_Name", StringType(), True),
                                         StructField("std_Error", DoubleType(), True),
                                         StructField("coefficient", DoubleType(), True),
                                         StructField("P_value", DoubleType(), True),
                                         StructField("T_value", DoubleType(), True),
                                         StructField("significance", StringType(), True)])
        summaryTableChartData = spark.createDataFrame(summaryTableChartList, schema=schemaSummaryTable)
        summaryTableChartDataFileName = \
            self.predictiveUtilitiesObj.writeToParquet(fileName="summaryTableChart",
                                                       locationAddress=self.locationAddress,
                                                       userId=self.userId,
                                                       data=summaryTableChartData)



        #creating the equation for the regression model
        intercept = round(regressor.intercept, 4)
        equation = self.labelColm, "=", intercept, "+"
        for feature, coeff in zip(self.idNameFeaturesOrdered.values(), coefficientDict.values()):
            coeffFeature = coeff, "*", feature, "+"
            equation += coeffFeature
        equation = list(equation[:-1])

        #training summary
        trainingSummary=regressor.summary
        RMSE=round(trainingSummary.rootMeanSquaredError,4)
        MAE = round(trainingSummary.meanAbsoluteError,4)
        MSE=round(trainingSummary.meanSquaredError,4)
        rSquare=round(trainingSummary.r2,4)
        adjustedRSquare=round(trainingSummary.r2adj,4)
        degreeOfFreedom = trainingSummary.degreesOfFreedom
        explainedVariance = round(trainingSummary.explainedVariance,4)
        totalNumberOfFeatures = regressor.numFeatures
        residualsTraining = trainingSummary.residuals  # sparkDataframe


        #test and training data predicted vs actual graphdata
        trainingPredictionActual = \
            trainingSummary.predictions.select(self.labelColm, "prediction")
        trainingPredictionActualGraphFileName = \
            self.predictiveUtilitiesObj.writeToParquet(fileName="trainingPredictedVsActual",
                                                       locationAddress=self.locationAddress,
                                                       userId=self.userId,
                                                       data=trainingPredictionActual)

        testPredictionActual = \
            regressor.transform(self.testData).select(self.labelColm, "prediction")
        testPredictionActualGraphFileName = \
            self.predictiveUtilitiesObj.writeToParquet(fileName="testPredictedVsActual",
                                                       locationAddress=self.locationAddress,
                                                       userId=self.userId,
                                                       data=testPredictionActual)

        #residual vs fitted graph

        residualsPredictiveDataTraining= \
            self.predictiveUtilitiesObj.residualsFittedGraph(residualsData=residualsTraining,
                                                             predictionData=trainingPredictionActual)
        residualsVsFittedGraphFileName = \
            self.predictiveUtilitiesObj.writeToParquet(fileName="residualsVsFitted",
                                                       locationAddress=self.locationAddress,
                                                       userId=self.userId,
                                                       data=residualsPredictiveDataTraining)
        #scale location plot
        sqrtStdResiduals = \
            self.predictiveUtilitiesObj.scaleLocationGraph(label=self.labelColm,
                                                      predictionTargetData=trainingPredictionActual,
                                                      residualsData=residualsTraining)
        scaleLocationGraphFileName= \
            self.predictiveUtilitiesObj.writeToParquet(fileName="scaleLocation",
                                              locationAddress=self.locationAddress,
                                              userId=self.userId,
                                              data=sqrtStdResiduals)
        #quantile plot
        quantileQuantileData = \
            self.predictiveUtilitiesObj.quantileQuantileGraph(residualsData=residualsTraining,
                                                              spark=spark)

        quantileQuantileGraphFileName = \
            self.predictiveUtilitiesObj.writeToParquet(fileName="quantileQuantile",
                                              locationAddress=self.locationAddress,
                                              userId=self.userId,
                                              data=quantileQuantileData)

        # creating dictionary for the graph data and summary stats
        graphNameDict = {"residualsVsFittedGraphFileName":residualsVsFittedGraphFileName,
                         "scaleLocationGraphFileName":scaleLocationGraphFileName,
                         "quantileQuantileGraphFileName":quantileQuantileGraphFileName,
                         "trainingPredictionActualGraphFileName":trainingPredictionActualGraphFileName,
                         "testPredictionActualGraphFileName":testPredictionActualGraphFileName}
        summaryStats = {"RMSE":RMSE,"MSE":MSE,
                        "MAE":MAE,"rSquare":rSquare,
                        "adjustedRSquare":adjustedRSquare,
                        "intercept":intercept,
                        "degreeOfFreedom":degreeOfFreedom,
                        "explainedVariance":explainedVariance,
                        "totalNumberOfFeatures":totalNumberOfFeatures}

        summaryTable = {"summaryTableChartDataFileName":summaryTableChartDataFileName}



        response = {"graphData":graphNameDict,
                    "statData":summaryStats,
                    "tableData":summaryTable,
                    "equation":equation}

        return response

    def linearModel(self):
        linearRegressionModelfit = \
            LinearRegression(featuresCol=self.featuresColm, labelCol=self.labelColm)
        regressor = linearRegressionModelfit.fit(self.trainData)
        regressionStat=self.regressionModelStat(regressor=regressor)

        #persisting the model
        modelName="linearRegressionModel"
        extention = ".parquet"
        modelStorageLocation=self.locationAddress + self.userId.upper() + modelName.upper() + extention
        regressor.write().overwrite().save(modelStorageLocation)

        regressionStat["modelPersistLocation"] = {"modelName":modelName,
                                                  "modelStorageLocation":modelStorageLocation}


        return regressionStat

    def ridgeLassoModel(self,regParam):
        regParam = 0.05 if regParam==None else float(regParam)
        elasticNetPara=1 if self.algoName=="lasso_reg" else 0
        ridgeLassoModelFit = \
            LinearRegression(featuresCol=self.featuresColm,
                                  labelCol=self.labelColm,
                                  elasticNetParam=elasticNetPara,
                                   regParam=regParam)
        regressor = ridgeLassoModelFit.fit(self.trainData)
        regressionStat = self.regressionModelStat(regressor=regressor)

        #persisting model
        modelName = "lassoRegressionModel" if self.algoName=="lasso_reg" \
            else "ridgeRegressionModel"
        extention = ".parquet"
        modelStorageLocation = self.locationAddress + self.userId.upper() + modelName.upper() + extention
        regressor.write().overwrite().save(modelStorageLocation)

        regressionStat["modelPersistLocation"] = {"modelName": modelName,
                                                  "modelStorageLocation": modelStorageLocation}

        return regressionStat

    #for future
    def randomForestRegressorModel(self):
        randomForestRegressorModelFit = \
            RandomForestRegressor(labelCol=self.labelColm,
                                  featuresCol=self.featuresColm,
                                  numTrees=10)
        regressor = randomForestRegressorModelFit.fit(self.trainData)









