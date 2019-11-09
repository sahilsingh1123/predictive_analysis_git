from pyspark.sql.types import *
from pyspark.sql.functions import col
from PredictionAlgorithms.PredictiveUtilities import PredictiveUtilities
from PredictionAlgorithms.PredictiveConstants import PredictiveConstants

# this class is inherited by predictiveRegressionModel and will
# be used by other classes also in the future development
class PredictiveEvaluation(object):

    # getting stats for regressor model (ridge,lasso and linear)
    def regressionModelEvaluation(self, regressor, spark):

        import builtins
        round = getattr(builtins, 'round')

        try:
            coefficientStdErrorList = regressor.summary.coefficientStandardErrors
            coefficientStdErrorDict = {}
            statsDictName = "coefficientStdErrorDictWithName"

            coefficientStdErrorDictWithName = self.statsDict(coefficientStdErrorList,coefficientStdErrorDict)

            pValuesList = regressor.summary.pValues
            pValuesDict = {}

            pValuesDictWithName = self.statsDict(pValuesList,pValuesDict)


            tValuesList = regressor.summary.tValues
            tValuesDict = {}

            tValuesDictWithName = self.statsDict(tValuesList,tValuesDict)

            significanceDict = {}
            for pkey, pVal in pValuesDict.items():
                if (0 <= pVal < 0.001):
                    significanceDict[pkey] = '***'
                if (0.001 <= pVal < 0.01):
                    significanceDict[pkey] = '**'
                if (0.01 <= pVal < 0.05):
                    significanceDict[pkey] = '*'
                if (0.05 <= pVal < 0.1):
                    significanceDict[pkey] = '.'
                if (0.1 <= pVal < 1):
                    significanceDict[pkey] = '-'
            significanceDictWithName = \
                PredictiveUtilities.summaryTable(featuresName=self.idNameFeaturesOrdered,
                                                         featuresStat=significanceDict)
        except:
            coefficientStdErrorDictWithName = {}
            pValuesDictWithName = {}
            tValuesDictWithName = {}
            significanceDictWithName = {}

        coefficientList = list(map(float, list(regressor.coefficients)))
        coefficientDict = {}
        coefficientDictWithName = self.statsDict(coefficientList,coefficientDict)

        # creating the table chart data
        summaryTableChartList = []
        if self.algoName != "lasso_reg":
            for (keyOne, valueOne), valueTwo, valueThree, valueFour, valueFive in \
                    zip(coefficientStdErrorDictWithName.items(), coefficientDictWithName.values(),
                        pValuesDictWithName.values(),
                        tValuesDictWithName.values(), significanceDictWithName.values()):
                chartList = [keyOne, valueOne, valueTwo, valueThree, valueFour, valueFive]
                summaryTableChartList.append(chartList)
            schemaSummaryTable = StructType([StructField("Column_Name", StringType(), True),
                                             StructField("std_Error", DoubleType(), True),
                                             StructField("coefficient", DoubleType(), True),
                                             StructField("P_value", DoubleType(), True),
                                             StructField("T_value", DoubleType(), True),
                                             StructField("significance", StringType(), True)])

        if(coefficientStdErrorDictWithName == {} or self.algoName == "lasso_reg"):
            for (keyOne, valueOne) in coefficientDictWithName.items():
                chartList = [keyOne, valueOne]
                summaryTableChartList.append(chartList)

            schemaSummaryTable = StructType([StructField("Column_Name", StringType(), True),
                                             StructField("coefficient", DoubleType(), True)])

        summaryTableChartData = spark.createDataFrame(summaryTableChartList, schema=schemaSummaryTable)
        summaryTableChartDataFileName = \
            PredictiveUtilities.writeToParquet(fileName="summaryTableChart",
                                                       locationAddress=self.locationAddress,
                                                       userId=self.userId,
                                                       data=summaryTableChartData)

        # creating the equation for the regression model
        intercept = round(regressor.intercept, 4)
        equation = self.labelColm, "=", intercept, "+"
        for feature, coeff in zip(self.idNameFeaturesOrdered.values(), coefficientDict.values()):
            coeffFeature = coeff, "*", feature, "+"
            equation += coeffFeature
        equation = list(equation[:-1])

        # training summary
        trainingSummary = regressor.summary
        RMSE = round(trainingSummary.rootMeanSquaredError, 4)
        MAE = round(trainingSummary.meanAbsoluteError, 4)
        MSE = round(trainingSummary.meanSquaredError, 4)
        rSquare = round(trainingSummary.r2, 4)
        adjustedRSquare = round(trainingSummary.r2adj, 4)
        degreeOfFreedom = trainingSummary.degreesOfFreedom
        explainedVariance = round(trainingSummary.explainedVariance, 4)
        totalNumberOfFeatures = regressor.numFeatures
        residualsTraining = trainingSummary.residuals  # sparkDataframe

        # test and training data predicted vs actual graphdata

        trainingPredictionAllColm = trainingSummary.predictions
        trainingPredictionActual = \
            trainingPredictionAllColm.select(self.labelColm, self.modelSheetName)
        trainingPredictionActualGraphFileName = \
            PredictiveUtilities.writeToParquet(fileName="trainingPredictedVsActual",
                                                       locationAddress=self.locationAddress,
                                                       userId=self.userId,
                                                       data=trainingPredictionActual)
        testPredictionAllColm = regressor.transform(self.testData)
        testPredictionActual = \
            testPredictionAllColm.select(self.labelColm, self.modelSheetName)
        testPredictionActualGraphFileName = \
            PredictiveUtilities.writeToParquet(fileName="testPredictedVsActual",
                                                       locationAddress=self.locationAddress,
                                                       userId=self.userId,
                                                       data=testPredictionActual)

        # appending train and test dataset together
        # for future use only
        trainTestMerged = trainingPredictionAllColm.union(testPredictionAllColm)
        trainTestMergedFileName = \
            PredictiveUtilities.writeToParquet(fileName="trainTestMerged",
                                                       locationAddress=self.locationAddress,
                                                       userId=self.userId,
                                                       data=trainTestMerged)

        # residual vs fitted graph

        residualsPredictiveDataTraining = \
            PredictiveUtilities.residualsFittedGraph(residualsData=residualsTraining,
                                                             predictionData=trainingPredictionActual,
                                                             modelSheetName=self.modelSheetName)
        residualsVsFittedGraphFileName = \
            PredictiveUtilities.writeToParquet(fileName="residualsVsFitted",
                                                       locationAddress=self.locationAddress,
                                                       userId=self.userId,
                                                       data=residualsPredictiveDataTraining)
        # scale location plot
        sqrtStdResiduals = \
            PredictiveUtilities.scaleLocationGraph(label=self.labelColm,
                                                           predictionTargetData=trainingPredictionActual,
                                                           residualsData=residualsTraining,
                                                           modelSheetName=self.modelSheetName)
        scaleLocationGraphFileName = \
            PredictiveUtilities.writeToParquet(fileName="scaleLocation",
                                                       locationAddress=self.locationAddress,
                                                       userId=self.userId,
                                                       data=sqrtStdResiduals)
        # quantile plot
        quantileQuantileData = \
            PredictiveUtilities.quantileQuantileGraph(residualsData=residualsTraining,
                                                              spark=spark)

        quantileQuantileGraphFileName = \
            PredictiveUtilities.writeToParquet(fileName="quantileQuantile",
                                                       locationAddress=self.locationAddress,
                                                       userId=self.userId,
                                                       data=quantileQuantileData)

        # creating dictionary for the graph data and summary stats
        graphNameDict = {PredictiveConstants.RESIDUALSVSFITTEDGRAPHFILENAME: residualsVsFittedGraphFileName,
                         PredictiveConstants.SCALELOCATIONGRAPHFILENAME: scaleLocationGraphFileName,
                         PredictiveConstants.QUANTILEQUANTILEGRAPHFILENAME: quantileQuantileGraphFileName,
                         PredictiveConstants.TRAININGPREDICTIONACTUALFILENAME: trainingPredictionActualGraphFileName,
                         PredictiveConstants.TESTPREDICTIONACTUALFILENAME: testPredictionActualGraphFileName}
        summaryStats = {PredictiveConstants.RMSE: RMSE, PredictiveConstants.MSE: MSE,
                        PredictiveConstants.MAE: MAE, PredictiveConstants.RSQUARE: rSquare,
                        PredictiveConstants.ADJRSQUARE: adjustedRSquare,
                        PredictiveConstants.INTERCEPT: intercept,
                        PredictiveConstants.DOF: degreeOfFreedom,
                        PredictiveConstants.EXPLAINEDVARIANCE: explainedVariance,
                        PredictiveConstants.TOTALFEATURES: totalNumberOfFeatures}

        summaryTable = {"summaryTableChartDataFileName": summaryTableChartDataFileName}

        response = {PredictiveConstants.GRAPHDATA: graphNameDict,
                    PredictiveConstants.STATDATA: summaryStats,
                    PredictiveConstants.TABLEDATA: summaryTable,
                    PredictiveConstants.EQUATION: equation}

        return response


    #getting stats for ensemble regression model
    def randomGradientRegressionModelEvaluation(self,regressor):
        trainPredictedData = regressor.transform(self.trainData)
        testPredictedData = regressor.transform(self.testData)
        from pyspark.ml.evaluation import RegressionEvaluator
        metricsList = ['r2', 'rmse', 'mse', 'mae']
        trainDataMetrics = {}
        metricName = ''
        for metric in metricsList:
            if metric.__eq__("r2"):
                metricName = PredictiveConstants.RSQUARE
            elif metric.__eq__("rmse"):
                metricName = PredictiveConstants.RMSE
            elif metric.__eq__("mse"):
                metricName = PredictiveConstants.MSE
            elif metric.__eq__("mae"):
                metricName = PredictiveConstants.MAE
            evaluator = RegressionEvaluator(labelCol=self.labelColm,
                                            predictionCol=self.modelSheetName,
                                            metricName=metric)
            metricValue = evaluator.evaluate(trainPredictedData)
            trainDataMetrics[metricName] = metricValue

        #training Actual vs Predicted dataset
        trainingPredictionActual = \
            trainPredictedData.select(self.labelColm, self.modelSheetName)
        trainingPredictionActualGraphFileName = \
            PredictiveUtilities.writeToParquet(fileName="trainingPredictedVsActualEnsemble",
                                               locationAddress=self.locationAddress,
                                               userId=self.userId,
                                               data=trainingPredictionActual)
        #test Actual Vs Predicted dataset
        testPredictionActual = \
            testPredictedData.select(self.labelColm, self.modelSheetName)
        testPredictionActualGraphFileName = \
            PredictiveUtilities.writeToParquet(fileName="testPredictedVsActualEnsemble",
                                               locationAddress=self.locationAddress,
                                               userId=self.userId,
                                               data=testPredictionActual)

        # summary stats
        noTrees = regressor.getNumTrees
        treeWeights = regressor.treeWeights
        treeNodes = list(regressor.trees)
        totalNoNodes = regressor.totalNumNodes
        debugString = regressor.toDebugString

        debugString = str(debugString).splitlines()

        featuresImportance = list(regressor.featureImportances)
        featuresImportance = [round(x, 4) for x in featuresImportance]
        print(featuresImportance)
        featuresImportanceDict = {}
        for importance in featuresImportance:
            featuresImportanceDict[featuresImportance.index(importance)] = importance

        featuresImportanceDictWithName = \
            PredictiveUtilities.summaryTable(featuresName=self.idNameFeaturesOrdered,
                                             featuresStat=featuresImportanceDict)

        trainDataMetrics["No Trees"] = noTrees
        trainDataMetrics["Total Nodes"] = totalNoNodes

        summaryStats = {'noTrees':noTrees,
                        'treeWeights': treeWeights,
                        'totalNodes':totalNoNodes,
                        'featuresImportance':featuresImportanceDictWithName,
                        'metrics':trainDataMetrics,
                        'debugString': debugString,
                        }


        #creating the residual vs fitted graph data
        residualDataColm = trainingPredictionActual.withColumn('residuals',
                                                               col(self.labelColm) - col(self.modelSheetName))
        residualDataColm = residualDataColm.select('residuals')
        residualsPredictiveDataTraining = \
            PredictiveUtilities.residualsFittedGraph(residualsData=residualDataColm,
                                                     predictionData=trainingPredictionActual,
                                                     modelSheetName=self.modelSheetName)
        residualsVsFittedGraphFileName = \
            PredictiveUtilities.writeToParquet(fileName="residualsVsFittedEnsemble",
                                               locationAddress=self.locationAddress,
                                               userId=self.userId,
                                               data=residualsPredictiveDataTraining)

        graphNameDict = {PredictiveConstants.RESIDUALSVSFITTEDGRAPHFILENAME: residualsVsFittedGraphFileName,
                         PredictiveConstants.TRAININGPREDICTIONACTUALFILENAME: trainingPredictionActualGraphFileName,
                         PredictiveConstants.TESTPREDICTIONACTUALFILENAME: testPredictionActualGraphFileName}

        response = {PredictiveConstants.STATDATA:summaryStats,
                    PredictiveConstants.GRAPHDATA:graphNameDict}



        return response

    # method to create key value pair for statistics
    def statsDict(self,statList, statDict):
        for index, value in enumerate(statList):
            statDict[index] = round(value, 4)

        return PredictiveUtilities.summaryTable(featuresName=self.idNameFeaturesOrdered,
                                             featuresStat=statDict)