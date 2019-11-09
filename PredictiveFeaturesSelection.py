from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql import SparkSession

from PredictionAlgorithms.PredictiveDataTransformation import PredictiveDataTransformation
from PredictionAlgorithms.PredictiveStatisticalTest import PredictiveStatisticalTest
from PredictionAlgorithms.PredictiveUtilities import PredictiveUtilities
from PredictionAlgorithms.PredictiveConstants import PredictiveConstants


class PredictiveFeaturesSelection:
    def __init__(self,spark):
        self.spark = spark

    def featuresSelection(self, dataset_add, feature_colm,
                          label_colm, relation_list, relation, userId, algoName,
                          locationAddress):
        dataset = self.spark.read.parquet(dataset_add)
        # PredictiveUtilities = PredictiveUtilities()

        # changing the relationship of the colm(log,squareroot,exponential)
        dataTransformationObj = PredictiveDataTransformation(dataset=dataset)
        dataset = dataTransformationObj.colmTransformation(colmTransformationList=relation_list) \
            if relation == PredictiveConstants.NON_LINEAR  else dataset
        # transformation
        dataTransformationObj = PredictiveDataTransformation(dataset=dataset)
        dataTransformationResult = dataTransformationObj.dataTranform(labelColm=label_colm,
                                                                      featuresColm=feature_colm,
                                                                      userId=userId)
        dataset = dataTransformationResult.get(PredictiveConstants.DATASET)
        categoricalFeatures = dataTransformationResult.get(PredictiveConstants.CATEGORICALFEATURES)
        numericalFeatures = dataTransformationResult.get(PredictiveConstants.NUMERICALFEATURES)
        maxCategories = dataTransformationResult.get(PredictiveConstants.MAXCATEGORIES)
        categoryColmStats = dataTransformationResult.get(PredictiveConstants.CATEGORYCOLMSTATS)
        indexedFeatures = dataTransformationResult.get(PredictiveConstants.INDEXEDFEATURES)
        label = dataTransformationResult.get(PredictiveConstants.LABEL)
        idNameFeaturesOrdered = dataTransformationResult.get(PredictiveConstants.IDNAMEFEATURESORDERED)
        oneHotEncodedFeaturesList = dataTransformationResult.get(PredictiveConstants.ONEHOTENCODEDFEATURESLIST)
        indexedLabelNameDict = dataTransformationResult.get(PredictiveConstants.INDEXEDLABELNAMEDICT)
        featuresColm = dataTransformationResult.get(PredictiveConstants.VECTORFEATURES)

        # statistics
        columnListForfeaturesStats = numericalFeatures.copy()
        columnListForfeaturesStats.insert(0, label)
        dataTransformationObj = PredictiveDataTransformation(dataset=dataset)
        dataStatsResult = \
            dataTransformationObj.dataStatistics(categoricalFeatures=categoricalFeatures,
                                                 numericalFeatures=columnListForfeaturesStats,
                                                 categoricalColmStat=categoryColmStats)
        summaryDict = dataStatsResult

        # creating the dataset for statschart visualization in features selection chart
        datasetForStatsChart = dataset.select(columnListForfeaturesStats)
        datasetForStatsChartFileName = \
            PredictiveUtilities.writeToParquet(fileName="datasetForStatsChart",
                                                  locationAddress=locationAddress,
                                                  userId=userId,
                                                  data=datasetForStatsChart)

        featuresStatsDict = {"columnsName": columnListForfeaturesStats,
                             "datasetFileName": datasetForStatsChartFileName}

        # applying the algorithm
        ##calling the pearson test
        trainData, testData = dataset.randomSplit([0.80, 0.20], seed=40)

        keyStatsTest = ''
        statisticalTestResult = {}
        if algoName == PredictiveConstants.RANDOMREGRESSOR:
            statisticalTestObj = PredictiveStatisticalTest(dataset=dataset,
                                                           features=numericalFeatures,
                                                           labelColm=label)
            statisticalTestResult = statisticalTestObj.pearsonTest()
            randomForestModel = \
                RandomForestRegressor(labelCol=label,
                                      featuresCol=featuresColm,
                                      numTrees=10)
            keyStatsTest = "pearson_test_data"
        if algoName == PredictiveConstants.RANDOMCLASSIFIER:
            statisticalTestObj = PredictiveStatisticalTest(dataset=dataset,
                                                           features=indexedFeatures,
                                                           labelColm=label)
            statisticalTestResult = \
                statisticalTestObj.chiSquareTest(categoricalFeatures=categoricalFeatures,
                                                 maxCategories=maxCategories)
            randomForestModel = RandomForestClassifier(labelCol=label,
                                                       featuresCol=featuresColm,
                                                       numTrees=10)
            keyStatsTest = "ChiSquareTestData"
        randomForestModelFit = randomForestModel.fit(trainData)
        # predictions = randomForestModelFit.transform(testData)
        print(randomForestModelFit.featureImportances)
        # feature_importance = randomForestModelFit.featureImportances.toArray().tolist()
        # print(feature_importance)
        import pyspark.sql.functions as F
        import builtins
        round = getattr(builtins, 'round')

        featuresImportance = list(randomForestModelFit.featureImportances)
        featuresImportance = [round(x, 4) for x in featuresImportance]
        featuresImportanceDict = {}
        for importance in featuresImportance:
            featuresImportanceDict[featuresImportance.index(importance)] = round(importance, 4)

        featuresImportanceDictWithName = \
            PredictiveUtilities.summaryTable(featuresName=idNameFeaturesOrdered,
                                                featuresStat=featuresImportanceDict)

        # feature_importance = randomForestModelFit.featureImportances.toArray().tolist()
        # print(feature_importance)
        # featureImportance = []
        # for x in feature_importance:
        #     featureImportance.append(round(x, 4))
        # features_column_for_user = numericalFeatures + categoricalFeatures
        featuresColmList = idNameFeaturesOrdered
        feat = []
        for val in featuresColmList.values():
            feat.append(val)
        feature_imp = {PredictiveConstants.FEATURE_IMPORTANCE: featuresImportance, "feature_column": feat}

        response_dict = {
            PredictiveConstants.FEATURE_IMPORTANCE: feature_imp,
            keyStatsTest: statisticalTestResult,
            'summaryDict': summaryDict,
            'categoricalSummary': categoryColmStats,
            "featuresImportanceDict": featuresImportanceDictWithName,
            "featuresStatsDict": featuresStatsDict
        }
        return response_dict

