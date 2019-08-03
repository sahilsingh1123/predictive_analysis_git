from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql import SparkSession
from PredictionAlgorithms.dataTranformation import DataTransformation
from PredictionAlgorithms.StatisticalTest import StatisticalTest

spark = SparkSession.builder.appName("predictive analysis").master("local[*]").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
class FeaturesSelection:
    def __init__(self):
        pass
    def featuresSelection(self,dataset_add,feature_colm,label_colm,relation_list,relation,userId,algoName):
        try:
            dataset=spark.read.parquet(dataset_add)

            #changing the relationship of the colm
            dataTransformationObj = DataTransformation(dataset=dataset)
            dataset=dataTransformationObj.colmTransformation(colmTransformationList=relation_list) if relation=="non_linear" else dataset
            #transformation
            dataTransformationObj = DataTransformation(dataset=dataset)
            dataTransformationResult=dataTransformationObj.dataTranform(labelColm=label_colm,featuresColm=feature_colm)
            dataset = dataTransformationResult["dataset"]
            categoricalFeatures = dataTransformationResult["categoricalFeatures"]
            numericalFeatures = dataTransformationResult["numericalFeatures"]
            maxCategories = dataTransformationResult["maxCategories"]
            categoryColmStats=dataTransformationResult["categoryColmStats"]
            indexedFeatures=dataTransformationResult["indexedFeatures"]
            label=dataTransformationResult["label"]
            #statistics
            dataTransformationObj = DataTransformation(dataset=dataset)
            dataStatsResult=dataTransformationObj.dataStatistics(categoricalFeatures=categoricalFeatures,numericalFeatures=numericalFeatures)
            summaryDict=dataStatsResult

            # applying the algorithm
            ##calling the pearson test
            trainData,testData=dataset.randomSplit([0.80,0.20],seed=40)

            if algoName=="random_regressor":
                statisticalTestObj=StatisticalTest(dataset=dataset,features=numericalFeatures,labelColm=label)
                statisticalTestResult=statisticalTestObj.pearsonTest()
                randomForestModel = RandomForestRegressor(labelCol=label, featuresCol='vec_indexed_features', numTrees=10,
                                           maxBins=maxCategories)
                keyStatsTest = "pearson_test_data"
            if algoName=="random_classifier":
                statisticalTestObj=StatisticalTest(dataset=dataset,features=indexedFeatures,labelColm=label)
                statisticalTestResult=statisticalTestObj.chiSquareTest(categoricalFeatures=categoricalFeatures,maxCategories=maxCategories)
                randomForestModel = RandomForestClassifier(labelCol=label, featuresCol='vec_indexed_features', numTrees=10,
                                            maxBins=maxCategories)
                keyStatsTest = "ChiSquareTestData"
            randomForestModelFit = randomForestModel.fit(trainData)
            # predictions = randomForestModelFit.transform(testData)
            print(randomForestModelFit.featureImportances)
            # feature_importance = randomForestModelFit.featureImportances.toArray().tolist()
            # print(feature_importance)
            import pyspark.sql.functions as F
            import builtins
            round = getattr(builtins, 'round')
            feature_importance = randomForestModelFit.featureImportances.toArray().tolist()
            print(feature_importance)
            featureImportance = []
            for x in feature_importance:
                featureImportance.append(round(x, 4))
            print(featureImportance)

            features_column_for_user = numericalFeatures + categoricalFeatures
            feature_imp = {'feature_importance': featureImportance, "feature_column": features_column_for_user}
            
            response_dict = {
                'feature_importance': feature_imp,
                keyStatsTest: statisticalTestResult,
                'summaryDict': summaryDict,
                'categoricalSummary': categoryColmStats
            }
            return response_dict

        except Exception as e:
            print(str(e))