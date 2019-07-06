import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import *

spark = SparkSession.builder.appName("predictive_Analysis").master("local[*]").getOrCreate()
# pyspark --py-files /home/fidel/Downloads/xgboost4j-0.72.jar
spark.sparkContext.setLogLevel("ERROR")
spark.sparkContext.addPyFile('/home/fidel/Downloads/sparkxgb.zip')
# spark.sparkContext.addPyFile('/home/fidel/Downloads/xgboost4j-0.72.jar')
# spark.sparkContext.addPyFile('/home/fidel/Downloads/xgboost4j-spark-0.72.jar')




class Lasso_reg():
    def __init__(self, xt=[0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 0.005, 0.8, 0.3]):
        self.xt = xt

    def lasso(self, data_add):

        Rsqr_list = []
        Rsqr_regPara = {}
        print(self.xt)
        print(data_add)



if __name__=="__main__":
    Lasso_reg().lasso('l')


datasetTest = spark.read.csv(dataset_add, sep=';', header=True, inferSchema=True)
datasetTest.show()
label = ''
for val in label_colm:
    label = val
Schema = datasetTest.schema
stringFeatures = []
numericalFeatures = []
for x in Schema:
    if (str(x.dataType) == "StringType"):
        for y in feature_colm:
            if x.name == y:
                stringFeatures.append(x.name)
    else:
        for y in feature_colm:
            if x.name == y:
                numericalFeatures.append(x.name)
if relation=='linear':
    print('linear relationship')
if relation=='non_linear':
    datasetTest = Relationship(datasetTest, relation_list)
datasetTest.show()
indexed_features = []
for colm in stringFeatures:
    indexer = StringIndexer(inputCol=colm, outputCol='indexed_' + colm).fit(datasetTest)
    indexed_features.append('indexed_' + colm)
    datasetTest = indexer.transform(datasetTest)
final_features = numericalFeatures + indexed_features
featureassembler = VectorAssembler(inputCols=final_features,
                                   outputCol="features")
datasetTest= featureassembler.transform(datasetTest)
vectorIndexer = VectorIndexer(inputCol= 'features', outputCol='vectorIndexedFeatures',maxCategories=10).fit(datasetTest)
datasetTest = vectorIndexer.transform(datasetTest)
datasetTest.show()
trainDataRatioTransformed = self.trainDataRatio
testDataRatio = 1 - trainDataRatioTransformed
trainingData, testData = datasetTest.randomSplit([trainDataRatioTransformed, testDataRatio], seed=0)
gradientBoostingmodel= GBTRegressor(labelCol=label, featuresCol='vectorIndexedFeatures', maxIter=10)
gradientBoostFittingTrainingData = gradientBoostingmodel.fit(trainingData)
gradientBoostTransformTestData = gradientBoostFittingTrainingData.transform(testData)

evaluator = RegressionEvaluator(
    labelCol=label, predictionCol="prediction", metricName="r2")
rSquare = evaluator.evaluate(gradientBoostTransformTestData)
print("Root Mean Squared Error (RSquare) on test data = %g" % rSquare)
# gbtModel = gradientBoostFittingTrainingData.stages
featureImportance = gradientBoostFittingTrainingData.featureImportances.toArray().tolist()
print(featureImportance)
print(gbtModel)
