from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import json


from PredictionAlgorithms.relationship import Relationship

spark = SparkSession.builder.appName('predictive_Analysis').master('local[*]').getOrCreate()
spark.sparkContext.setLogLevel('ERROR')
'''
trainDataRation = ration of training data, take input from the user
learningRate = learning rate applied on the model, take input from the user
dataset_add = dataset address, from the user
feature_colm = column as a features is taken from the user
relation_list = relationship list of each column needed to be applied
relation = whether it is linear relation or non linear taken from the user end
'''

class GradientBoostClassification():
    def __init__(self, trainDataRatio=0.80, learningRate=0.1):
        self.trainDataRatio = trainDataRatio
        self.learningRate = learningRate

    def GradientBoostingClassification(self, dataset_add, feature_colm, label_colm, relation_list, relation):
        try:
            dataset = spark.read.csv(dataset_add, sep=';',header=True, inferSchema=True)
            dataset.show()
            stepSize=self.learningRate
            label = ''
            for val in label_colm:
                label = val
            #ETL part
            Schema = dataset.schema
            stringFeatures = []
            numericalFeatures = []
            for x in Schema:
                if (str(x.dataType) == "StringType" or str(x.dataType) == 'TimestampType' or str(
                        x.dataType) == 'DateType' or str(x.dataType) == 'BooleanType' or str(x.dataType) == 'BinaryType'):
                    for y in feature_colm:
                        if x.name == y:
                            dataset = dataset.withColumn(y, dataset[y].cast(StringType()))
                            stringFeatures.append(x.name)
                else:
                    for y in feature_colm:
                        if x.name == y:
                            numericalFeatures.append(x.name)

            if relation == 'linear':
                dataset = dataset
            if relation == 'non_linear':
                dataset = Relationship(dataset, relation_list)


            categoryColmList = []
            categoryColmListFinal = []
            categoryColmListDict = {}
            countOfCategoricalColmList = []
            for value in stringFeatures:
                categoryColm = value
                listValue = value
                listValue = []
                categoryColm = dataset.groupby(value).count()
                countOfCategoricalColmList.append(categoryColm.count())
                categoryColmJson = categoryColm.toJSON()
                for row in categoryColmJson.collect():
                    categoryColmSummary = json.loads(row)
                    listValue.append(categoryColmSummary)
                categoryColmListDict[value] = listValue

            if not stringFeatures:
                maxCategories = 5
            else:
                maxCategories = max(countOfCategoricalColmList)
            for x in Schema:
                if (str(x.dataType) == "StringType" and x.name == label):
                    for labelkey in label_colm:
                        label_indexer = StringIndexer(inputCol=label, outputCol='indexed_' + label).fit(dataset)
                        dataset = label_indexer.transform(dataset)
                        label = 'indexed_' + label
                else:
                    label = label
            indexed_features = []
            for colm in stringFeatures:
                indexer = StringIndexer(inputCol=colm, outputCol='indexed_' + colm).fit(dataset)
                indexed_features.append('indexed_' + colm)
                dataset = indexer.transform(dataset)
            final_features = numericalFeatures + indexed_features
            featureassembler = VectorAssembler(inputCols=final_features,
                                               outputCol="features")
            dataset = featureassembler.transform(dataset)
            vectorIndexer = VectorIndexer(inputCol='features', outputCol='vectorIndexedFeatures', maxCategories=maxCategories).fit(dataset)
            dataset = vectorIndexer.transform(dataset)
            trainDataRatioTransformed = self.trainDataRatio
            testDataRatio = 1 - trainDataRatioTransformed
            trainingData, testData = dataset.randomSplit([trainDataRatioTransformed, testDataRatio], seed=0)

            gradientBoostingmodel = GBTClassifier(labelCol=label, featuresCol='vectorIndexedFeatures', maxIter=10,stepSize=stepSize)
            gradientBoostFittingTrainingData = gradientBoostingmodel.fit(trainingData)
            gBPredictionTrainData = gradientBoostFittingTrainingData.transform(trainingData)
            gBPredictionTestData = gradientBoostFittingTrainingData.transform(testData)
            gBPredictionTestData.select('prediction', label).show()
            # gbtModel = gradientBoostFittingTrainingData.stages
            featureImportance = gradientBoostFittingTrainingData.featureImportances.toArray().tolist()
            print(featureImportance)


            # prediction graph data
            from pyspark.sql.functions import col
            TrainPredictedTargetData = gBPredictionTrainData.select(label, 'prediction', 'probability','rawPrediction')
            residualsTrainData = TrainPredictedTargetData.withColumn('residuals', col(label) - col('prediction'))
            residualsTrainData.show()

            TestPredictedTargetData = gBPredictionTestData.select(label, 'prediction', 'probability','rawPrediction')
            residualsTestData = TestPredictedTargetData.withColumn('residuals', col(label) - col('prediction'))
            residualsTestData.show()

            # train Test data Metrics
            gBPredictionDataDict = {'gBPredictionTestData': gBPredictionTestData,
                                    'gBPredictionTrainData': gBPredictionTrainData}
            metricsList = ['f1','weightedPrecision','weightedRecall','accuracy']
            for key, value in gBPredictionDataDict.items():
                if key == 'gBPredictionTestData':
                    testDataMetrics = {}
                    for metric in metricsList:
                        evaluator = MulticlassClassificationEvaluator(labelCol=label, predictionCol="prediction", metricName=metric)
                        metricValue = evaluator.evaluate(gBPredictionTestData)
                        testDataMetrics[metric] = metricValue
                    print('testDataMetrics :', testDataMetrics)

                if key == 'gBPredictionTrainData':
                    trainDataMetrics = {}
                    for metric in metricsList:
                        evaluator = MulticlassClassificationEvaluator(labelCol=label, predictionCol="prediction", metricName=metric)
                        metricValue = evaluator.evaluate(gBPredictionTrainData)
                        trainDataMetrics[metric] = metricValue
                    print('trainDataMetrics :', trainDataMetrics)

            # while fitting the training data
            totalNumberTrees = gradientBoostFittingTrainingData.getNumTrees
            print('Total number of trees used is :', totalNumberTrees)
            totalNumberNodes = gradientBoostFittingTrainingData.totalNumNodes
            print('Total number of node is :', totalNumberNodes)
            treeWeight = gradientBoostFittingTrainingData.treeWeights
            print('Weights on each tree is :', treeWeight)
            treeInfo = gradientBoostFittingTrainingData.trees
            for eachTree in treeInfo:
                print('info of each tree is :', eachTree)






        except Exception as e:
            print('exception is --', e)