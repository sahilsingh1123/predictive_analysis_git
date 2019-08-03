from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import json
spark = SparkSession.builder.appName('predictive_Analysis').master('local[*]').getOrCreate()
spark.sparkContext.setLogLevel('ERROR')
class CallCenter():
    def __init__(self):
        pass
    def callCenter(self):
        dataset = spark.read.csv("/home/fidel/Downloads/CallCenterFinalTillAprilData", sep=',', header=True, inferSchema=True)
        dataset.show()
        feature_colm=["col_2_SKILLNAME_2","col_2_SKILLNAME_3"]
        label_colm=["CALLDATE"]
        label=""
        for val in label_colm:
            label = val
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

        categoryColmList = []
        categoryColmListFinal = []
        categoryColmListDict = {}
        countOfCategoricalColmList = []
        for value in stringFeatures:
            categoryColm = value
            listValue = value
            listValue = []
            categoryColm = dataset.groupby(value).count()
            print(categoryColm)
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
        maxCategories=13

        for x in Schema:
            if (str(x.dataType) == "StringType" and x.name == label):
                for labelkey in label_colm:
                    label_indexer = StringIndexer(inputCol=label, outputCol='indexed_' + label).fit(dataset)
                    dataset = label_indexer.transform(dataset)
                    label = 'indexed_' + label
            else:
                label = label
        dataset.show()
        indexed_features = []
        for colm in stringFeatures:
            indexer = StringIndexer(inputCol=colm, outputCol='indexed_' + colm).fit(dataset)
            indexed_features.append('indexed_' + colm)
            dataset = indexer.transform(dataset)
        final_features = numericalFeatures + indexed_features
        featureassembler = VectorAssembler(inputCols=final_features,
                                           outputCol="features")
        dataset = featureassembler.transform(dataset)
        vectorIndexer = VectorIndexer(inputCol='features', outputCol='vectorIndexedFeatures',
                                      maxCategories=maxCategories).fit(dataset)
        dataset = vectorIndexer.transform(dataset)
        import csv
        dataset = dataset.select("CALLDATE", "col_2_SKILLNAME_2", "col_2_SKILLNAME_3", "indexed_CALLDATE",
                                 "indexed_col_2_SKILLNAME_2", "indexed_col_2_SKILLNAME_3")
        # dataset.to_csv("/home/fidel/Downloads/Callcenterdata/callFinalFormated.csv")
        # dataset.write.csv("/home/fidel/Downloads/Callcenterdata/callFinalF.csv")
        # dataset.write.csv("/home/fidel/Downloads/Callcenterdata/callFinal.csv")
        # dataset.show()
        dataset.toPandas().to_csv("/home/fidel/Downloads/Callcenterdata/callcsv.csv")
        trainDataRatioTransformed = 0.80
        testDataRatio = 1 - trainDataRatioTransformed
        trainingData, testData = dataset.randomSplit([trainDataRatioTransformed, testDataRatio], seed=0)
        #applying the model
        randomForestModel = RandomForestClassifier(labelCol=label, featuresCol='vectorIndexedFeatures', numTrees=10,
                                                   maxBins=maxCategories)
        randomForestModelFit = randomForestModel.fit(trainingData)
        predictions = randomForestModelFit.transform(testData)

        # Select example rows to display.
        predictions.select("predictedLabel", "label", "features").show(5)
        evaluator = MulticlassClassificationEvaluator(
            labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)
        print("Test Error = %g" % (1.0 - accuracy))

classObj = CallCenter()
classObj.callCenter()