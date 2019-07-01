import csv
from pyspark.ml.regression import LinearRegressionModel


import pyspark.sql.functions as f
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer
from PredictionAlgorithms.LinearRegressionTest import LinearRegressionModel

from pyspark.sql.types import *

from PredictionAlgorithms.relationship import Relationship

spark = SparkSession.builder.appName("predictive_Analysis").master("local[*]").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")



class LinearRegressionPersistModel():
    def __init__(self, trainDataRatio=0.80):
        self.trainDataRatio = trainDataRatio


    def linearRegPersist(self, dataset_add, feature_colm, label_colm, relation_list, relation,userId):
        try:
            dataset = spark.read.csv(dataset_add, header=True, inferSchema=True)
            dataset.show()
            label = ''
            for val in label_colm:
                label = val
            Schema = dataset.schema
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
            if relation == 'linear':
                print('linear relationship')
            if relation == 'non_linear':
                dataset = Relationship(dataset, relation_list)
            dataset.show()
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
            vectorIndexer = VectorIndexer(inputCol='features', outputCol='vectorIndexedFeatures', maxCategories=4).fit(
                dataset)
            dataset = vectorIndexer.transform(dataset)
            # Loading the persisted model
            locationAddress = 'hdfs://10.171.0.181:9000/dev/dmxdeepinsight/datasets/'
            modelPersist = 'linearRegressorModel.parquet'
            persistedModelLocation = locationAddress + userId + modelPersist
            regressorTest = LinearRegressionModel.load(persistedModelLocation)
            predictedData = regressorTest.transform(dataset)

            predictedData.show()

        except Exception as e:
            print('exception is :' , e)