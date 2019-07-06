from pyspark.ml.regression import LinearRegression, LinearRegressionModel
from pyspark.sql import SparkSession
from PredictionAlgorithms.relationship import Relationship
import pyspark.sql.functions as f
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pyspark.sql.types import *




spark = SparkSession.builder.appName("predictive_Analysis").master("local[*]").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")



def loadModel(dataset_add, feature_colm, label_colm, relation_list, relation):
    try:
        # dataset = spark.read.csv('/home/fidel/mltest/testData.csv', header=True, inferSchema=True)
        # testDataFetched =  testDataFetched.select('Independent_features', 'MPG')
        # testDataFetched.show()
        # testDataFetched.printSchema()

        dataset = spark.read.csv(dataset_add, header=True, inferSchema=True)
        dataset.show()

        # renaming the colm
        # print(label_colm)
        # dataset.withColumnRenamed(label_colm, "label")
        # print(label_colm)
        # dataset.show()

        label = ''
        for y in label_colm:
            label = y

        print(label)

        dictionary_list = {'log_list': ["CYLINDERS"],
                           'sqrt_list': ["WEIGHT"],
                           'cubic_list': ["ACCELERATION"]}

        relationship_val = 'linear_reg'

        if relationship_val == 'linear_reg':
            print('linear relationship')
        else:
            dataset = Relationship(dataset, dictionary_list)

        dataset.show()

        # implementing the vector assembler

        featureassembler = VectorAssembler(inputCols=feature_colm,
                                           outputCol="Independent_features")

        output = featureassembler.transform(dataset)

        output.show()
        output = output.select("Independent_features")

        # finalized_data = output.select("Independent_features", label)

        # finalized_data.show()


        regressorTest = LinearRegressionModel.load('/home/fidel/mltest/linearRegressorFitModel')
        predictedData =  regressorTest.transform(output)

        predictedData.show()


    except Exception as e:
        print('exception ' + str(e))
#
# if __name__== '__main__':
#     loadModel()