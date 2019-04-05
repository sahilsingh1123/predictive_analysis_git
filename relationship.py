import math
from pyspark.sql.functions import udf
from pyspark.sql.functions import col
import numpy as np
from pyspark.sql.types import *




def Relationship(dataset, dictionary_list):
    # Relationship_val = 'log_list'
    # Relationship_colm = ["CYLINDERS", "WEIGHT", "ACCELERATION", "DISPLACEMENT"]
    #

    # dictionary_list = {'log_list': ["CYLINDERS"],
    #                    'sqrt_list': ["WEIGHT"],
    #                    'cubic_list': ["ACCELERATION"]}

    def Relation_dataset(dictionary_list, dataset):
        # creating the udf



        def log_list(x):
            return math.log(x)

        def exponent_list(x):
            return math.exp(x)

        def square_list(x):
            return x ** 2

        def cubic_list(x):
            return x ** 3

        def quadritic_list(x):
            return x ** 4

        def sqrt_list(x):
            return math.sqrt(x)

        square_list_udf = udf(lambda y: square_list(y), FloatType())
        log_list_udf = udf(lambda y: log_list(y), FloatType())
        exponent_list_udf = udf(lambda y: exponent_list(y), FloatType())
        cubic_list_udf = udf(lambda y: cubic_list(y), FloatType())
        quadratic_list_udf = udf(lambda y: quadritic_list(y), FloatType())
        sqrt_list_udf = udf(lambda y: sqrt_list(y), FloatType())

        # spark.udf.register("squaredWithPython", square_list)

        # square_list_udf = udf(lambda y: square_list(y), ArrayType(FloatType))

        # square_list_udf = udf(lambda y: exponent_list(y), FloatType())
        #

        #
        # # dataset.select('MPG', square_list_udf(col('MPG').cast(FloatType())).alias('MPG')).show()
        #
        # dataset.withColumn('MPG', square_list_udf(col('MPG').cast(FloatType()))).show()

        # Relationship_val = 'square_list'
        # Relationship_colm = ["CYLINDERS", "WEIGHT", "ACCELERATION", "DISPLACEMENT"]
        # Relationship_model = ['log_list', 'exponent_list', 'square_list', 'cubic_list', 'quadritic_list',
        #                       'sqrt_list']

        for key, value in dictionary_list.items():
            if key == 'square_list':
                for colm in value:
                    # Relationship_val.strip("'")
                    # square_list_udf = udf(lambda y: square_list(y), FloatType())
                    dataset = dataset.withColumn(colm, square_list_udf(col(colm).cast(FloatType())))
            if key == 'log_list':
                for colm in value:
                    # Relationship_val.strip("'")
                    # square_list_udf = udf(lambda y: square_list(y), FloatType())
                    dataset = dataset.withColumn(colm, log_list_udf(col(colm).cast(FloatType())))
            if key == 'exponent_list':
                for colm in value:
                    # Relationship_val.strip("'")
                    # square_list_udf = udf(lambda y: square_list(y), FloatType())
                    dataset = dataset.withColumn(colm, exponent_list_udf(col(colm).cast(FloatType())))
            if key == 'cubic_list':
                for colm in value:
                    # Relationship_val.strip("'")
                    # square_list_udf = udf(lambda y: square_list(y), FloatType())
                    dataset = dataset.withColumn(colm, cubic_list_udf(col(colm).cast(FloatType())))
            if key == 'quadritic_list':
                for colm in value:
                    # Relationship_val.strip("'")
                    # square_list_udf = udf(lambda y: square_list(y), FloatType())
                    dataset = dataset.withColumn(colm, quadratic_list_udf(col(colm).cast(FloatType())))
            if key == 'sqrt_list':
                for colm in value:
                    # Relationship_val.strip("'")
                    # square_list_udf = udf(lambda y: square_list(y), FloaType())
                    dataset = dataset.withColumn(colm, sqrt_list_udf(col(colm).cast(FloatType())))
            else:
                print('not found')

        return (dataset)

    obj =  Relation_dataset(dictionary_list,dataset)
    return obj


