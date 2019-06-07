import math

from pyspark.sql.functions import col
from pyspark.sql.functions import udf
from pyspark.sql.types import *


def Relationship(dataset, dictionary_list):
    dataset.show()
    # Relationship_val = 'log_list'
    # Relationship_colm = ["CYLINDERS", "WEIGHT", "ACCELERATION", "DISPLACEMENT"]
    #

    # dictionary_list = {'log_list': ["CYLINDERS"],
    #                    'sqrt_list': ["WEIGHT"],
    #                    'cubic_list': ["ACCELERATION"]}

    def Relation_dataset(dictionary_list, dataset):
        # creating the udf
        dataset.show()


        def log_list(x):
            try:
                return math.log(x, 10)
            except Exception as e:
                print('(log error)number should not be less than or equal to zero: ' + str(e))
                pass
            # finally:
            #     print('pass')
            #     pass


        def exponent_list(x):
            try:
                return math.exp(x)
            except Exception as e:
                print('(exception error)number should not be large enough to get it infinity: ' + str(e))

        def square_list(x):
            try:
                return x ** 2
            except Exception as e:
                print('(square error)number should not be negative: ' + str(e))

        def cubic_list(x):
            try:
                return x ** 3
            except Exception as e:
                print('(cubic error)number should not be negative: ' + str(e))

        def quadritic_list(x):
            try:
                return x ** 4
            except Exception as e:
                print('(quadratic error )number should not be negative: ' + str(e))

        def sqrt_list(x):
            try:
                return math.sqrt(x)
            except Exception as e:
                print('(sqare root error) number should not be negative: ' + str(e))

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
                if len(value) == 0:
                    print('length is null')
                else:
                    for colm in value:
                        # Relationship_val.strip("'")
                        # square_list_udf = udf(lambda y: square_list(y), FloatType())
                        dataset = dataset.withColumn(colm, square_list_udf(col(colm).cast(FloatType())))
            if key == 'log_list':
                if len(value) == 0:
                    print('length is null')
                else:
                    for colm in value:
                        # Relationship_val.strip("'")
                        # square_list_udf = udf(lambda y: square_list(y), FloatType())
                        dataset = dataset.withColumn(colm, log_list_udf(col(colm).cast(FloatType())))
            if key == 'exponent_list':
                if len(value) == 0:
                    print('length is null')
                else:
                    for colm in value:
                        # Relationship_val.strip("'")
                        # square_list_udf = udf(lambda y: square_list(y), FloatType())
                        dataset = dataset.withColumn(colm, exponent_list_udf(col(colm).cast(FloatType())))
            if key == 'cubic_list':
                if len(value) == 0:
                    print('length is null')
                else:
                    for colm in value:
                        # Relationship_val.strip("'")
                        # square_list_udf = udf(lambda y: square_list(y), FloatType())
                        dataset = dataset.withColumn(colm, cubic_list_udf(col(colm).cast(FloatType())))
            if key == 'quadratic_list':
                if len(value) == 0:
                    print('length is null')
                else:
                    for colm in value:
                        # Relationship_val.strip("'")
                        # square_list_udf = udf(lambda y: square_list(y), FloatType())
                        dataset = dataset.withColumn(colm, quadratic_list_udf(col(colm).cast(FloatType())))
            if key == 'sqrt_list':
                if len(value) == 0:
                    print('length is null')
                else:
                    for colm in value:
                        # Relationship_val.strip("'")
                        # square_list_udf = udf(lambda y: square_list(y), FloaType())
                        dataset = dataset.withColumn(colm, sqrt_list_udf(col(colm).cast(FloatType())))
            else:
                print('not found')

        return (dataset)

    obj =  Relation_dataset(dictionary_list,dataset)
    return obj


