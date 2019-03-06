from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
import pandas as pd
from scipy import stats



if __name__ == '__main__':


    spark = SparkSession.builder.appName("predictive_analysis").master("local[*]").getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    print"\nspark session created sucessfully:: \n"

    dataset = spark.read.csv("/home/fidel/mltest/auto-miles-per-gallon.csv", header=True, inferSchema=True)

    df = pd.read_csv("/home/fidel/mltest/bank.csv", delimiter=";")

    dataset.show()
    dataset.printSchema()
    ###################################
    ###finding the frequency in colm
    crosstab = pd.crosstab([df.marital], [df.housing], margins=False)

    array = crosstab.iloc[: , :].values
    print array

    #observed = [[27,32], [141, 192], [65, 84]]
    chi2, p, dof, expected = stats.chi2_contingency(array)
    msg = "\ntest statistics: {}\np-value: {}\nDoF:{}\n"
    print (msg.format(round(chi2,2), round(p,2), dof ))
    print (expected,"\n")



    ######################################################
    ##frequency of the each element in colm

    frequency = dataset.groupBy("NAME").count()

    frequency.show()

    dataset.describe().show()

    #######################################################
    ##string indexing

    indexer = StringIndexer(inputCol="NAME", outputCol="Indexed_name")
    indexed = indexer.fit(dataset).transform(dataset)
    indexed.show()
    # indexed_cont = indexed.select("MPG","CYLINDERS","DISPLACEMENT","HORSEPOWER","WEIGHT","ACCELERATION","Indexed_name")
    # indexed_cont.show()
    # #
    # encoder = OneHotEncoder(inputCol="Indexed_name", outputCol="Indexed_vector")
    # encoded = encoder.transform(indexed)
    # encoded.show()


    #######################################################
    ##frequency of the each element in colm
    #
    # frequency1 = dataset.groupBy("Indexed_name").count()
    #
    # frequency1.show()


    ##creating vector assembler

    featureassembler = VectorAssembler(inputCols=["CYLINDERS", "DISPLACEMENT", "WEIGHT", "ACCELERATION"],
                                       outputCol="Independent_features")

    output = featureassembler.transform(indexed)

    output.show()
    output.select("Independent_features").show()

    finalized_data = output.select("Independent_features", "Indexed_name")

    finalized_data.show()


    # chi square test (hypothesis testng)

    from pyspark.ml.stat import ChiSquareTest

    r = ChiSquareTest.test(finalized_data, "Independent_features", "Indexed_name").head()
    print("pValues : " + str(r.pValues))
    print("degreeOfFreedom : " + str(r.degreesOfFreedom))
    print("statistics : " + str(r.statistics))

