import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.feature import RFormula

from pyspark.ml.feature import VectorAssembler

from pyspark.mllib.stat import Statistics

from pyspark.ml.regression import LinearRegression


if __name__=="__main__":
    x=[5,8,9,4,3,5,7]
    y=[1,2,3,4,5,6,7]
    #
    # plt.plot(x, y , label="test", color = 'k')
    # plt.bar(x, y , label="test", color = 'k')
    # scatter=plt.scatter(x, y , label="test", color = 'k', marker="*", s= 10)
    # print scatter
    plt.xlabel("x_label")
    plt.ylabel("y_label")

    plt.title("title of the graph")
    plt.legend()
    plt.plot()
    plt.show()

    spark = SparkSession.builder.appName("predictive_analysis").master("local[*]").getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    dataset = spark.read.csv("/home/fidel/mltest/auto-miles-per-gallon.csv", header=True, inferSchema=True)

    dataset.show()

    abc = dataset.schema.fields

    featuresCol = []

    for x in abc:
        # print(type(x.dataType))
        if(isinstance(x.dataType,StringType)):
            print(x.name + "   "+ str(x.dataType))
            # dataset.select(x.name)
            featuresCol.append(x.name)

    f = ""
    f = "ACCELERATION" + " ~ "


    for x in featuresCol:
        f = f +x + "+"
    f = f[:-1]
    f = (f)

    print(f)

    formula = RFormula(formula=f,featuresCol="features",labelCol= "label")

    output = formula.fit(dataset).transform(dataset)

    output.show(truncate=False)





