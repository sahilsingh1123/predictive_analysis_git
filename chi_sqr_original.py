from pyspark.sql import SparkSession
from pyspark.ml.feature import RFormula
from pyspark.ml.stat import ChiSquareTest

# if __name__=="__main__":


dataset_add= "/home/fidel/mltest/bank.csv"
features = ["default","housing","loan"]
label = "marital"


spark = SparkSession.builder.appName("predictive_analysis").master("local[*]").getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

class chi:

    def Chi_sqr(dataset_add, features, label):
        dataset = spark.read.csv(dataset_add, header=True, inferSchema=True, sep=";")

        dataset.show()

        # using the rformula for indexing, encoding and vectorising

        f = ""
        f = label+" ~ "

        for x in features:
            f = f + x + "+"
        f = f[:-1]
        f = (f)

        formula = RFormula(formula= f,
                           featuresCol="features",
                           labelCol= "label")

        output = formula.fit(dataset).transform(dataset)

        output.select("features", "label").show()


        # chi selector

        from pyspark.ml.feature import ChiSqSelector

        selector = ChiSqSelector(numTopFeatures=3, featuresCol="features", outputCol="selected_features", labelCol="label")

        result = selector.fit(output).transform(output)

        print("chi2 output with top %d features selected " % selector.getNumTopFeatures())
        result.show()

        # runnin gfor the chi vallue test

        r = ChiSquareTest.test(result, "selected_features", "label").head()
        print("pValues: " + str(r.pValues))
        print("degreesOfFreedom: " + str(r.degreesOfFreedom))
        print("statistics: " + str(r.statistics))

    Chi_sqr(dataset_add, features, label)