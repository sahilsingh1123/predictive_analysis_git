from pyspark.sql import SparkSession
from pyspark.ml.feature import RFormula
from pyspark.ml.stat import ChiSquareTest

#
# dataset_add= "/home/fidel/mltest/heart.csv"
# features_colm = ["age","sex","cp","trestbps","chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
# print(features_colm.__len__())
# label_colm = ["target"]


spark = SparkSession.builder.appName("predictive_analysis").master("local[*]").getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

def Chi_sqr(dataset_add, feature_colm, label_colm):
    dataset = spark.read.csv(dataset_add, header=True, inferSchema=True)

    dataset.show()

    # using the rformula for indexing, encoding and vectorising

    label = ''
    for y in label_colm:
        label = y

    print(label)


    f = ""
    f = label+" ~ "

    for x in feature_colm:
        f = f + x + "+"
    f = f[:-1]
    f = (f)

    formula = RFormula(formula= f,
                       featuresCol="features",
                       labelCol= "label")

    length=feature_colm.__len__()

    output = formula.fit(dataset).transform(dataset)

    output.select("features", "label").show()

    # chi selector
    from pyspark.ml.feature import ChiSqSelector

    selector = ChiSqSelector(numTopFeatures=length, featuresCol="features", outputCol="selected_features", labelCol="label")

    result = selector.fit(output).transform(output)

    print("chi2 output with top %d features selected " % selector.getNumTopFeatures())
    result.show()

    #runnin gfor the chi vallue test

    r = ChiSquareTest.test(result, "selected_features", "label").head()
    print("pValues: " + str(r.pValues))
    p_values = str(r.pValues)
    print("degreesOfFreedom: " + str(r.degreesOfFreedom))

    print("statistics: " + str(r.statistics))


    json_response = {'pvalues' : p_values}

    return json_response

# Chi_sqr(dataset_add, features_colm, label_colm)