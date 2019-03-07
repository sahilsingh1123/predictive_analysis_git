from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
import json

spark = SparkSession.builder.appName("predictive_Analysis").master("local[*]").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

#
# dataset_add = "/home/fidel/mltest/auto-miles-per-gallon.csv"
# feature_colm = ["CYLINDERS", "WEIGHT", "ACCELERATION", "DISPLACEMENT", "MODELYEAR"]
# label_colm = "MPG"
#

def Linear_reg(dataset_add, feature_colm, label_colm):
    dataset = spark.read.csv(dataset_add, header=True , inferSchema=True)
    dataset.show()

    featureassembler = VectorAssembler(inputCols=feature_colm,
        outputCol="Independent_features")


    output = featureassembler.transform(dataset)

    output.show()
    output.select("Independent_features").show()

    finalized_data = output.select("Independent_features", label_colm)

    finalized_data.show()



    # splitting the dataset into taining and testing

    train_data, test_data = finalized_data.randomSplit([0.75, 0.25], seed=40)

    # applying the model

    lr = LinearRegression(featuresCol="Independent_features", labelCol=label_colm)
    regressor = lr.fit(train_data)

    # print regressor.featureImportances

    # print(dataset.orderBy(feature_colm, ascending=True))


    # pred = regressor.transform(test_data)

    # coefficeint & intercept

    print("coefficient : " + str(regressor.coefficients))
    coefficient_t = str(regressor.coefficients)

    print("intercept : " + str(regressor.intercept))
    intercept_t =  str(regressor.intercept)

    prediction = regressor.evaluate(test_data)


    prediction.predictions.show()

    training_summary = regressor.summary



    print("numof_Iterations...%d\n" % training_summary.totalIterations)
    print("ObjectiveHistory...%s\n" % str(training_summary.objectiveHistory))
    print("RMSE...%f\n" % training_summary.rootMeanSquaredError)
    print("MSE....%f\n" % training_summary.meanSquaredError)
    print("r**2(r-square)....::%f\n" % training_summary.r2)
    print("r**2(r-square adjusted)....%f\n" % training_summary.r2adj)
    print("deviance residuals %s" % str(training_summary.devianceResiduals))
    training_summary.residuals.show()
    print("coefficient standard errors: \n" + str(training_summary.coefficientStandardErrors))
    print(" Tvalues :\n" + str(training_summary.tValues))
    print(" p values :\n" + str(training_summary.pValues))

    json_response = {"adjusted r**2 value" : training_summary.r2adj}


    return str(json.dumps(json_response)).encode("utf-8")

    # matplot visualization
    #
    # plt.scatter(coefficient_t,intercept_t , color="r")
    # plt.show()

#
# Linear_reg(dataset_add, feature_colm, label_colm)