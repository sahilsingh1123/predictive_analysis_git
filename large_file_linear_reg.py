from pyspark.sql import SparkSession
from pyspark.ml.feature import RFormula
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

if __name__ == '__main__':
    spark = SparkSession.builder.appName("predictive_analysis").master("local[*]").getOrCreate()

    spark.sparkContext.setLogLevel("INFO")

    dataset = spark.read.csv("hdfs://10.171.0.7:9000/TestFiles/10Million.csv", header=True, inferSchema=True)

    dataset.show(100)
    dataset=dataset.persist()
    #
    # formula = RFormula(formula="Profit ~ Sales + quantity + TaxAmount",
    #                    featuresCol="features",
    #                    labelCol="label")
    #
    # output = formula.fit(dataset).transform(dataset)
    #
    # # finalized_data = output.select("features", "label")


    featureassembler = VectorAssembler(inputCols=["Sales", "quantity", "TaxAmount"],
                                       outputCol="Independent_features")

    output = featureassembler.transform(dataset)

    output_features = output.select("independent_features")
    output_label = output.select("Profit")

    finalized_data = output.select("Independent_features", "Profit")

    #
    # train_data, test_data = finalized_data.randomSplit([0.75, 0.25] , seed=40)
    #
    # regressor = LinearRegression(featuresCol="features", labelCol="label")
    # regressor = regressor.fit(train_data)
    #
    #
    # print("coefficient : " + str(regressor.coefficients))
    #
    # coefficents_m = str(regressor.coefficients)
    #
    # print("intercept : " + str(regressor.intercept))
    #
    # intercept_b =  regressor.intercept
    #
    # prediction_va = regressor.evaluate(test_data)
    #
    #
    # prediction_val =    prediction_va.predictions
    # prediction_val.show()
    #
    # prediction_val_pand = prediction_val.select("label", "prediction").toPandas()
    #
    # prediction_val_pand = prediction_val_pand.assign(residual_vall=prediction_val_pand["label"] - prediction_val_pand["prediction"])
    #
    #
    # prediction_val_pand_residual = prediction_val_pand["residual_vall"]
    # print prediction_val_pand_residual
    # prediction_val_pand_predict = prediction_val_pand["prediction"]
    # print prediction_val_pand_predict
    # prediction_val_pand_label = prediction_val_pand["label"]
    # print prediction_val_pand_label
    #
    #
    # training_summary = regressor.summary
    #
    #
    # print("numof_Iterations...%d\n" % training_summary.totalIterations)
    # print("ObjectiveHistory...%s\n" % str(training_summary.objectiveHistory))
    # print("RMSE...%f\n" % training_summary.rootMeanSquaredError)
    # print("MSE....%f\n" % training_summary.meanSquaredError)
    # print("r**2(r-square)....%f\n" % training_summary.r2)
    # print("r**2(r-square adjusted)....%f\n" % training_summary.r2adj)
    #
    # training_summary.residuals.show()
    # residual_graph = training_summary.residuals
    # residual_graph_pandas = residual_graph.toPandas()
    # print("coefficient standard errors: \n" + str(training_summary.coefficientStandardErrors))
    # print(" Tvalues :\n" + str(training_summary.tValues))
    # print(" p values :\n" + str(training_summary.pValues))
    # print("  :\n" + str(training_summary.devianceResiduals))



