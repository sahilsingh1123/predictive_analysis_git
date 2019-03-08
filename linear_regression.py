from pyspark.sql import SparkSession

from pyspark.ml.feature import VectorAssembler

from pyspark.ml.regression import LinearRegression

if __name__ == '__main__':
    spark = SparkSession.builder.appName("predictive_analysis").master("local[*]").getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    dataset = spark.read.csv("/home/fidel/mltest/auto-miles-per-gallon.csv", header=True, inferSchema=True)

    dataset.show()


    #
    # dataset2=dataset.describe().toPandas().transpose()
    # dataset2.collect()



    # dataset.na.replace("?", "80.0")
    # dataset.na.replace(["null"], ["80.0"], 'HORSEPOWER')

    # dataset = dataset.withColumn("HORSEPOWER", col("HORSEPOWER").cast("integer"))
    # # dataset.na.fill({"HORSEPOWER": 80.0})
    # dataset.printSchema()
    # dataset.fillna(80)
    # dataset.show(50)


    # creating vector assembler

    print "heloooo"

    featureassembler = VectorAssembler(inputCols=["CYLINDERS", "WEIGHT", "ACCELERATION","DISPLACEMENT", "MODELYEAR"],
                                       outputCol="Independent features")

    output = featureassembler.transform(dataset)

    output.show()
    output.select("Independent features").show()

    output_features = output.select("independent features")
    output_label = output.select("MPG")

    finalized_data = output.select("Independent features", "MPG")

    finalized_data.show()

    # correlation
    #
    # featureassembler_correlation = VectorAssembler(
    #     inputCols=["MPG", "CYLINDERS", "DISPLACEMENT", "WEIGHT", "ACCELERATION"], outputCol="correlation_colm")
    # output_corr = featureassembler_correlation.transform(dataset)
    # output_corr.show()
    #
    # finalized_corr = output_corr.select("correlation_colm")
    # finalized_corr.show()
    # from pyspark.ml.stat import Correlation
    #
    # r2 = Correlation.corr(output_corr, "correlation_colm").head()
    # print("pearson correlation matrix : \n : " + str(r2[0]))
    #
    # r1 = Correlation.corr(output, "Independent features").head()
    # print("pearson correlation matrix  : \n" + str(r1[0]))
    #
    # r1s = Correlation.corr(output_corr, "correlation_colm", "spearman").head()
    # print(" spearman correlation...: \n" + str(r1s[0]))

    # statistics of the tables

    #
    # rdd_stats = finalized_corr.rdd.map(list)
    # summary = Statistics.colStats(rdd_stats)
    # summary.variance()
    # summary.mean()
    # summary.max()
    # summary.min()
    # summary.numNonzeros()

    # pyspark sql functions
    #
    # import pyspark.sql.functions
    #
    # finalized_corr.show()
    # dataset.cov("MPG", "ACCELERATION")
    # dataset.summary().show()
    # dataset.describe().show()

    # chi square test (hypothesis testng)

    # from pyspark.ml.stat import ChiSquareTest
    #
    # r = ChiSquareTest.test(finalized_data, "Independent features", "MPG").head()
    # print("pValues : " + str(r.pValues))
    # print("degreeOfFreedom : " + str(r.degreesOfFreedom))
    # print("statistics : " + str(r.statistics))

    # splitting the model into training and testing dataset

    train_data, test_data = finalized_data.randomSplit([0.75, 0.25] , seed=40)

    # applying the model

    regressor = LinearRegression(featuresCol="Independent features", labelCol="MPG")
    regressor = regressor.fit(train_data)

    print regressor.numFeatures

    #
    # model_fitted_y = regressor.fittedvalues
    # model_fitted_y.show()
    # model_residuals= regressor.resid
    # model_residuals.show()

    # coefficeint & intercept

    print("coefficient : " + str(regressor.coefficients))

    coefficents_m = str(regressor.coefficients)

    print("intercept : " + str(regressor.intercept))

    intercept_b =  regressor.intercept


    #
    # plt.plot(output_features, output_label)
    # plt.plot(output_features, intercept_b + coefficents_m*output_features, "-")
    # plt.show()
    #


    prediction_va = regressor.evaluate(test_data)


    prediction_val =    prediction_va.predictions
    prediction_val.show()


    prediction_val_pand = prediction_val.select("MPG", "prediction").toPandas()

    prediction_val_pand = prediction_val_pand.assign(residual_vall=prediction_val_pand["MPG"] - prediction_val_pand["prediction"])


    prediction_val_pand_residual = prediction_val_pand["residual_vall"]
    print prediction_val_pand_residual
    prediction_val_pand_predict = prediction_val_pand["prediction"]
    print prediction_val_pand_predict

    import matplotlib.pyplot as plt

    plt.scatter(prediction_val_pand_predict,prediction_val_pand_residual)
    plt.axhline(y=0.0, color = "red")
    plt.xlabel("prediction")
    plt.ylabel("residual")
    plt.title("residual vs fitted ")
    plt.show()

    print prediction_val_pand

    # import pandas as pd
    #
    #
    # df = pd.DataFrame([["Australia", 1, 3, 5],
    #                    ["Bambua", 12, 33, 56],
    #                    ["Tambua", 14, 34, 58]
    #                    ], columns=["Country", "Val1", "Val2", "Val10"]
    #                   )
    #
    # df = df.assign(Val10_minus_Val1=df['Val10'] - df['Val1'])
    # print df

    # for MPG, prediction in prediction_val_pand.iterrows():
    #     residual_val.append(MPG-prediction)
    #
    # print residual_val

    # prediction.predictions.show()

    # prediction.groupBy("MPG", "prediction").count().show()

    # for train data

    # lr_prediction_train = regressor.transform(train_data)
    # lr_prediction_train.show()
    # lr_prediction_train_pandas = lr_prediction_train.toPandas()
    # lr_prediction_train_predictcol = lr_prediction_train_pandas.select("prediction")
    # lr_prediction_train_labelcol = lr_prediction_train_pandas.select("MPG")

    # for test data

    lr_prediction = regressor.transform(test_data)

    lr_prediction.groupBy("MPG", "prediction").count().show()

    lr_prediction_quantile = lr_prediction.select("MPG", "prediction")
    lr_prediction_quantile.show()
    # lr_prediction_pandas = lr_prediction.toPandas()

    #
    # lr_prediction_sql_prediction = lr_prediction.select("prediction")

    # from pyspark.ml.evaluation import RegressionEvaluator
    #
    # lr_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="MPG", metricName="r2")
    #
    # print ("R sqrd on test data = %g" % lr_evaluator.evaluate(lr_prediction))
    #

    training_summary = regressor.summary


    print("numof_Iterations...%d\n" % training_summary.totalIterations)
    print("ObjectiveHistory...%s\n" % str(training_summary.objectiveHistory))
    print("RMSE...%f\n" % training_summary.rootMeanSquaredError)
    print("MSE....%f\n" % training_summary.meanSquaredError)
    print("r**2(r-square)....%f\n" % training_summary.r2)
    print("r**2(r-square adjusted)....%f\n" % training_summary.r2adj)

    training_summary.residuals.show()
    residual_graph = training_summary.residuals
    residual_graph_pandas = residual_graph.toPandas()
    print("coefficient standard errors: \n" + str(training_summary.coefficientStandardErrors))
    print(" Tvalues :\n" + str(training_summary.tValues))
    print(" p values :\n" + str(training_summary.pValues))
    print("  :\n" + str(training_summary.devianceResiduals))


    # converting the dataset into pandas dataset

    dataset_pandas = dataset.toPandas()

    import matplotlib.pyplot as plt

    data_iloc = dataset_pandas.iloc[0:9, :].values
    print data_iloc
    print dataset_pandas.mean()
    print dataset_pandas.max()
    print dataset_pandas.std()

    # plotting the graph
    # import pandas as pd
    # dataset_graph = dataset.toPandas()
    #
    # train_p = train_data.toPandas()
    # test_p = test_data.toPandas()
    #
    # lr_prediction_sql_label = lr_prediction.select("MPG").toPandas()
    # lr_prediction_sql_label_topandas = lr_prediction_sql_label.toPandas()
    # lr_prediction_sql_label_topandas.show()
    # lr_prediction_sql_prediction_topandas=lr_prediction_sql_prediction.toPandas()
    # lr_prediction_sql_prediction_topandas.show()
    #
    #
    # import matplotlib.pyplot as plt
    #
    #
    # plt.plot(lr_prediction.MPG, lr_prediction.prediction, color="r")
    # plt.show()
    #


    # x =[]
    # for r in lr_prediction_sql_topandas:
    #     x.append(r)
    # print x
    # plt.plot(lr_prediction_sql_label_topandas, lr_prediction_sql_prediction_topandas, color = "red")
    # plt.scatter(lr_prediction_sql_topandas, lr_prediction_sql_label, color="blue")
    # plt.scatter(lr_prediction_sql_label, lr_prediction_sql_topandas, color = "blue")
    # plt.show()
    #
    # import seaborn as sns
    #
    # sns.lmplot(x="MPG",y="prediction", data=lr_prediction_pandas)
    #

    # plt.plot()
    # plt.scatter(dataset_graph["WEIGHT"], dataset_graph["MPG"])
    # plt.show()
    #
    # from pyspark_dist_explore import hist
    # import matplotlib.pyplot as plt
    #
    # fig, ax = plt.subplots()
    # hist(ax, dataset, bins=20, color=['red'])



    ##########################################################################

    # DATA VISUALIZATION PART

    ## finding the quantile in the dataset

    quantile_label = lr_prediction_quantile.approxQuantile("MPG", [0.25, 0.50, 0.75, 0.99], 0.01)
    print quantile_label
    quantile_prediction = lr_prediction_quantile.approxQuantile("prediction", [0.25, 0.50, 0.75, 0.99], 0.01)
    print quantile_prediction

    ## finding the residual vs fitted graph data

    residual_graph.show()
    prediction_col = lr_prediction_quantile.select("prediction")

    prediction_col.show()

    ## residual vs leverage graph data

    residual_graph
    # extreme value in the predictor colm
    prediction_col_extremeval = lr_prediction_quantile.agg({"prediction" : "max"})
    prediction_col_extremeval.show()


    ## scale location graph data

    residual_graph.show()
    prediction_col = lr_prediction_quantile.select("prediction")

    prediction_col.show()

