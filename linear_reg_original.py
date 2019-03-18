from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql.functions import col
import csv
from itertools import izip
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

    # renaming the colm
    print label_colm
    dataset.withColumnRenamed(label_colm,"label")
    print label_colm
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

    VI_IMP = 2




    prediction_val =    prediction.predictions
    prediction_val.show()


    prediction_val_pand = prediction_val.select(label_colm, "prediction").toPandas()

    prediction_val_pand = prediction_val_pand.assign(residual_vall=prediction_val_pand[label_colm] - prediction_val_pand["prediction"])


    prediction_val_pand_residual = prediction_val_pand["residual_vall"]
    # print prediction_val_pand_residual
    prediction_val_pand_predict = prediction_val_pand["prediction"]
    # print prediction_val_pand_predict



    # test_summary = prediction.summary

    # for test data

    lr_prediction = regressor.transform(test_data)

    lr_prediction.groupBy(label_colm, "prediction").count().show()

    lr_prediction_quantile = lr_prediction.select(label_colm, "prediction")
    # lr_prediction_quantile.show()

    training_summary = regressor.summary



    print("numof_Iterations...%d\n" % training_summary.totalIterations)
    print("ObjectiveHistory...%s\n" % str(training_summary.objectiveHistory))
    print("RMSE...%f\n" % training_summary.rootMeanSquaredError)
    RMSE = training_summary.rootMeanSquaredError
    print("MSE....%f\n" % training_summary.meanSquaredError)
    MSE = training_summary.meanSquaredError
    print("r**2(r-square)....::%f\n" % training_summary.r2)
    r_square = training_summary.r2
    print("r**2(r-square adjusted)....%f\n" % training_summary.r2adj)
    adjsted_r_square = training_summary.r2adj
    print("deviance residuals %s" % str(training_summary.devianceResiduals))
    training_summary.residuals.show()
    residual_graph = training_summary.residuals
    residual_graph_pandas = residual_graph.toPandas()
    print("coefficient standard errors: \n" + str(training_summary.coefficientStandardErrors))
    coefficient_error = str(training_summary.coefficientStandardErrors)
    print(" Tvalues :\n" + str(training_summary.tValues))
    T_values = str(training_summary.tValues)
    print(" p values :\n" + str(training_summary.pValues))
    P_values = str(training_summary.pValues)


    # creating the dictionary for storing the result

    json_response = {

        "Intercept" : intercept_t,
        "Coefficients" : coefficient_t,
        "RMSE" : RMSE,
        "MSE" : MSE,
        "R_square" : r_square,
        "Adj_R_square" : adjsted_r_square,
        "Coefficient_error" : coefficient_error,
        "T_value" : T_values,
        "P_value" : P_values


    }

    print json_response

    # json_response = {"adjusted r**2 value" : training_summary.r2adj}

    # DATA VISUALIZATION PART

    ## finding the quantile in the dataset(Q_Q plot)
    import matplotlib.pyplot as plt

    y = 0.1
    x=[]

    for i in range(0,90):
        x.append(y)
        y=round(y+0.01,2)
    #
    # for z in x:
    #     print ("~~~~~   ",z)
    #



    quantile_label = lr_prediction_quantile.approxQuantile(label_colm, x, 0.01)
    # print quantile_label
    quantile_prediction = lr_prediction_quantile.approxQuantile("prediction", x, 0.01)
    # print quantile_prediction


    with open('Q_Q_plot.csv', 'w') as Q_Q:
        writer_Q_Q = csv.writer(Q_Q)
        writer_Q_Q.writerows(izip(quantile_label, quantile_prediction))


    plt.scatter(quantile_label, quantile_prediction)
    plt.show()


    ## finding the residual vs fitted graph data




    plt.scatter(prediction_val_pand_predict,prediction_val_pand_residual)
    plt.axhline(y=0.0, color = "red")
    plt.xlabel("prediction")
    plt.ylabel("residual")
    plt.title("residual vs fitted ")
    plt.show()

    # creating the csv file and writitng into it



    with open('residual_vs_fitted.csv', 'w') as r_f:
        writer_r_f = csv.writer(r_f)
        writer_r_f.writerows(izip(prediction_val_pand_predict, prediction_val_pand_residual))


    ## residual vs leverage graph data

    prediction_val_pand_residual
    # extreme value in the predictor colm
    prediction_col_extremeval = lr_prediction_quantile.agg({"prediction": "max"})
    # prediction_col_extremeval.show()

    # plt.plot(prediction_col_extremeval, prediction_val_pand_residual)
    # plt.show()

    ## scale location graph data

    prediction_val_pand_residual
    prediction_val_pand_predict
    prediction_val_pand_residual_abs = prediction_val_pand_residual.abs()
    import math
    sqrt_residual=[]
    for x in prediction_val_pand_residual_abs:
        sqrt_residual.append(math.sqrt(x))
        # print ("____________________  ",x)

    sqrt_residual

    plt.scatter(sqrt_residual, prediction_val_pand_predict)
    plt.show()



    with open('scale_location_plot.csv', 'w') as s_l:
        writer_s_l = csv.writer(s_l)
        writer_s_l.writerows(izip(prediction_val_pand_predict, sqrt_residual))

    # dumping the dictionary into json object

    return str(json.dumps(json_response)).encode("utf-8")



#
# Linear_reg(dataset_add, feature_colm, label_colm)