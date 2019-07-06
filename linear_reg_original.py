import math
import statistics

import pyspark.sql.functions as f
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from scipy.stats import norm

from PredictionAlgorithms.relationship import Relationship

# from itertools import izip
# from more_itertools import unzip

spark = SparkSession.builder.appName("predictive_Analysis").master("local[*]").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")


#
# dataset_add = "/home/fidel/mltest/auto-miles-per-gallon.csv"
# feature_colm = ["CYLINDERS", "WEIGHT", "ACCELERATION", "DISPLACEMENT", "MODELYEAR"]
# label_colm = "MPG"
#

def Linear_reg(dataset_add, feature_colm, label_colm,relation_list, relation):
    try:
        dataset = spark.read.parquet(dataset_add)
        dataset.show()

        label = ''
        for y in label_colm:
            label = y

        print(label)

        # relationship

        if relation=='linear':
            print('linear relationship')
        if relation=='non_linear':
            dataset = Relationship(dataset, relation_list)

        dataset.show()






        # renaming the colm
        # print (label)
        # dataset.withColumnRenamed(label,"label")
        # print (label)
        # dataset.show()

        featureassembler = VectorAssembler(inputCols=feature_colm,
                                           outputCol="Independent_features")

        output = featureassembler.transform(dataset)

        output.show()
        output.select("Independent_features").show()

        finalized_data = output.select("Independent_features", label)

        finalized_data.show()

        # splitting the dataset into taining and testing

        train_data, test_data = finalized_data.randomSplit([0.75, 0.25], seed=40)

        # applying the model

        lr = LinearRegression(featuresCol="Independent_features", labelCol=label)
        regressor = lr.fit(train_data)

        # print regressor.featureImportances

        # print(dataset.orderBy(feature_colm, ascending=True))

        # pred = regressor.transform(test_data)

        # coefficeint & intercept

        print("coefficient : " + str(regressor.coefficients))
        coefficient_t = str(regressor.coefficients)

        print("intercept : " + str(regressor.intercept))
        intercept_t = str(regressor.intercept)

        prediction = regressor.evaluate(test_data)

        VI_IMP = 2

        prediction_val = prediction.predictions
        prediction_val.show()

        prediction_val_pand = prediction_val.select(label, "prediction").toPandas()

        prediction_val_pand = prediction_val_pand.assign(
            residual_vall=prediction_val_pand[label] - prediction_val_pand["prediction"])

        prediction_val_pand_residual = prediction_val_pand["residual_vall"]


        prediction_val_pand_label = prediction_val_pand[label]

        # print prediction_val_pand_residual
        prediction_val_pand_predict = prediction_val_pand["prediction"]
        # print prediction_val_pand_predict

        # test_summary = prediction.summary

        # for test data

        lr_prediction = regressor.transform(test_data)

        lr_prediction.groupBy(label, "prediction").count().show()

        lr_prediction_quantile = lr_prediction.select(label, "prediction")
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
        #######################################################################################################
        # residual  vs predicted value

        prediction_data = regressor.summary.predictions
        prediction_data.show()
        prediction_data.select(['prediction']).show()
        predicted = prediction_data.select(['prediction'])
        regressor.summary.residuals.show()
        residuals = regressor.summary.residuals
        pred_d = predicted.withColumn('row_index', f.monotonically_increasing_id())
        res_d = residuals.withColumn('row_index', f.monotonically_increasing_id())

        pred_residuals = pred_d.join(res_d, on=['row_index']).sort('row_index').drop('row_index')
        pred_residuals.show()

        pred_residuals.write.parquet('hdfs://10.171.0.181:9000/dev/dmxdeepinsight/datasets/residual_fitted_plot.parquet',
                                     mode='overwrite')

        # scale location plot







        ############################################################################################################

        ####################################################################################
        # appending predicted value to the dataset
        target = dataset.select(label)
        pred = prediction_data.select(['prediction'])
        pred_d = pred.withColumn('row_index', f.monotonically_increasing_id())
        target_d = target.withColumn('row_index', f.monotonically_increasing_id())

        pred_target = pred_d.join(target_d, on=['row_index']).drop('row_index')
        pred_target.show()

        dataset.show()

        pred_target_data_update = dataset.join(pred_target, on=[label])

        pred_target_data_update.show(100)

        ##########################################################################################

        # creating the dictionary for storing the result

        table_response = {

            "Intercept": intercept_t,
            "Coefficients": coefficient_t,
            "RMSE": RMSE,
            "MSE": MSE,
            "R_square": r_square,
            "Adj_R_square": adjsted_r_square,
            "Coefficient_error": coefficient_error,
            "T_value": T_values,
            "P_value": P_values

        }

        # json_response = coefficient_t



        # json_response = {"adjusted r**2 value" : training_summary.r2adj}

        # DATA VISUALIZATION PART

        # finding the quantile in the dataset(Q_Q plot)

        y = 0.1
        x = []

        for i in range(0, 90):
            x.append(y)
            y = round(y + 0.01, 2)
        #
        # for z in x:
        #     print ("~~~~~   ",z)
        #

        quantile_label = lr_prediction_quantile.approxQuantile(label, x, 0.01)
        # print quantile_label
        quantile_prediction = lr_prediction_quantile.approxQuantile("prediction", x, 0.01)
        # print quantile_prediction


        Q_label_pred=''
        print(len(quantile_label))
        length = len(quantile_label)

        for i in range(0,len(quantile_label)):
            Q_label_pred += str(quantile_label[i]) + 't'  +  str(quantile_prediction[i]) + 'n'


        #
        #
        # with open('Q_Q_plot.csv', 'w') as Q_Q:
        #     writer_Q_Q = csv.writer(Q_Q)
        #     writer_Q_Q.writerows((quantile_label, quantile_prediction))

        # plt.scatter(quantile_label, quantile_prediction)
        # plt.show()

        ## finding the residual vs fitted graph data

        # plt.scatter(prediction_val_pand_predict, prediction_val_pand_residual)
        # plt.axhline(y=0.0, color="red")
        # plt.xlabel("prediction")
        # plt.ylabel("residual")
        # plt.title("residual vs fitted ")
        # # plt.show()

        # creating the csv file and writitng into it


        fitted_residual = ''
        print(len(prediction_val_pand_residual))
        length = len(prediction_val_pand_residual)

        for i in range(0, len(prediction_val_pand_residual)):
            fitted_residual += str(prediction_val_pand_predict[i]) + 't' + str(prediction_val_pand_residual[i]) + 'n'

        #
        # with open('residual_vs_fitted.csv', 'w') as r_f:
        #     writer_r_f = csv.writer(r_f)
        #     writer_r_f.writerows((prediction_val_pand_predict, prediction_val_pand_residual))

        ## residual vs leverage graph data

        # prediction_val_pand_residual
        # extreme value in the predictor colm
        # prediction_col_extremeval = lr_prediction_quantile.agg({"prediction": "max"})
        # prediction_col_extremeval.show()

        # plt.plot(prediction_col_extremeval, prediction_val_pand_residual)
        # plt.show()

        ## scale location graph data

        prediction_val_pand_residual
        prediction_val_pand_predict
        prediction_val_pand_residual_abs = prediction_val_pand_residual.abs()
        sqrt_residual = []
        for x in prediction_val_pand_residual_abs:
            sqrt_residual.append(math.sqrt(x))
            # print ("____________________  ",x)

        sqrt_residual


        ########################################

        # calculating std deviation

        print(statistics.stdev(prediction_val_pand_residual))
        stdev_ = statistics.stdev(prediction_val_pand_residual)

        # calcuate stnd residuals
        std_res = []
        for x in prediction_val_pand_residual:
            std_res.append(x / stdev_)
        print(std_res)

        # calculating the square root of std_res

        sqr_std_res = []
        for x in std_res:
            sqr_std_res.append(math.sqrt(abs(x)))
        print(sqr_std_res)

        #######################################

        #
        # # square root of label
        # sqrt_label = []
        # for x in prediction_val_pand_label:
        #     sqrt_label.append(math.sqrt(abs(x)))
        #
        # sqrt_label
        # prediction_val_pand_residual
        # std_residual = []
        # for sqr, resid in zip(sqrt_label, prediction_val_pand_residual):
        #     std_residual.append(resid / sqr)
        #     # print(std_sqrt_residual)
        #
        # # creating the std sqr root
        #
        # sqrt_std_residuals = []
        # for x in std_residual:
        #     # print(math.sqrt(abs(x)))
        #     sqrt_std_residuals.append(math.sqrt(abs(x)))
        # print(sqrt_std_residuals)
        #
        #
        #
        # t_sqrt_std_residuals = []
        # for x in sqrt_std_residuals:
        #     # print(math.sqrt(abs(x)))
        #     t_sqrt_std_residuals.append(math.sqrt(abs(x)))
        # # print(sqrt_std_residuals)
        #

        # print(std_sqrt_residual)

        scale_predict_residual = ''
        for pre, res in zip(prediction_val_pand_predict, sqr_std_res):
            scale_predict_residual += str(pre) + 't' + str(res) + 'n'
        print(scale_predict_residual)

        #######################################################################################3
        # QUANTILE

        y = 0.1
        x = []

        for i in range(0, 90):
            x.append(y)
            y = round(y + 0.01, 2)

        quantile_std_res = spark.createDataFrame(std_res, FloatType())
        quantile_std_res.show()
        quantile_std_res_t = quantile_std_res.approxQuantile('value', x, 0.01)
        print(quantile_std_res_t)
        print(x)


        # calculating the z_score


        ## sort the list
        sorted_std_res = sorted(std_res)

        mean = statistics.mean(sorted_std_res)
        stdev = statistics.stdev(sorted_std_res)
        # print(mean)
        quantile = []
        n = len(std_res)
        print(n)
        for x in range(0,n):
            quantile.append((x-0.5) / (n))

        print(quantile)

        # z_score theoratical
        z_theory = []
        for x in quantile:
            z_theory.append(norm.ppf(abs(x)))

        # z score for real val
        z_pract = []
        for x in sorted_std_res:
            z_pract.append((x-mean)/stdev)



        Q_label_pred = ''
        # print(len(quantile_label))
        # length = len(quantile_label)

        # z=[-2.0,-1.5,-1.0,-0.5,0, 0.5,1.0,1.5,2.0,2.5]

        for quant,val in zip(z_theory,z_pract):
            Q_label_pred += str(quant) + 't' + str(val) + 'n'


        # plt.scatter(z_pract,z_theory)
        # plt.savefig()

        #
        # plt.scatter(z_theory,z_pract)
        # plt.show()

        ####################################################

        ##########################################################################################
        # # plt.scatter(sqrt_residual, prediction_val_pand_predict)
        # # plt.show()
        #
        #
        #
        #
        # scale_predict_residual=''
        #
        # print(len(sqrt_residual))
        # length = len(sqrt_residual)
        #
        # for i in range(0, len(sqrt_residual)):
        #     scale_predict_residual += str(prediction_val_pand_predict[i]) + 't' + str(sqrt_residual[i]) + 'n'
        # #
        # with open('scale_location_plot.csv', 'w') as s_l:
        #     writer_s_l = csv.writer(s_l)
        #     writer_s_l.writerows((prediction_val_pand_predict, sqrt_residual))

        # dumping the dictionary into json object

        graph_response = {
            "Q_Q_plot": Q_label_pred,
            "residual_fitted": fitted_residual,
            "scale_location": scale_predict_residual
        }



        json_response = {

            'table_data': table_response,
            'graph_data' : graph_response


        }

        # json_response = coefficient_t

        print(json_response)



        # json_response = {'run_status': 'success', 'PredictiveResponse': resultdf}
        return (json_response)
    except Exception as e:
        print('exception is =' + str(e))




#
# Linear_reg(dataset_add, feature_colm, label)

# if __name__== "__main__":
#     Linear_reg()
