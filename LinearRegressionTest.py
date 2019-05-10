import csv

import pyspark.sql.functions as f
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pyspark.sql.types import *

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
class LinearRegressionModel():
    def __init__(self, trainDataRatio=0.80):
        self.trainDataRatio = trainDataRatio


    def linearReg(self, dataset_add, feature_colm, label_colm, relation_list, relation):
        try:
            dataset = spark.read.csv(dataset_add, header=True, inferSchema=True)
            dataset.show()
            
            # renaming the colm
            # print(label_colm)
            # dataset.withColumnRenamed(label_colm, "label")
            # print(label_colm)
            # dataset.show()

            label = ''
            for y in label_colm:
                label = y

            print(label)

            dictionary_list = {'log_list': ["CYLINDERS"],
                               'sqrt_list': ["WEIGHT"],
                               'cubic_list': ["ACCELERATION"]}

            relationship_val = 'linear_reg'

            if relationship_val == 'linear_reg':
                print('linear relationship')
            else:
                dataset = Relationship(dataset, dictionary_list)

            dataset.show()

            # implementing the vector assembler

            featureassembler = VectorAssembler(inputCols=feature_colm,
                                               outputCol="Independent_features")

            output = featureassembler.transform(dataset)

            output.show()
            output.select("Independent_features").show()

            finalized_data = output.select("Independent_features", label)

            finalized_data.show()

            for x in self.trainDataRatio:
                trainDataRatioTransformed = x
            testDataRatio = 1 - trainDataRatioTransformed

            # splitting the dataset into taining and testing
            # testDataRatio = 1-self.trainDataRatio

            train_data, test_data = finalized_data.randomSplit([trainDataRatioTransformed, testDataRatio], seed=40)

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

            # VI_IMP = 2

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
            lr_prediction_onlypred = lr_prediction.select('prediction')
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
            # residual_graph = training_summary.residuals
            # test = (residual_graph, lr_prediction_onlypred)
            # residual_graph.write.csv('/home/fidel/PycharmProjects/predictive_analysis_git', header=True, mode='append' )
            # print(test)
            # test.write.csv('/home/fidel/PycharmProjects/predictive_analysis_git', header=True, mode= 'append')
            # residual_graph_pandas = residual_graph.toPandas()
            print("coefficient standard errors: \n" + str(training_summary.coefficientStandardErrors))
            coefficientStdError = str(training_summary.coefficientStandardErrors)
            print(" Tvalues :\n" + str(training_summary.tValues))
            T_values = str(training_summary.tValues)
            print(" p values :\n" + str(training_summary.pValues))
            P_values = str(training_summary.pValues)

            intercept_t = float(intercept_t)
            coefficientList = regressor.coefficients
            equation = label, '=', intercept_t, '+'
            for feature, coeff in zip(feature_colm, coefficientList):
                coeffFeaure = coeff, '*', feature, '+'
                equation += coeffFeaure
            equation = equation[:-1]
            print(equation)
            st = list(equation)

            # significance value

            '''
                            PValue = str(P_values)
                for pValue in PValue:
                    print(pValue)
                
                for pValue in PValue:
                    if (0 <= pValue < 0.001):
                        print(pValue, '***')
                    if (0.001 <= pValue < 0.01):
                        print(pValue, '**')
                    if (0.01 <= pValue < 0.05):
                        print(pValue, '*')
                    if (0.05 <= pValue < 0.1):
                        print(pValue, '.')
                    if (0.1 <= pValue < 1):
                        print(pValue, ' ')
            '''


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

            pred_residuals.write.parquet('hdfs://10.171.0.181:9000/dev/dmxdeepinsight/datasets/Q_Q_PLOT.parquet',
                                         mode='overwrite')


            #################################################################################3
            # scale location plot
            from pyspark.sql.functions import sqrt

            from pyspark.sql.functions import abs as ab

            df_label = prediction_data.select(label, 'prediction',
                                              sqrt(ab(prediction_data[label])).alias("sqrt_label"))

            df_label.show()
            df_sqrt_label_index = df_label.withColumn('row_index', f.monotonically_increasing_id())
            df_sqrt_label_index.show()

            res_d.show()
            sqrt_label_residual_join = df_sqrt_label_index.join(res_d, on=['row_index']).sort('row_index').drop(
                'row_index')

            sqrt_label_residual_join.show()

            std_resid = sqrt_label_residual_join.select('sqrt_label', 'prediction', (
                    sqrt_label_residual_join['residuals'] / sqrt_label_residual_join['sqrt_label']).alias(
                'std_res'))

            std_resid.show()

            sqrt_std_res = std_resid.select("std_res", 'prediction',
                                            sqrt(ab(std_resid["std_res"])).alias("sqrt_std_resid"))

            sqrt_std_res.show()
            sqrt_std_res_fitted = sqrt_std_res.select('prediction', 'sqrt_std_resid')

            sqrt_std_res_fitted.write.parquet(
                'hdfs://10.171.0.181:9000/dev/dmxdeepinsight/datasets/scale_location_train.parquet',
                mode='overwrite')

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

            # scale location plot

            # for scale location plot
            # from pyspark.sql.functions import udf
            #
            # def std_res(x):
            #     res_list = []
            #     res_list.append(x)
            #
            # std_residuals = udf(lambda y: std_res(y), FloatType())
            #
            # residuals_std = residuals.withColumn('residuals', std_residuals(col('residuals').cast(FloatType())))
            #
            # import statistics
            # import numpy as np
            # residuals_panda = residuals.toPandas()
            # # residuals_panda.residuals = range(residuals_panda.shape[1])
            # residuals_panda = residuals_panda.values
            # print(residuals_panda)
            # stdev_training = statistics.stdev(residuals_panda)
            # print(stdev_training)

            ############################################################################################################

            # creating the dictionary for storing the result

            # json_response = coefficient_t

            # print(json_response)

            # json_response = {"adjusted r**2 value" : training_summary.r2adj}

            # DATA VISUALIZATION PART

            # finding the quantile in the dataset(Q_Q plot)
            import matplotlib.pyplot as plt

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
            #
            # Q_label_pred=''
            # print(len(quantile_label))
            # length = len(quantile_label)
            #
            # for i in range(0,len(quantile_label)):
            #     Q_label_pred += str(quantile_label[i]) + '|'  +  str(quantile_prediction[i]) + '\n'

            # writing it to the hdfs in parquet file
            #
            # quantile_label_tospark = spark.createDataFrame(quantile_label, FloatType())
            # quantile_label_tospark = quantile_label_tospark.withColumnRenamed("value", "Q_label")
            #
            # quantile_prediction_tospark = spark.createDataFrame(quantile_prediction, FloatType())
            # quantile_prediction_tospark = quantile_prediction_tospark.withColumnRenamed("value", "Q_prediction")
            #
            # quant_label = quantile_label_tospark.withColumn('row_index', f.monotonically_increasing_id())
            # quant_predtiction = quantile_prediction_tospark.withColumn('row_index', f.monotonically_increasing_id())
            #
            # final_quantile = quant_label.join(quant_predtiction,on=['row_index']).sort('row_index').drop('row_index')
            #
            # final_quantile.show()
            #
            # final_quantile.write.parquet('hdfs://10.171.0.181:9000/dev/dmxdeepinsight/datasets/Q_Q_PLOT.parquet',mode='overwrite')
            #
            #

            # print(str(Q_label_pred[i]))

            # with open('Q_Q_plot.csv', 'w') as Q_Q:
            #     writer_Q_Q = csv.writer(Q_Q)
            #     writer_Q_Q.writerows((quantile_label, quantile_prediction))
            #
            # plt.scatter(quantile_label, quantile_prediction)
            # plt.show()

            ## finding the residual vs fitted graph data

            #
            #
            # prediction_val_pand_predict_tospark = spark.createDataFrame(prediction_val_pand_predict, FloatType())
            # prediction_val_pand_predict_tospark = prediction_val_pand_predict_tospark.withColumnRenamed("value", "prediction")
            #
            # prediction_val_pand_residual_tospark = spark.createDataFrame(prediction_val_pand_residual, FloatType())
            # prediction_val_pand_residual_tospark = prediction_val_pand_residual_tospark.withColumnRenamed("value", "residual")
            #
            # pred_spark = prediction_val_pand_predict_tospark.withColumn('row_index', f.monotonically_increasing_id())
            # res_spark = prediction_val_pand_residual_tospark.withColumn('row_index', f.monotonically_increasing_id())
            #
            # final_res_fitted = pred_spark.join(res_spark, on=['row_index'])\
            #     .sort('row_index').drop('row_index')
            #
            # final_res_fitted.show()
            #
            # final_res_fitted.write.parquet('hdfs://10.171.0.181:9000/dev/dmxdeepinsight/datasets/RESIDUAL_FITTED_PLOT.parquet',
            #                              mode='overwrite')
            #

            # plt.scatter(prediction_val_pand_predict, prediction_val_pand_residual)
            # plt.axhline(y=0.0, color="red")
            # plt.xlabel("prediction")
            # plt.ylabel("residual")
            # plt.title("residual vs fitted ")
            # plt.show()

            # creating the csv file and writitng into it

            fitted_residual = ''
            print(len(prediction_val_pand_residual))
            length = len(prediction_val_pand_residual)

            for i in range(0, len(prediction_val_pand_residual)):
                fitted_residual += str(prediction_val_pand_predict[i]) + '|' + str(
                    prediction_val_pand_residual[i]) + '\n'

            with open('residual_vs_fitted.csv', 'w') as r_f:
                writer_r_f = csv.writer(r_f)
                writer_r_f.writerows((prediction_val_pand_predict, prediction_val_pand_residual))

            # parquet file writing

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
            sqrt_residual = []
            for x in prediction_val_pand_residual_abs:
                sqrt_residual.append(math.sqrt(x))
                # print ("____________________  ",x)

            sqrt_residual

            # plt.scatter(sqrt_residual, prediction_val_pand_predict)
            ####################################################################################3

            # calculating std deviation
            import statistics

            print(statistics.stdev(prediction_val_pand_residual))
            stdev_pred = statistics.stdev(prediction_val_pand_residual)
            # mean = statistics.mean(prediction_val_pand_residual)

            # calcuate stnd residuals
            std_res = []
            for x in prediction_val_pand_residual:
                std_res.append(x / stdev_pred)
            print(std_res)

            # calculating the square root of std_res
            import math
            sqr_std_res = []
            for x in std_res:
                sqr_std_res.append(math.sqrt(abs(x)))
            print(sqr_std_res)
            #######################################################################################3
            # QUANTILE

            ## sort the list
            sorted_std_res = sorted(std_res)
            print(sorted_std_res)
            #
            mean = statistics.mean(sorted_std_res)
            stdev = statistics.stdev(sorted_std_res)
            print(mean)
            quantile = []
            n = len(sorted_std_res)
            print(n)
            for x in range(0, n):
                quantile.append((x - 0.5) / (n))

            print(quantile)
            #
            # z_score theoritical
            from scipy.stats import norm

            z_theory = []
            for x in quantile:
                z_theory.append((norm.ppf(abs(x))))
            print(z_theory)
            # z score for real val
            z_pract = []
            for x in sorted_std_res:
                z_pract.append((x - mean) / stdev)

            #

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

            Q_label_pred = ''
            print(len(quantile_label))
            length = len(quantile_label)
            for quant, val in zip(z_theory, z_pract):
                Q_label_pred += str(val) + 't' + str(quant) + 'n'

            plt.scatter(z_theory, z_pract)
            plt.savefig('q_q')

            ####################################################

            # creating the std residuals

            # square root of label
            sqrt_label = []
            for x in prediction_val_pand_label:
                sqrt_label.append(math.sqrt(abs(x)))

            sqrt_label
            prediction_val_pand_residual
            std_residual = []
            for sqr, resid in zip(sqrt_label, prediction_val_pand_residual):
                std_residual.append(resid / sqr)
                # print(std_sqrt_residual)

            # creating the std sqr root

            sqrt_std_residuals = []
            for x in std_residual:
                # print(math.sqrt(abs(x)))
                sqrt_std_residuals.append(math.sqrt(abs(x)))
            print(sqrt_std_residuals)

            # print(std_sqrt_residual)

            scale_predict_residual = ''
            for pre, res in zip(prediction_val_pand_predict, sqrt_std_residuals):
                scale_predict_residual += str(pre) + 't' + str(res) + 'n'
            print(scale_predict_residual)

            ##########################################################################
            # import math
            # sqrt_stdres = []
            # for x in std_sqrt_residual:
            #     sqrt_stdres.append(math.sqrt(x))
            #
            # scale_predict_residual = ''
            # for pre, res in zip(prediction_val_pand_predict, sqrt_stdres):
            #     scale_predict_residual += str(pre) + 't' + str(res) + 'n'
            # print(scale_predict_residual)

            ###################################3

            # plt.show()

            # scale_predict_residual=''
            #
            # print(len(sqrt_residual))
            # length = len(sqrt_residual)
            #
            # for i in range(0, len(std_sqrt_residual)):
            #     scale_predict_residual += str(prediction_val_pand_predict[i]) + '|' + str(std_sqrt_residual[i]) + '\n'

            # with open('scale_location_plot.csv', 'w') as s_l:
            #     writer_s_l = csv.writer(s_l)
            #     writer_s_l.writerows((prediction_val_pand_predict, sqrt_residual))

            # writing to the parquet

            # prediction_val_pand_predict_tospark = spark.createDataFrame(prediction_val_pand_predict, FloatType())
            # prediction_val_pand_predict_tospark = prediction_val_pand_predict_tospark.withColumnRenamed("value",
            #                                                                                             "prediction")
            #
            # sqrt_residual_tospark= spark.createDataFrame(sqrt_residual, FloatType())
            # sqrt_residual_tospark = sqrt_residual_tospark.withColumnRenamed("value",
            #                                                                                               "sqrt_residual")
            #
            # pred_spark = prediction_val_pand_predict_tospark.withColumn('row_index', f.monotonically_increasing_id())
            # res_spark = sqrt_residual_tospark.withColumn('row_index', f.monotonically_increasing_id())
            #
            # final_scale_fitted = pred_spark.join(res_spark,on=['row_index']) \
            #     .sort('row_index').drop('row_index')
            #
            # final_scale_fitted.show()
            #
            # final_scale_fitted.write.parquet(
            #     'hdfs://10.171.0.181:9000/dev/dmxdeepinsight/datasets/SCALE_LOCATION_PLOT.parquet',
            #     mode='overwrite')
            #

            # dumping the dictionary into json object

            # json_response = {'run_status': 'success', 'PredictiveResponse': resultdf}

            json_response = {

                "Intercept": intercept_t,
                "Coefficients": coefficient_t,
                "RMSE": RMSE,
                "MSE": MSE,
                "R_square": r_square,
                "Adj_R_square": adjsted_r_square,
                "Coefficient_error": coefficientStdError,
                "T_value": T_values,
                "P_value": P_values,
                'Q_Q_plot': Q_label_pred,
                'residual_fitted': fitted_residual,
                'scale_location': scale_predict_residual

            }

            return json_response


        except Exception as e:
            print('exception is =' + str(e))
