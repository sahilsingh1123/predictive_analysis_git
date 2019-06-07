import pyspark.sql.functions as f
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pyspark.sql.types import *

from PredictionAlgorithms.relationship import Relationship

# from pyspark.sql.functions import max as _max
# from pyspark.sql.functions import abs

#
# os.environ['PYSPARK_SUBMIT_ARGS'] = '--jars /home/fidel/software/spark-2.4.0-bin-hadoop2.7/jars/xgboost4j-spark-0.72.jar,/home/fidel/software/spark-2.4.0-bin-hadoop2.7/jars/xgboost4j-0.72.jar pyspark-shell'
#


spark = SparkSession.builder.appName("predictive_Analysis").master("local[*]").getOrCreate()
# spark.sparkContext.addPyFile('/home/fidel/Downloads/sparkxgb.zip')
spark.sparkContext.setLogLevel("ERROR")


# spark.sparkContext.addPyFile('/home/fidel/Downloads/xgboost4j-0.72.jar')
# spark.sparkContext.addPyFile('/home/fidel/Downloads/xgboost4j-spark-0.72.jar')


class Ridge_reg():
    def __init__(self, xt=[0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 0.005, 0.8, 0.3]):
        self.xt = xt

    def ridge(self, dataset_add, feature_colm, label_colm, relation_list, relation):

        Rsqr_list = []
        Rsqr_regPara = {}
        print(self.xt)
        # print(data_add)

        try:
            dataset = spark.read.parquet(dataset_add)
            dataset.show()

            label = ''
            for y in label_colm:
                label = y

            print(label)
            if relation=='linear':
                print('linear relationship')
            if relation=='non_linear':
                dataset = Relationship(dataset, relation_list)
            dataset.show()
            featureassembler = VectorAssembler(inputCols=feature_colm,
                                               outputCol="Independent_features")
            output = featureassembler.transform(dataset)
            output.show()
            output.select("Independent_features").show()
            finalized_data = output.select("Independent_features", label)
            finalized_data.show()

            # splitting the dataset into taining and testing

            train_data, test_data = finalized_data.randomSplit([0.75, 0.25], seed=40)

            ######################################################################33

            for t in self.xt:
                lr1 = LinearRegression(featuresCol="Independent_features", labelCol=label, elasticNetParam=0,
                                       regParam=t)
                regressor1 = lr1.fit(train_data)
                print(t)
                print("coefficient : " + str(regressor1.coefficients))
                reg_sum = regressor1.summary
                r2 = reg_sum.r2
                Rsqr_list.append(r2)
                Rsqr_regPara[r2] = t
                print(r2)

            print(Rsqr_list)
            print(max(Rsqr_list))
            maximum_rsqr = max(Rsqr_list)
            print(Rsqr_regPara)
            final_regPara = []

            for key, val in Rsqr_regPara.items():
                if (key == maximum_rsqr):
                    print(val)
                    final_regPara.append(val)

            for reg in final_regPara:
                lr_lasso = LinearRegression(featuresCol="Independent_features", labelCol=label, elasticNetParam=0,
                                            regParam=reg)
                regressor = lr_lasso.fit(train_data)
                training_summary = regressor.summary
                r2 = training_summary.r2
                print(r2)

            print("coefficient : " + str(regressor.coefficients))
            coefficient_t = str(regressor.coefficients)

            print("intercept : " + str(regressor.intercept))
            intercept_t = str(regressor.intercept)
            prediction = regressor.evaluate(test_data)

            prediction_val = prediction.predictions
            prediction_val.show()
            prediction_val_pand = prediction_val.select(label, "prediction").toPandas()
            prediction_val_pand = prediction_val_pand.assign(
                residual_vall=prediction_val_pand[label] - prediction_val_pand["prediction"])

            prediction_val_pand_residual = prediction_val_pand["residual_vall"]
            prediction_val_pand_label = prediction_val_pand[label]
            prediction_val_pand_predict = prediction_val_pand["prediction"]
            lr_prediction = regressor.transform(test_data)
            lr_prediction.groupBy(label, "prediction").count().show()
            lr_prediction_quantile = lr_prediction.select(label, "prediction")
            lr_prediction_onlypred = lr_prediction.select('prediction')

            # training_summary = regressor.summary

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
            coefficientStdError = str(training_summary.coefficientStandardErrors)
            print(" Tvalues :\n" + str(training_summary.tValues))
            T_values = str(training_summary.tValues)
            tValuesList = training_summary.tValues
            print(" p values :\n" + str(training_summary.pValues))
            P_values = str(training_summary.pValues)

            # regression equation
            intercept_t = float(intercept_t)
            coefficientList = list(regressor.coefficients)
            equation = label, '=', intercept_t, '+'
            for feature, coeff in zip(feature_colm, coefficientList):
                coeffFeature = coeff, '*', feature, '+'
                equation += coeffFeature
            equation = equation[:-1]
            print(equation)
            equationAsList = list(equation)

            # significance value

            PValuesList = training_summary.pValues
            significanceObject = {}

            for pValue in PValuesList:
                if (0 <= pValue < 0.001):
                    significanceObject[pValue] = '***'
                if (0.001 <= pValue < 0.01):
                    significanceObject[pValue] = '**'
                if (0.01 <= pValue < 0.05):
                    significanceObject[pValue] = '*'
                if (0.05 <= pValue < 0.1):
                    significanceObject[pValue] = '.'
                if (0.1 <= pValue < 1):
                    significanceObject[pValue] = '-'
            print(significanceObject)

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

            # pred_residuals.write.parquet('hdfs://10.171.0.181:9000/dev/dmxdeepinsight/datasets/Q_Q_PLOT.parquet',
            #                              mode='overwrite')


            #################################################################################3
            # scale location plot
            from pyspark.sql.functions import abs as ab, sqrt, mean as meann, stddev as stdDev

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

            target = dataset.select(label)
            pred = prediction_data.select(['prediction'])
            pred_d = pred.withColumn('row_index', f.monotonically_increasing_id())
            target_d = target.withColumn('row_index', f.monotonically_increasing_id())

            pred_target = pred_d.join(target_d, on=['row_index']).drop('row_index')
            pred_target.show()

            dataset.show()

            pred_target_data_update = dataset.join(pred_target, on=[label])

            pred_target_data_update.show(100)
            table_response = {

                "Intercept": intercept_t,
                "Coefficients": coefficient_t,
                "RMSE": RMSE,
                "MSE": MSE,
                "R_square": r_square,
                "Adj_R_square": adjsted_r_square,
                "coefficientStdError": coefficientStdError,
                "T_value": T_values,
                "P_value": P_values

            }
            y = 0.1
            x = []

            for i in range(0, 90):
                x.append(y)
                y = round(y + 0.01, 2)
            quantile_label = lr_prediction_quantile.approxQuantile(label, x, 0.01)
            quantile_prediction = lr_prediction_quantile.approxQuantile("prediction", x, 0.01)
            Q_label_pred=''
            print(len(quantile_label))
            length = len(quantile_label)

            for i in range(0,len(quantile_label)):
                Q_label_pred += str(quantile_label[i]) + 't'  +  str(quantile_prediction[i]) + 'n'
            import math

            fitted_residual = ''
            print(len(prediction_val_pand_residual))
            length = len(prediction_val_pand_residual)

            for i in range(0, len(prediction_val_pand_residual)):
                fitted_residual += str(prediction_val_pand_predict[i]) + 't' + str(prediction_val_pand_residual[i]) + 'n'
            ## scale location graph data
            import math

            prediction_val_pand_residual
            prediction_val_pand_predict
            prediction_val_pand_residual_abs = prediction_val_pand_residual.abs()
            import math
            sqrt_residual = []
            for x in prediction_val_pand_residual_abs:
                sqrt_residual.append(math.sqrt(x))
                # print ("____________________  ",x)

            sqrt_residual
            # calculating std deviation
            import statistics

            print(statistics.stdev(prediction_val_pand_residual))
            stdev_ = statistics.stdev(prediction_val_pand_residual)

            # calcuate stnd residuals
            std_res = []
            for x in prediction_val_pand_residual:
                std_res.append(x / stdev_)
            print(std_res)

            # calculating the square root of std_res
            import math
            sqr_std_res = []
            for x in std_res:
                sqr_std_res.append(math.sqrt(abs(x)))
            print(sqr_std_res)

            scale_predict_residual = ''
            for pre, res in zip(prediction_val_pand_predict, sqr_std_res):
                scale_predict_residual += str(pre) + 't' + str(res) + 'n'
            print(scale_predict_residual)
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
            from scipy.stats import norm

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
            for quant,val in zip(z_theory,z_pract):
                Q_label_pred += str(quant) + 't' + str(val) + 'n'
            graph_response = {
                "Q_Q_plot": Q_label_pred,
                "residual_fitted": fitted_residual,
                "scale_location": scale_predict_residual
            }

            tableContent = \
                {
                    'coefficientValuesKey': coefficientList,
                    'tValuesKey': tValuesList,
                    'pValuesKey': PValuesList,
                    'significanceValuesKey': significanceObject,
                    'interceptValuesKey': intercept_t,
                    "RMSE": RMSE,
                    "RSquare": r_square,
                    "AdjRSquare": adjsted_r_square,
                    "CoefficientStdError": coefficientStdError,
                    'equationKey' : equation
                }
            print(tableContent)

            json_response = {

                'table_data': table_response,
                'graph_data': graph_response


            }
            print(json_response)
            return (json_response)
        except Exception as e:
            print('exception is =' + str(e))
