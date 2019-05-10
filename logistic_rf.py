from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import RFormula
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
from pyspark.ml.classification import BinaryLogisticRegressionTrainingSummary

spark = SparkSession.builder.appName("predictive_Analysis").master("local[*]").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

dataset_add = "/home/fidel/mltest/bank.csv"
feature_colm = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']
label = "y"


class logistic_reg:
    def Logistic_regression(dataset_add, feature_colm, label_colm):

        dataset = spark.read.csv(dataset_add, header=True, inferSchema=True, sep=";")

        dataset.show()

        dataset.groupBy("y").count().show()

        label = ''
        for y in label_colm:
            label = y


        f = ""
        f = label + " ~ "

        for x in feature_colm:
            f = f + x + "+"
        f = f[:-1]
        f = (f)

        formula = RFormula(formula=f,
                           featuresCol="features",
                           labelCol="label")

        output = formula.fit(dataset).transform(dataset)

        finalized_data = output.select("features", "label")

        finalized_data.show()



        train_data, test_data = finalized_data.randomSplit([0.75, 0.25], seed = 40)

        Accuracy_list = []

        FPR_list = []
        TPR_list = []
        precision_list = []
        recall_list = []
        lr = LogisticRegression(maxIter=5)
        lrModel = lr.fit(train_data)


        print ("coefficients:" + str(lrModel.coefficientMatrix))
        print("intercept: " + str(lrModel.interceptVector))
        training_summary = lrModel.summary
        BinaryLogisticRegressionTrainingSummary.accuracy
        print (" area under roc : " , training_summary.areaUnderROC)
        print ("  roc : " , training_summary.roc)
        roc = training_summary.roc
        roc.show()
        roc.write.parquet('hdfs://10.171.0.181:9000/dev/dmxdeepinsight/datasets/ROC_plot.parquet',
                          mode='overwrite')
        print (" pr value : " , training_summary.pr)
        pr = training_summary.pr
        pr.show()
        pr.write.parquet('hdfs://10.171.0.181:9000/dev/dmxdeepinsight/datasets/PR_plot.parquet',
                          mode='overwrite')
        print (" precision by threshold : " , training_summary.precisionByThreshold)
        prec_by_threshold = training_summary.precisionByThreshold
        prec_by_threshold.show()
        print (" accuracy : ", training_summary.accuracy)
        accuracy_d = training_summary.accuracy
        print (accuracy_d)
        fMeasure = training_summary.fMeasureByThreshold
        fMeasure.show()
        maxFMeasure = fMeasure.groupBy().max('F-Measure').select('max(F-Measure)').head()
        bestThreshold = fMeasure.where(fMeasure['F-Measure'] == maxFMeasure['max(F-Measure)']) \
            .select('threshold').head()['threshold']
        lr.setThreshold(bestThreshold)
        objectiveHistory = training_summary.objectiveHistory
        print ("objectiveHistory")
        for objective in objectiveHistory:
            print (objective)
        print ("false positive rate by label:")
        for i, rate in enumerate(training_summary.falsePositiveRateByLabel):
            print ("label %d: %s" % (i, rate))
        print("True positive rate")
        for i, rate in enumerate(training_summary.truePositiveRateByLabel):
            print ("label %d : %s" % (i, rate))
        print("Precision by label:")
        for i, prec in enumerate(training_summary.precisionByLabel):
            print("label %d: %s" % (i, prec))
        print("Recall by label:")
        for i, rec in enumerate(training_summary.recallByLabel):
            print("label %d: %s" % (i, rec))
        print("F-measure by label:")
        for i, f in enumerate(training_summary.fMeasureByLabel()):
            print("label %d: %s" % (i, f))
        accuracy = training_summary.accuracy
        falsePositiveRate = training_summary.weightedFalsePositiveRate
        truePositiveRate = training_summary.weightedTruePositiveRate
        fMeasure = training_summary.weightedFMeasure()
        precision = training_summary.weightedPrecision
        recall = training_summary.weightedRecall
        print("Accuracy: %s\nFPR: %s\nTPR: %s\nF-measure: %s\nPrecision: %s\nRecall: %s"
              % (accuracy, falsePositiveRate, truePositiveRate, fMeasure, precision, recall))
        Accuracy_list.append(accuracy)
        FPR_list.append(falsePositiveRate)
        TPR_list.append(truePositiveRate)
        precision_list.append(precision)
        recall_list.append(recall)
        print (Accuracy_list)
        print (FPR_list)
        print (TPR_list)
        print (precision_list)
        print (recall_list)
        fpr = roc.select("FPR").toPandas()
        tpr = roc.select("TPR").toPandas()
        plt.plot(fpr, tpr)
        plt.show()
        pr_recall = pr.select("recall").toPandas()
        pr_precision = pr.select("precision").toPandas()
        plt.plot(pr_precision,pr_recall)
        plt.show()
        prediction_val = lrModel.transform(test_data)
        prediction_val.groupBy("label", "prediction").count().show()
        prediction_val.show()
        prediction_val.groupBy("prediction").count().show()
        prediction_val.groupBy("prediction", "probability").count().show()





    # Logistic_regression(dataset_add, feature_colm, label)
