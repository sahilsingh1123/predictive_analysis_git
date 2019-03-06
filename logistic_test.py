from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import RFormula
from pyspark.sql import SparkSession

# if __name__=="__main__":
spark = SparkSession.builder.appName("predictive_Analysis").master("local[*]").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

dataset_add = "/home/fidel/mltest/bank.csv"
features = ["default", "housing", "marital"]
label = "loan"


class logistic_reg:
    def Logistic_regression(dataset_add, features, label):

        dataset = spark.read.csv(dataset_add, header=True, inferSchema=True, sep=";")

        dataset.show()

        # using the rformula for indexing, encoding and vectorising

        f = ""
        f = label + " ~ "

        for x in features:
            f = f + x + "+"
        f = f[:-1]
        f = (f)

        formula = RFormula(formula=f,
                           featuresCol="features",
                           labelCol="label")

        output = formula.fit(dataset).transform(dataset)

        output_2 = output.select("features", "label")

        output_2.show()

        # implementing the logistic regression
        lr1 = LogisticRegression()

        lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.6, family="multinomial")


        # splitting the dataset

        train_data, test_data = output_2.randomSplit([0.75, 0.25], seed=40)

        # fit the model


        Model = lr.fit(train_data)


        lrModel = Model.transform(test_data)
        lrModel.groupBy("label", "prediction").count().show()

        # print the coefficients and the intercept for the logistic regression
        #
        # print ("coefficients:" + str(lrModel.coefficientMatrix))
        # # mat = (lrModel.coefficientMatrix)
        # # print mat
        # print("intercept: " + str(lrModel.interceptVector))

        # getting the summary of the model

        training_summary = lrModel.summary

        # obtain the objective per iteration

        objectiveHistory = training_summary.objectiveHistory
        print ("objectiveHistory")
        for objective in objectiveHistory:
            print objective

        # for a multiclass we can inspect  a matrix on a per label basis

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

    Logistic_regression(dataset_add, features, label)
