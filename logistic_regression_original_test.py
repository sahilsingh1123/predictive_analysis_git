from pyspark.ml.classification import LogisticRegression
# from pyspark.ml.feature import RFormula
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler, VectorIndexer
# if __name__=="__main__":
spark = SparkSession.builder.appName("predictive_Analysis").master("local[*]").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")


dataset_add = "/home/fidel/mltest/bank.csv"
features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']
label = "y"


class logistic_reg:
    def Logistic_regression(dataset_add, feature_colm, label_colm):

        dataset = spark.read.csv(dataset_add, header=True, inferSchema=True, sep=";")

        dataset.show()

        dataset.groupBy("y").count().show()
        label = ''
        for y in label_colm:
            label = y

        print(label)

        # using the rformula for indexing, encoding and vectorising

        # f = ""
        # f = label + " ~ "
        #
        # for x in features:
        #     f = f + x + "+"
        # f = f[:-1]
        # f = (f)


        # extracting the schema

        val = dataset.schema

        string_features = []
        integer_features = []

        for x in val:
            if (str(x.dataType) == "StringType" ):
                for y in feature_colm:
                    if x.name == y:
                        string_features.append(x.name)
            else:
                for y in feature_colm:
                    if x.name == y:
                        integer_features.append(x.name)

        print(string_features)
        print(integer_features)
        print(val)
        # print(label)
        # label = 'y'

        for z in val:
            if (z.name == label and str(z.dataType) == "StringType"):
                label_indexer = StringIndexer(inputCol=label, outputCol='indexed_' + label).fit(dataset)
                dataset = label_indexer.transform(dataset)
            if (z.name == label and str(z.dataType) == ("IntegerType" or "FloatType" or "DoubleType")):
                dataset = dataset.withColumnRenamed(label,'indexed_'+label)


        ###########################################################################
        indexed_features = []
        encoded_features = []
        for col in string_features:
            indexer = StringIndexer(inputCol=col, outputCol='indexed_' + col).fit(dataset)
            indexed_features.append('indexed_'+col)
            dataset = indexer.transform(dataset)
            # dataset.show()
            # encoder = OneHotEncoderEstimator(inputCols=['indexed_'+col], outputCols=['encoded_'+col]).fit(dataset)
            # encoded_features.append('encoded_'+col)
            # dataset = encoder.transform(dataset)
            # dataset.show()

        print(indexed_features)
        print(encoded_features)

        # combining both the features colm together

        final_features = integer_features + indexed_features

        print(final_features)



        # now using the vector assembler


        featureassembler = VectorAssembler(
            inputCols=final_features,
            outputCol="features")

        dataset = featureassembler.transform(dataset)
        dataset.show()

        # combining both the features colm together


        # output.show()
        # output.select("features").show()

        # output_features = dataset.select("features")

        # using the vector indexer (for categorical data kind of one hot encoding)

        vec_indexer = VectorIndexer(inputCol='features', outputCol='vec_indexed_features', maxCategories=15).fit(dataset)

        categorical_features = vec_indexer.categoryMaps
        print("Chose %d categorical features: %s" %
              (len(categorical_features), ", ".join(str(k) for k in categorical_features.keys())))

        vec_indexed = vec_indexer.transform(dataset)
        vec_indexed.show()

        # preparing the finalized data

        finalized_data = vec_indexed.select('indexed_' + label, 'vec_indexed_features')
        finalized_data.show()

        # formula = RFormula(formula=f,
        #                    featuresCol="features",
        #                    labelCol="label")
        #
        # output = formula.fit(dataset).transform(dataset)
        #
        # output_2 = output.select("features", "label")
        #
        # output_2.show()

        # splitting the dataset into train and test

        train_data, test_data = finalized_data.randomSplit([0.75, 0.25], seed = 40)

        # implementing the logistic regression
        # lr1 =LogisticRegression()

        Accuracy_list = []
        # Accuracy_list.append(accuracy)
        FPR_list = []
        # FPR_list.append(falsePositiveRate)
        TPR_list = []
        precision_list = []
        recall_list = []






        y= 0.1
        # x=[]
        for i in range(0,3):
            y=round(y+0.1,2)

            lr = LogisticRegression(featuresCol='vec_indexed_features', labelCol='indexed_' + label,maxIter=5, regParam=0.1, elasticNetParam=1.0, threshold=0.3)



            # fit the model


            lrModel = lr.fit(train_data)
            lrModel

            # print the coefficients and the intercept for the logistic regression

            print ("coefficients:" + str(lrModel.coefficientMatrix))
            # mat = (lrModel.coefficientMatrix)
            # print mat
            print("intercept: " + str(lrModel.interceptVector))





            # getting the summary of the model

            # f-measure calculation
            from pyspark.ml.classification import BinaryLogisticRegressionTrainingSummary

            training_summary = lrModel.summary

            BinaryLogisticRegressionTrainingSummary.accuracy

            print (" area under roc : " , training_summary.areaUnderROC)
            print ("  roc : " , training_summary.roc)
            roc = training_summary.roc
            roc.show()
            print (" pr value : " , training_summary.pr)
            pr = training_summary.pr
            pr.show()
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

            # obtain the objective per iteration

            objectiveHistory = training_summary.objectiveHistory
            print ("objectiveHistory")
            for objective in objectiveHistory:
                print (objective)


            # for a multiclass we can inspect  a matrix on a per label basis

            print ("false positive rate by label:")
            for i, rate in enumerate(training_summary.falsePositiveRateByLabel):
                print ("label %d: %s" % (i, rate))


            print("True positive rate")
            for i, rate in enumerate(training_summary.truePositiveRateByLabel):
                print ("label %d : %s" % (i, rate))
            #
            # print("True Negative rate")
            # for i, rate in enumerate(training_summary)

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
            # Accuracy_list = []
            Accuracy_list.append(accuracy)
            # FPR_list = []
            FPR_list.append(falsePositiveRate)
            # TPR_list=[]
            TPR_list.append(truePositiveRate)
            precision_list.append(precision)
            recall_list.append(recall)

        print (Accuracy_list)
        print (FPR_list)
        print (TPR_list)
        print (precision_list)
        print (recall_list)

        import matplotlib.pyplot as plt
        #
        # plt.plot(recall_list, FPR_list)
        # plt.show()

        #
        # fpr = [0.0,0.0,0.0,0.0,0.003067484662576687, 0.003067484662576687, 0.006134969325153374, 0.11042944785276074, 0.1165644171779141, 0.1165644171779141, 0.23006134969325154, 0.9723926380368099, 0.9846625766871165 ]
        # tpr = [0.0, 0.09767441860465116, 0.10232558139534884, 0.13488372093023257 ,0.17674418604651163 ,0.3674418604651163 , 0.37209302325581395  , 0.7534883720930232, 0.8651162790697674 , 0.8697674418604651 , 0.9069767441860465, 0.9953488372093023, 1.0]
        # data visualization

        # ROC graph
        fpr = roc.select("FPR").toPandas()

        tpr = roc.select("TPR").toPandas()


        plt.plot(fpr, tpr)
        plt.show()


        # PR graph

        pr_recall = pr.select("recall").toPandas()
        pr_precision = pr.select("precision").toPandas()

        plt.plot(pr_precision,pr_recall)
        plt.show()


        # now applying the fit on the test data


        prediction_val = lrModel.transform(test_data)
        prediction_val.groupBy('indexed_' + label, "prediction").count().show()
        prediction_val.show()

        prediction_val.groupBy("prediction").count().show()

        prediction_val.groupBy("prediction", "probability").count().show()





    Logistic_regression(dataset_add, features, label)
