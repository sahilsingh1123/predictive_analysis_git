from PredictionAlgorithms.pearson_test_importance import Correlation_test_imp
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

from PredictionAlgorithms.relationship import Relationship

spark = SparkSession.builder.appName("predictive analysis").master("local[*]").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

def randomClassifier(dataset_add, feature_colm, label_colm,relation_list, relation):
    try:
        # dataset = spark.read.parquet(dataset_add)
        dataset = spark.read.csv(dataset_add, header=True, inferSchema=True)

        dataset.show()

        label = ''
        for y in label_colm:
            label = y

        print(label)

        summaryList = ['mean', 'stddev', 'min', 'max']
        summaryDict = {}
        for colm in feature_colm:
            summaryListTemp = []
            for value in summaryList:
                summ = list(dataset.select(colm).summary(value).toPandas()[colm])
                summaryListTemp.append(summ)
            summaryDict[colm] = summaryListTemp
        summaryDict['summaryName'] = summaryList

        # print(summaryDict)
        varianceDict = {}
        for colm in feature_colm:
            varianceListTemp = list(dataset.select(variance(col(colm)).alias(colm)).toPandas()[colm])
            varianceDict[colm] = varianceListTemp
        # print(varianceDict)

        summaryAll = {'summaryDict': summaryDict, 'varianceDict': varianceDict}
        print(summaryAll)

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

        if relation == 'linear':
            dataset = dataset
        if relation == 'non_linear':
            dataset = Relationship(dataset, relation_list)


        # calling pearson test fuction

        response_pearson_test = Correlation_test_imp(dataset=dataset, features = integer_features, label_col= label)


        # dataset = dataset.withColumnRenamed(label , 'indexed_'+ label)


        # dataset_pearson = dataset

        #
        # label_indexer = StringIndexer(inputCol=label, outputCol='indexed_'+label).fit(dataset)
        # dataset = label_indexer.transform(dataset)



        ###########################################################################
        indexed_features = []
        encoded_features = []
        for colm in string_features:
            indexer = StringIndexer(inputCol=colm, outputCol='indexed_' + colm).fit(dataset)
            indexed_features.append('indexed_'+colm)
            dataset = indexer.transform(dataset)
            # dataset.show()
            # encoder = OneHotEncoderEstimator(inputCols=['indexed_'+colm], outputCols=['encoded_'+colm]).fit(dataset)
            # encoded_features.append('encoded_'+colm)
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

        # output.show()
        # output.select("features").show()

        # output_features = dataset.select("features")



        #using the vector indexer

        vec_indexer = VectorIndexer(inputCol='features', outputCol='vec_indexed_features', maxCategories=4).fit(dataset)

        categorical_features = vec_indexer.categoryMaps
        print("Chose %d categorical features: %s" %
                    (len(categorical_features), ", ".join(str(k) for k in categorical_features.keys())))

        vec_indexed = vec_indexer.transform(dataset)
        vec_indexed.show()


        # preparing the finalized data

        finalized_data = vec_indexed.select(label, 'vec_indexed_features')
        finalized_data.show()





        # renaming the colm
        # print (label)
        # dataset.withColumnRenamed(label,"label")
        # print (label)
        # dataset.show()

        # f = ""
        # f = label + " ~ "
        #
        # for x in features:
        #     f = f + x + "+"
        # f = f[:-1]
        # f = (f)
        #
        # formula = RFormula(formula=f,
        #                    featuresCol="features",
        #                    labelCol="label")
        #
        # output = formula.fit(dataset).transform(dataset)
        #
        # output_2 = output.select("features", "label")
        #
        # output_2.show()
        #
        #
        #
        # splitting the dataset into taining and testing

        train_data, test_data = finalized_data.randomSplit([0.75, 0.25], seed=40)


        rf=RandomForestRegressor(labelCol=label,featuresCol='vec_indexed_features',numTrees=10)

        # Convert indexed labels back to original labels.

        # Train model.  This also runs the indexers.
        model = rf.fit(train_data)

        # Make predictions.
        predictions = model.transform(test_data)

        # Select example rows to display.
        # predictions.select("prediction", "label", "features").show(10)

        print(model.featureImportances)
        feature_importance = model.featureImportances.toArray().tolist()
        print(feature_importance)


        features_column_for_user = integer_features + string_features

        feature_imp = { 'feature_importance': feature_importance,"feature_column" : features_column_for_user}


        response_dict = {
            'feature_importance': feature_imp,
            'pearson_test_data': response_pearson_test
        }

        return response_dict
        print(response_dict)



        # Select (prediction, true label) and compute test error
        # evaluator = MulticlassClassificationEvaluator(
        #     labelCol="label", predictionCol="prediction", metricName="accuracy")
        # accuracy = evaluator.evaluate(predictions)
        # print("Test Error = %g" % (1.0 - accuracy))

        # rfModel = model.stages[2]
        # print(rfModel)  # summary only






    except Exception as e :
        print("exception is  = " + str(e))
#
# if __name__== "__main__":
#     randomClassifier(dataset_add, features, label)

