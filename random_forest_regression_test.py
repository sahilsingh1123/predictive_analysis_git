from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import RFormula
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator




# if __name__== "__main__":
spark = SparkSession.builder.appName("predictive analysis").master("local[*]").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")



dataset_add = "/home/fidel/mltest/bank.csv"
features = ['balance', 'day', 'duration', 'campaign','job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']
# features = ['age', 'balance', 'day', 'duration', 'campaign']
label = ["age"]

def randomClassifier(dataset_add, feature_colm, label_colm):
    try:
        dataset = spark.read.csv(dataset_add,  header=True, inferSchema=True, sep=";")

        dataset.show()

        label = ''
        for y in label_colm:
            label = y

        print(label)


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




        dataset = dataset.withColumnRenamed(label , 'indexed_'+ label)




        #
        # label_indexer = StringIndexer(inputCol=label, outputCol='indexed_'+label).fit(dataset)
        # dataset = label_indexer.transform(dataset)



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

        finalized_data = vec_indexed.select('indexed_'+label, 'vec_indexed_features')
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


        rf=RandomForestRegressor(labelCol='indexed_'+label,featuresCol='vec_indexed_features',numTrees=10)

        # Convert indexed labels back to original labels.

        # Train model.  This also runs the indexers.
        model = rf.fit(train_data)

        # Make predictions.
        predictions = model.transform(test_data)

        # Select example rows to display.
        # predictions.select("prediction", "label", "features").show(10)

        print(model.featureImportances)



        # Select (prediction, true label) and compute test error
        # evaluator = MulticlassClassificationEvaluator(
        #     labelCol="label", predictionCol="prediction", metricName="accuracy")
        # accuracy = evaluator.evaluate(predictions)
        # print("Test Error = %g" % (1.0 - accuracy))

        # rfModel = model.stages[2]
        # print(rfModel)  # summary only






    except Exception as e :
        print("exception is  = " + str(e))

if __name__== "__main__":
    randomClassifier(dataset_add, features, label)

