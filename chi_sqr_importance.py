from pyspark.sql import SparkSession
from pyspark.ml.feature import RFormula
from pyspark.ml.stat import ChiSquareTest
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer, VectorAssembler
from pyspark.ml.feature import ChiSqSelector


def chi_square_test(dataset,features, label_col):
    spark = SparkSession.builder.appName("predictive_analysis").master("local[*]").getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    length=features.__len__()

    featureassembler = VectorAssembler(
        inputCols=features,
        outputCol="features")

    dataset = featureassembler.transform(dataset)
    dataset.show()

    vec_indexer = VectorIndexer(inputCol='features', outputCol='vec_indexed_features', maxCategories=4).fit(dataset)

    categorical_features = vec_indexer.categoryMaps
    print("Chose %d categorical features: %s" %
          (len(categorical_features), ", ".join(str(k) for k in categorical_features.keys())))

    vec_indexed = vec_indexer.transform(dataset)
    vec_indexed.show()

    finalized_data = vec_indexed.select(label_col, 'vec_indexed_features')
    finalized_data.show()


    # using chi selector
    selector = ChiSqSelector(numTopFeatures=length, featuresCol="vec_indexed_features", outputCol="selected_features",
                             labelCol=label_col)

    result = selector.fit(finalized_data).transform(finalized_data)

    print("chi2 output with top %d features selected " % selector.getNumTopFeatures())
    result.show()

    # runnin gfor the chi vallue test

    r = ChiSquareTest.test(result, "selected_features", label_col).head()
    print("pValues: " + str(r.pValues))
    p_values = str(r.pValues)
    print("degreesOfFreedom: " + str(r.degreesOfFreedom))

    print("statistics: " + str(r.statistics))

    return_data = {'pvalues': p_values}

    return return_data


