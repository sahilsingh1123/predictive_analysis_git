from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.feature import VectorIndexer, VectorAssembler
from pyspark.ml.stat import ChiSquareTest
from pyspark.sql import SparkSession


def chi_square_test(dataset,features, label_col,stringFeatures):
    spark = SparkSession.builder.appName("predictive_analysis").master("local[*]").getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    stringFeatures
    length=features.__len__()
    datasetChi = dataset

    featureassembler = VectorAssembler(
        inputCols=features,
        outputCol="features", handleInvalid="skip")

    datasetChi = featureassembler.transform(datasetChi)
    datasetChi.show()

    vec_indexer = VectorIndexer(inputCol='features', outputCol='vec_indexed_features', maxCategories=4, handleInvalid="skip").fit(datasetChi)

    categorical_features = vec_indexer.categoryMaps
    print("Chose %d categorical features: %s" %
          (len(categorical_features), ", ".join(str(k) for k in categorical_features.keys())))

    vec_indexed = vec_indexer.transform(datasetChi)
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
    p_values = list(r.pValues)
    PValues = []
    for val in p_values:
        PValues.append(round(val, 4))
    print(PValues)
    dof = list(r.degreesOfFreedom)
    stats = list(r.statistics)
    statistics = []
    for val in stats:
        statistics.append(round(val, 4))
    print(statistics)
    chiSquareDict = {}
    for pval, doF, stat, colm in zip(PValues, dof, statistics, stringFeatures):
        print(pval, doF, stat)
        chiSquareDict[colm] = pval, doF, stat
    chiSquareDict['summaryName'] = ['pValue','DoF','statistics']
    print(chiSquareDict)

    return_data = {'pvalues': chiSquareDict}


    return return_data


