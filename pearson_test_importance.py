from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler

def Correlation_test_imp(dataset, features, label_col):
    spark = SparkSession.builder.appName("predictive_analysis").master(
        "spark://fidel-Latitude-E5570:7077").getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    label_col = [label_col]

    All_colms =  label_col + features

    featureassembler_correlation = VectorAssembler(
        inputCols=All_colms, outputCol="correlation_colm")
    output_corr = featureassembler_correlation.transform(dataset)
    output_corr.show()

    finalized_corr = output_corr.select("correlation_colm")
    finalized_corr.show()
    from pyspark.ml.stat import Correlation

    r1p = Correlation.corr(output_corr, "correlation_colm").head()
    print("pearson correlation matrix : \n : " + str(r1p[0]))
    pearson_value = []

    for x in r1p[0].toArray():
        pearson_value.append(x[0])

    print(pearson_value)

    #
    # r1s = Correlation.corr(output_corr, "correlation_colm", "spearman").head()
    # print(" spearman correlation...: \n" + str(r1s[0]))

    result_pearson = {'pearson value : ': pearson_value}
    # print(json_response)

    return result_pearson
