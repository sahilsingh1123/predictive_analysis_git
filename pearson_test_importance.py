from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession


def Correlation_test_imp(dataset, features, label_col):

    label_col = [label_col]

    All_colms =  label_col + features

    featureassembler_correlation = VectorAssembler(
        inputCols=All_colms, outputCol="correlation_colm", handleInvalid="skip")
    output_corr = featureassembler_correlation.transform(dataset)
    output_corr.show()

    finalized_corr = output_corr.select("correlation_colm")
    finalized_corr.show()
    from pyspark.ml.stat import Correlation

    r1p = Correlation.corr(output_corr, "correlation_colm").head()
    print("pearson correlation matrix : \n : " + str(r1p[0]))
    pearson_matrix = r1p[0].toArray().tolist()

    pearsonMatrix = []
    for everylist in pearson_matrix:
        insideList = []
        for listinlist in everylist:
            insideList.append(round(listinlist, 4))

        pearsonMatrix.append(insideList)

    print(pearsonMatrix)


    pearson_value_d = []

    for x in r1p[0].toArray():
        pearson_value_d.append(round(x[0],4))
        # pearson_value_d.append(x[0])

    print(pearson_value_d)

    pearson_value = {}
    for col,val in zip(All_colms,pearson_value_d):
        pearson_value[col] = val


    print(pearson_value)


    #
    # r1s = Correlation.corr(output_corr, "correlation_colm", "spearman").head()
    # print(" spearman correlation...: \n" + str(r1s[0]))

    result_pearson = {'pearson_value': pearson_value,
                      'matrix': pearsonMatrix}
    # print(json_response)

    return result_pearson
