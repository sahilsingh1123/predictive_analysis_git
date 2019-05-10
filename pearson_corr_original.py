from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession

# if __name__=="__main__":
spark = SparkSession.builder.appName("predictive_analysis").master("spark://fidel-Latitude-E5570:7077").getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
#
#
# feature_colm = ["CYLINDERS", "WEIGHT" , "HORSEPOWER","ACCELERATION", "DISPLACEMENT", "MODELYEAR"]
# label_colm = ["MPG"]
# dataset_address = "/home/fidel/mltest/auto-miles-per-gallon.csv"
# All_colms = ["MPG", "CYLINDERS", "DISPLACEMENT", "WEIGHT", "ACCELERATION"]

def Correlation(dataset_add, feature_colm, label_colm):

    # dataset_add = str(dataset_add).replace("10.171.0.181", "dhiraj")
    print("Dataset Name  ", dataset_add)
    dataset = spark.read.parquet(dataset_add)
    #dataset = spark.read.parquet("hdfs://dhiraj:9000/dev/dmxdeepinsight/datasets/123_AUTOMILES.parquet")

    dataset.show()

    All_colms =  label_colm + feature_colm

    # correlation

    featureassembler_correlation = VectorAssembler(
        inputCols=All_colms, outputCol="correlation_colm")
    output_corr = featureassembler_correlation.transform(dataset)
    output_corr.show()

    finalized_corr = output_corr.select("correlation_colm")
    finalized_corr.show()
    from pyspark.ml.stat import Correlation

    r1p = Correlation.corr(output_corr, "correlation_colm").head()
    print("pearson correlation matrix \n : " + str(r1p[0]))
    print("pearson correlation matrix \n : " + str(r1p[0].toArray()))
    pearson_matrix = r1p[0].toArray().tolist()
    pearson_value = []

    for x in r1p[0].toArray():
        pearson_value.append(x[0])

    print(pearson_value)

    #
    # r1s = Correlation.corr(output_corr, "correlation_colm", "spearman").head()
    # print(" spearman correlation...: \n" + str(r1s[0]))

    json_response = {'pearson_value' : pearson_value,
                     'matrix': pearson_matrix}
    print(json_response)


    return json_response

# Correlation(dataset_address, feature_colm, label_colm)