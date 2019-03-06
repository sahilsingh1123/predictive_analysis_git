from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler

if __name__=="__main__":

    #
    #
    # dataset_address = "/home/fidel/mltest/auto-miles-per-gallon.csv"
    # All_colms = ["MPG", "CYLINDERS", "DISPLACEMENT", "WEIGHT", "ACCELERATION"]

    def Correlation(dataset_address, All_colms):
        spark = SparkSession.builder.appName("predictive_analysis").master("local[*]").getOrCreate()

        spark.sparkContext.setLogLevel("ERROR")

        dataset = spark.read.csv(dataset_address, header=True, inferSchema=True)

        dataset.show()

        # correlation

        featureassembler_correlation = VectorAssembler(
            inputCols=All_colms, outputCol="correlation_colm")
        output_corr = featureassembler_correlation.transform(dataset)
        output_corr.show()

        finalized_corr = output_corr.select("correlation_colm")
        finalized_corr.show()
        from pyspark.ml.stat import Correlation

        r1p = Correlation.corr(output_corr, "correlation_colm").head()
        print("pearson correlation matrix : \n : " + str(r1p[0]))

        r1s = Correlation.corr(output_corr, "correlation_colm", "spearman").head()
        print(" spearman correlation...: \n" + str(r1s[0]))

# Correlation(dataset_address, All_colms)