
from pyspark.sql import SparkSession

if __name__ == '__main__':


    spark = SparkSession.builder.appName("predictive_analysis").master("local[*]").getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    print"\nspark session created sucessfully:: \n"