from pyspark.sql import SparkSession
from pyspark.ml.feature import RFormula
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

if __name__ == '__main__':
    spark = SparkSession.builder.appName("predictive_analysis").master("local[*]").getOrCreate()

    spark.sparkContext.setLogLevel("INFO")

    dataset = spark.read.csv("/home/fidel/mltest/auto-miles-per-gallon.csv", header=True, inferSchema=True)

    dataset.show(100)
    dataset=dataset.persist()

    formula = RFormula(formula="MPG ~ CYLINDERS + WEIGHT + ACCELERATION",
                       featuresCol="features",
                       labelCol="label")

    output = formula.fit(dataset).transform(dataset)