from pyspark.sql import SparkSession
from pyspark.ml.feature import PCA
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler


if __name__ == '__main__':


    spark = SparkSession.builder.appName("predictive_analysis").master("local[*]").getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    print"\nspark session created sucessfully:: \n"

    dataset = spark.read.csv("/home/fidel/mltest/auto-miles-per-gallon.csv", header=True, inferSchema=True)

    dataset.show()

    # PCA test (priciple component analysis)

        # creating vector assembler

    featureassembler = VectorAssembler(inputCols=["CYLINDERS", "DISPLACEMENT", "WEIGHT","ACCELERATION"],
                                       outputCol="Independent features")

    output = featureassembler.transform(dataset)

    #output.show()
    output.select("Independent features").show()

    finalized_data = output.select("Independent features")

    finalized_data.show()

    # standard scaler implementing on the dataset


    scaler = StandardScaler(inputCol = "Independent features" , outputCol= "scaled_features", withStd=True, withMean=False)

    scalerModel = scaler.fit(finalized_data)

    scaledData = scalerModel.transform(finalized_data)
    scaledData.show()


    pca = PCA(k=4,inputCol="Independent features", outputCol= "pca_features")
    model = pca.fit(finalized_data)

    result = model.transform(finalized_data).select("pca_features")
    result.show(truncate = False)

