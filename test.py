from pyspark.sql import SparkSession
from pyspark.context import SparkContext

from pyspark.ml.feature import VectorAssembler

from pyspark.ml.regression import LinearRegression
from pyspark.sql.types import FloatType
import pyspark.sql.functions as f
from pyspark.sql import window as w
# import pyspark.sql.window.Window as w

import pyspark.sql.types as t


if __name__ == '__main__':
    # spark = SparkSession.builder.appName("predictive_analysis").master("local[*]").getOrCreate()
    try:
        # spark = SparkContext.getOrCreate()
        spark = SparkSession.builder.appName("predictive_analysis").master("local[*]").getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")


        sc = spark.sparkContext


        # sqlContext = SQLContext(spark)


        dataset = spark.read.csv("/home/fidel/mltest/auto-miles-per-gallon.csv", header=True, inferSchema=True)

        # dataset = sqlContext.read.parquet("/home/fidel/mltest/sci_kit_test.parquet")


        dataset.show()

        # sdf = dataset


        featureassembler = VectorAssembler(inputCols=["CYLINDERS", "WEIGHT", "ACCELERATION","DISPLACEMENT", "MODELYEAR"],
                                           outputCol="feature_list")

        output = featureassembler.transform(dataset)

        # output.show()
        output.select("feature_list").show()

        output_features = output.select("feature_list")

        feature_uniqueid = output_features.withColumn('unique_id', f.monotonically_increasing_id())
        output=output.withColumn('unique_id', f.monotonically_increasing_id())
        # feature_uniqueid.write.parquet('/home/fidel/mltest/sci_kit_test.parquet', mode = 'overwrite')
        sdf = feature_uniqueid

        ####################################################################################
        # loading the sci-kit learn algo

        from sklearn.linear_model import LogisticRegression

        lg = LogisticRegression()

        # broadcasting the model


        model_broadcast = sc.broadcast(lg)

        def predict_new(feature_map):

            ids, features = zip(*[(k,v) for d in feature_map for k,v in d.items()])

            ind = model_broadcast.value.classes_.tolist().index(1.0)

            probs = [float(v) for v in model_broadcast.value.predict_proba(features)[:, ind]]

            return dict(zip(ids, probs))

        predict_new_udf = f.udf(predict_new, t.MapType(t.LongType(), t.FloatType()))

        nparts = 10

        # putting every thing together

        outcome_udf = ( sdf.select(f.create_map(f.col('unique_id'), f.col('feature_list')).alias('feature_map'),
                                   (f.row_number().over(w.Window.partitionBy(f.lit(1).orderBy(f.lit(1))) % nparts).alias('grouper'))
                                   .groupby(f.col('grouper'))
                                   .agg(f.collect_list(f.col('feature_map')).alias('feature_map'))
                                   .select(predict_new_udf(f.col('feature_map')).alias('results'))
                                   .select(f.explode(f.col('results')).alias('unique_id', 'probability_estimate'))
                                   ))






























        # feature_uniqueid_feature = feature_uniqueid.select("features")
        # feature_uniqueid_index = feature_uniqueid.select('row_index')

        # (feature_uniqueid.select('*', (f.row_number().over(w.Window.partitionBy('features')).orderBy('features')))  % 10).alias('group').show(500)

        # outcome_sdf = (
        #     feature_uniqueid.select(f.create_map(
        #             f.col('unique_id'),
        #             f.col('feature_list')
        #         ).alias('feature_map'),
        #         (
        #                 f.row_number().over(
        #                     w.partitionBy(f.lit(1)).orderBy(f.lit(1))
        #                 ) % nparts
        #         ).alias('grouper')
        #     )
        #         .groupby(f.col('grouper'))
        #         .agg(
        #         f.collect_list(f.col('feature_map')).alias('feature_map')
        #     )
        #         .select(
        #         predict_new_udf(f.col('feature_map')).alias('results')
        #     )
        #         .select(
        #         f.explode(f.col('results'))
        #             .alias('unique_id', 'probability_estimate')
        #     )





    except Exception as e:
        print('exception is =' + str(e))
