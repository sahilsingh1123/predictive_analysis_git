import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import *

spark = SparkSession.builder.appName("predictive_Analysis").master("local[*]").getOrCreate()
# pyspark --py-files /home/fidel/Downloads/xgboost4j-0.72.jar
spark.sparkContext.setLogLevel("ERROR")
spark.sparkContext.addPyFile('/home/fidel/Downloads/sparkxgb.zip')
# spark.sparkContext.addPyFile('/home/fidel/Downloads/xgboost4j-0.72.jar')
# spark.sparkContext.addPyFile('/home/fidel/Downloads/xgboost4j-spark-0.72.jar')




class Lasso_reg():
    def __init__(self, xt=[0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 0.005, 0.8, 0.3]):
        self.xt = xt

    def lasso(self, data_add):

        Rsqr_list = []
        Rsqr_regPara = {}
        print(self.xt)
        print(data_add)



if __name__=="__main__":
    Lasso_reg().lasso('l')
