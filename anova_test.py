import pandas as pd
from pyspark.sql import SparkSession
import statsmodels.api as sm
from statsmodels.formula.api import ols

import matplotlib.pyplot

# import  statsmodels
# from statsmodels.stats.anova import AnovaRM
# statsmodels.__version__


if __name__ == '__main__':


    spark = SparkSession.builder.appName("predictive_analysis").master("local[*]").getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    print"\nspark session created sucessfully:: \n"

    dataset = spark.read.csv("/home/fidel/mltest/auto-miles-per-gallon.csv", header=True, inferSchema=True)

    dataset.printSchema()
    df = pd.read_csv("/home/fidel/mltest/bank.csv", delimiter=";")
    print df.describe()

    ######creating the box plot
    #

    boxplot = df.boxplot('age', by='marital', figsize=(12, 8))


    mod = ols("age ~ housing", data=df).fit()
    aov_table = sm.stats.anova_lm(mod , typ=2)
    print aov_table





    # using 1st test
    #
    # anovarm = AnovaRM(df, "MPG", "ACCELERATION" , within=["WEIGHT"])
    # fit = anovarm.fit()
    # fit.summary()



    # 2nd method





