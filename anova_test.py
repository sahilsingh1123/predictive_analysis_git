import pandas as pd
from pyspark.sql import SparkSession
import statsmodels.api as sm
from statsmodels.formula.api import ols

import matplotlib.pyplot

# import  statsmodels
from statsmodels.stats.anova import AnovaRM



if __name__ == '__main__':


    spark = SparkSession.builder.appName("predictive_analysis").master("local[*]").getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    print"\nspark session created sucessfully:: \n"

    dataset = spark.read.csv("/home/fidel/mltest/bank.csv", header=True, inferSchema=True)

    dataset.printSchema()
    df = pd.read_csv("/home/fidel/mltest/bank.csv", delimiter=";")
    print df.describe()

    ######creating the box plot
    #

    boxplot = df.boxplot('age', by='marital', figsize=(12, 8))

    df_anova = dataset.toPandas()

    mod = ols("age ~ housing", data=df_anova).fit()
    aov_table = sm.stats.anova_lm(mod , typ=2)
    print aov_table





    # using 1st test


    anovarm = AnovaRM(df_anova, "age", "default" , within=["marital"])
    fit = anovarm.fit()
    fit.summary()



    # 2nd method





