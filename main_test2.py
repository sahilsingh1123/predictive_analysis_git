# from chi_sqr_original import *
# from linear_reg_original import *
# from pearson_corr_original import *
import pearson_corr_original
# from logistic_regression_original import *

analysis_type = "pearson_test"

dataset_address = "/home/fidel/mltest/auto-miles-per-gallon.csv"
All_colms = ["MPG", "CYLINDERS", "DISPLACEMENT", "WEIGHT", "ACCELERATION"]

def Predictive_analysis(analysis_type):
    # if analysis_type == "chi_test":
    #     print("chi_test")
    #     chi()
    # elif analysis_type == "linear_reg":
    #     print("liner_reg")
    #     linear_reg()
    if analysis_type == "pearson_test":
        print("pearson_test")
        pearson_corr_original.Correlation(dataset_address=dataset_address, All_colms=All_colms)
    # elif analysis_type == "logistic_reg":
    #     print
    #     logistic_reg()



Predictive_analysis(analysis_type)