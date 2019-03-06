from pearson_corr_original import pearson_test
# from chi_sqr_original import *
analysis_type = "pearson_test"


# def Predictive_analysis(analysis_type):
if analysis_type == "chi_test=======================":
    print("chi_test")
    # chi()
elif analysis_type == "linear_reg======================":
    print("liner_reg")
    # linear_reg()
elif analysis_type == "pearson_test":
    print("pearson_test  ============================")
    pearson_test.Correlation()
elif analysis_type == "logistic_reg":
    print
    # logistic_reg()



# Predictive_analysis(analysis_type)