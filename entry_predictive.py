import linear_reg_original_test
import pearson_corr_original
import chi_sqr_original
import json

# if __name__== "__main__":

#
# dataset_add= "/home/fidel/mltest/bank.csv"
# feature_colm = ["default","housing","loan"]
# label_colm = ["marital"]


dataset_add = "/home/fidel/mltest/auto-miles-per-gallon.csv"
feature_colm = ["CYLINDERS", "WEIGHT" , "HORSEPOWER","ACCELERATION", "DISPLACEMENT", "MODELYEAR"]
label_colm = ["MPG"]
algorithm = "linear_reg"


def application(data_add, feat_col, label_col, algo):
    response_data=''
    print("received request = ")

    print (algo)

    if algo == "linear_reg":
        response_data = linear_reg_original_test.Linear_reg(dataset_add= data_add, feature_colm=feat_col, label_colm= label_col)

    elif algo == 'pearson_test':
        response_data = pearson_corr_original.Correlation(dataset_add=data_add, feature_colm=feat_col, label_colm=label_col)
    elif algo == 'chi_square_test':
        response_data = chi_sqr_original.Chi_sqr(dataset_add=data_add, feature_colm=feat_col, label_colm=label_col)


    else:
        print ("sorry unable to process.....")


    print ("done")
    # return iter([response_data])

    print(response_data)


application(dataset_add, feature_colm, label_colm, algorithm)