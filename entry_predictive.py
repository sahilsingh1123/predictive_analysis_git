import linear_reg_original
import json

dataset_add = "/home/fidel/mltest/auto-miles-per-gallon.csv"
feature_colm = ["CYLINDERS", "WEIGHT" , "HORSEPOWER","ACCELERATION", "DISPLACEMENT", "MODELYEAR"]
label_colm = "MPG"
algorithm = "linear_reg"


def application(data_add, feat_col, label_col, algo):
    response_data=''
    print("received request = ")

    print algo

    if algo == "linear_reg":
        response_data = linear_reg_original.Linear_reg(dataset_add= data_add, feature_colm=feat_col, label_colm= label_col)

    else:
        print "sorry unable to process....."


    print "done"
    return iter([response_data])




application(dataset_add, feature_colm, label_colm, algorithm)