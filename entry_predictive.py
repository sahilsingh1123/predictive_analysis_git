import linear_reg_original_test
import random_forest_classifier_test
import random_forest_regression_test

from PredictionAlgorithms import lasso_regression_test

# if __name__== "__main__":

#
# dataset_add= "/home/fidel/mltest/bank.csv"
# feature_colm = ["default","housing","loan"]
# label_colm = ["marital"]

#
# dataset_add = "/home/fidel/mltest/bank.csv"
# feature_colm = ['balance', 'day', 'duration', 'campaign','age','job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']
# # features = ['age', 'balance', 'day', 'duration', 'campaign']
# label_colm = ["y"]
# algorithm = "random_classifier"

#
# feature_colm = ['balance', 'day', 'duration', 'campaign','job','y', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']
# label_colm = ["age"]0
# algorithm = "random_regressor"
# dataset_add = "/home/fidel/mltest/bank.csv"
#

# large dataset
dataset_add= '/home/fidel/mltest/BI.csv'
feature_colm=['Sub Total', 'Tax Amount', 'Freight']
label_colm=['Profit']
relation_list={}
relation= 'linear_reg'
algorithm='lasso_reg'



# dataset_add = "/home/fidel/mltest/auto-miles-per-gallon.csv"
# feature_colm = ["CYLINDERS", "WEIGHT" , "HORSEPOWER","ACCELERATION", "DISPLACEMENT", "MODELYEAR"]
# label_colm = ["MPG"]
# relation_list = {}
# relation = 'linear_reg'
# algorithm = "lasso_reg"


def application(data_add, feat_col, label_col, algo):
    response_data=''
    print("received request = ")

    print (algo)

    if algo == "linear_reg":
        response_data = linear_reg_original_test.Linear_reg(dataset_add= data_add, feature_colm=feat_col, label_colm= label_col)

    elif algo == 'random_classifier':
        response_data = random_forest_classifier_test.randomClassifier(dataset_add=data_add, feature_colm=feat_col, label_colm=label_col)
    elif algo == 'random_regressor':
        response_data = random_forest_regression_test.randomClassifier(dataset_add=data_add, feature_colm=feat_col, label_colm=label_col)
    elif algo == 'lasso_reg':
        response_data = lasso_regression_test.Lasso_reg(dataset_add= data_add, feature_colm=feat_col, label_colm= label_col, relation_list=relation_list, relation=relation)


    else:
        print ("sorry unable to process.....")


    print ("done")
    # return iter([response_data])

    print(response_data)


application(dataset_add, feature_colm, label_colm, algorithm)