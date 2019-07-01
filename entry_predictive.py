from PredictionAlgorithms.LinearRegressionTest import LinearRegressionModel
from PredictionAlgorithms.LinearPersistModelTest import LinearRegressionPersistModel

from PredictionAlgorithms.GradientBoostingRegressionTest import GradientBoostRegression
from PredictionAlgorithms.GradientBoostingClassificationTest import GradientBoostClassification
from PredictionAlgorithms import random_forest_classifier_test
from PredictionAlgorithms import random_forest_regression_test

from PredictionAlgorithms.lasso_regression_test import Lasso_reg
from PredictionAlgorithms.ridge_regression_test import Ridge_reg
from PredictionAlgorithms.logistic_rf import  logistic_reg
from PredictionAlgorithms.LoadModel import  loadModel

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
# algorithm = "Gradient_Boosting_classification"
# relation = 'linear'
# relation_list = {}
# trainData = [0.80]
# modelUUID = '6786103f-b49b-42f2-ba40-aa8168b65e67'

# Loading the model for the test data
# dataset_add= '/home/fidel/mltest/auto-miles-per-gallon.csv'
# feature_colm= ["CYLINDERS", "WEIGHT" , "HORSEPOWER","ACCELERATION", "DISPLACEMENT", "MODELYEAR"]
# label_colm=["MPG"]
# modelUUID = '6786103f-b49b-42f2-ba40-aa8168b65e67'
# relation = 'linear'
# relation_list = {}
# algorithm = 'modelPersist'


# dataset_add = "/home/fidel/mltest/bank.csv"
# feature_colm = ['balance', 'day', 'duration', 'campaign','age','job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']
# # features = ['age', 'balance', 'day', 'duration', 'campaign']
# label_colm = ["y"]
# algorithm = "linear_reg"
# relation = 'linear'
# relation_list = {}
# trainData = [0.80]
# modelUUID = '6786103f-b49b-42f2-ba40-aa8168b65e67'


dataset_add = '/home/fidel/mltest/BI.csv'
feature_colm= ['Sales Reason','Online Order Flag',	'Customer Name','Territory','Ship Method','Currency Code','Card Type','City','OrderDate','DueDate','ShipDate','Sub Total','Freight','Total Due','Sales','Category']
label_colm= ['Profit']
algorithm = "Gradient_Boosting_regression"
relation = 'linear'
relation_list = {}
trainData = [0.80]
modelUUID = '6786103f-b49b-42f2-ba40-aa8168b65e67'

# large dataset
# dataset_add= '/home/fidel/mltest/auto-miles-per-gallon.csv'
# feature_colm= ["CYLINDERS", "WEIGHT" , "HORSEPOWER","ACCELERATION", "DISPLACEMENT", "MODELYEAR"]
# label_colm=["MPG"]
# relation_list={}
# relation= 'linear'
# algorithm='linear_reg'
# xt=[0.5]
# trainData = [0.80]
# modelUUID = '6786103f-b49b-42f2-ba40-aa8168b65e67'


#
# dataset_add = "/home/fidel/mltest/auto-miles-per-gallon.csv"
# feature_colm = ["CYLINDERS", "WEIGHT" , "HORSEPOWER","ACCELERATION", "DISPLACEMENT", "MODELYEAR"]
# label_colm = ["MPG"]
# relation_list = {}
# relation = 'linear_reg'
# algorithm = "lasso_reg"


def application(data_add, feat_col, label_col, algo,relation_list,relation, modelUUID):
    response_data=''
    print("received request = ")

    print (algo)

    if algo == "linear_reg":
        response_data = LinearRegressionModel().linearReg(dataset_add= data_add, feature_colm=feat_col, label_colm= label_col, relation_list=relation_list, relation=relation, userId=modelUUID)
    if algo == 'modelPersist':
        response_data = LinearRegressionPersistModel().linearRegPersist(dataset_add= data_add, feature_colm=feat_col, label_colm= label_col, relation_list=relation_list, relation=relation, userId=modelUUID)

    if algo == "Gradient_Boosting_regression":
        response_data = GradientBoostRegression().GradientBoostingRegression(dataset_add= data_add, feature_colm=feat_col, label_colm= label_col, relation_list=relation_list, relation=relation)

    if algo == "Gradient_Boosting_classification":
        response_data = GradientBoostClassification().GradientBoostingClassification(dataset_add= data_add, feature_colm=feat_col, label_colm= label_col, relation_list=relation_list, relation=relation)

    if algo == "loadModel":
            response_data = loadModel(dataset_add= data_add, feature_colm=feat_col, label_colm= label_col, relation_list=relation_list, relation=relation)

    elif algo == 'random_classifier':
        response_data = random_forest_classifier_test.randomClassifier(dataset_add=data_add, feature_colm=feat_col, label_colm=label_col)
    elif algo == 'random_regressor':
        response_data = random_forest_regression_test.randomClassifier(dataset_add=data_add, feature_colm=feat_col, label_colm=label_col, relation_list=relation_list, relation=relation)
    elif algo == 'lasso_reg':
        response_data = Lasso_reg(xt=[0.5]).lasso(dataset_add= data_add, feature_colm=feat_col, label_colm= label_col, relation_list=relation_list, relation=relation)
    elif algo == 'ridge_reg':
        response_data = Ridge_reg().ridge(dataset_add= data_add, feature_colm=feat_col, label_colm= label_col, relation_list=relation_list, relation=relation)
    elif algo == 'logistic_reg':
        response_data = logistic_reg.Logistic_regression(dataset_add= data_add, feature_colm= feat_col, label_colm= label_col)

    else:
        print ("sorry unable to process.....")


    print ("done")
    # return iter([response_data])

    print(response_data)


application(dataset_add, feature_colm, label_colm, algorithm,relation_list,relation, modelUUID)