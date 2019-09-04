from pyspark.ml.regression import  LinearRegressionModel
from pyspark.sql import SparkSession
from PredictionAlgorithms.PredictiveUtilities import PredictiveUtilities
spark = \
    SparkSession.builder.appName('predictive_Analysis').master('local[*]').getOrCreate()
spark.sparkContext.setLogLevel('ERROR')

class PredictivePrediction():
    def __init__(self,trainDataRatio,dataset_add,
                 feature_colm, label_colm, relation_list,relation):
        self.trainDataRatio = trainDataRatio
        self.datasetAdd = dataset_add
        self.featuresColmList = feature_colm
        self.labelColmList = label_colm
        self.relationshipList = relation_list
        self.relation = relation

        # only for etlpart of the dataset
        self.predictiveUtilitiesObj = PredictiveUtilities()

        ETLOnDatasetStats = \
            self.predictiveUtilitiesObj.ETLOnDataset(datasetAdd=self.datasetAdd,
                                                     featuresColmList=self.featuresColmList,
                                                     labelColmList=self.labelColmList,
                                                     relationshipList=self.relationshipList,
                                                     relation=self.relation,
                                                     trainDataRatio=self.trainDataRatio,
                                                     spark=spark)
        self.dataset = ETLOnDatasetStats.get("dataset")


    def loadModel(self,modelStoringLocation,algoName):

        if algoName == "linear_reg" or algoName == \
                "ridge_reg" or algoName == "lasso_reg" :
            regressionPrediction = LinearRegressionModel.load(modelStoringLocation)

        predictionData = regressionPrediction.transform(self.dataset)

