from pyspark.ml.regression import LinearRegressionModel
from pyspark.sql import SparkSession

from PredictionAlgorithms.PredictiveUtilities import PredictiveUtilities

spark = \
    SparkSession.builder.appName('predictive_Analysis').master('local[*]').getOrCreate()
spark.sparkContext.setLogLevel('ERROR')

class PredictivePrediction():
    def __init__(self,trainDataRatio,dataset_add,
                 feature_colm, label_colm, relation_list,relation,userId,
                 locationAddress,modelStorageLocation,algoName):
        self.trainDataRatio = None if trainDataRatio == None else trainDataRatio
        self.datasetAdd = dataset_add
        self.featuresColmList = feature_colm
        self.labelColmList = None if label_colm == None else label_colm
        self.relationshipList = relation_list
        self.relation = relation
        self.userId = userId
        self.locationAddress = locationAddress
        self.modelStorageLocation = modelStorageLocation
        self.algoName = algoName

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


    def loadModel(self):

        if self.algoName == "linear_reg" or self.algoName == \
                "ridge_reg" or self.algoName == "lasso_reg" :
            regressionPrediction = LinearRegressionModel.load(self.modelStorageLocation)

        predictionData = regressionPrediction.transform(self.dataset)

        self.featuresColmList.append("prediction")
        predictionData = predictionData.select(self.featuresColmList)

        predictionTableData = \
            self.predictiveUtilitiesObj.writeToParquet(fileName="predictionData",
                                                       locationAddress=self.locationAddress,
                                                       userId=self.userId,
                                                       data=predictionData)
        return predictionTableData

