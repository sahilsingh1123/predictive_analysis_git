from pyspark.ml.regression import LinearRegressionModel,RandomForestRegressionModel,GBTRegressionModel
from pyspark.sql import SparkSession

from PredictionAlgorithms.PredictiveUtilities import PredictiveUtilities

# spark = \
#     SparkSession.builder.appName('predictive_Analysis').master('local[*]').getOrCreate()
# spark.sparkContext.setLogLevel('ERROR')

class PredictivePrediction():
    def __init__(self,trainDataRatio,dataset_add,
                 feature_colm, label_colm, relation_list,relation,userId,
                 locationAddress,modelStorageLocation,algoName,modelSheetName,datasetName,
                 spark):
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
        self.modelSheetName = "prediction_" + modelSheetName
        self.datasetName = datasetName
        self.spark = spark

        # only for etlpart of the dataset
        # PredictiveUtilities = PredictiveUtilities()

        ETLOnDatasetStats = \
            PredictiveUtilities.ETLOnDataset(datasetAdd=self.datasetAdd,
                                                     featuresColmList=self.featuresColmList,
                                                     labelColmList=self.labelColmList,
                                                     relationshipList=self.relationshipList,
                                                     relation=self.relation,
                                                     trainDataRatio=self.trainDataRatio,
                                                     spark=spark,
                                                     userId=userId)
        self.dataset = ETLOnDatasetStats.get("dataset")
        self.featuresColm = ETLOnDatasetStats.get("featuresColm")
        self.indexedFeatures = ETLOnDatasetStats.get("indexedFeatures")
        self.oneHotEncodedFeaturesList = ETLOnDatasetStats.get("oneHotEncodedFeaturesList")


    def loadModel(self):

        if self.algoName == "linear_reg" or self.algoName == \
                "ridge_reg" or self.algoName == "lasso_reg" :
            regressionPrediction = LinearRegressionModel.load(self.modelStorageLocation)
        if self.algoName == "RandomForestAlgo" :
            regressionPrediction = RandomForestRegressionModel.load(self.modelStorageLocation)
        if self.algoName == "GradientBoostAlgo":
            regressionPrediction = GBTRegressionModel.load(self.modelStorageLocation)

        #dropping the already existed column of prediction on same model
        self.dataset = self.dataset.drop(self.modelSheetName)

        predictionData = regressionPrediction.transform(self.dataset)
        predictionData = predictionData.drop(self.featuresColm)

        #dropping extra added column
        if self.indexedFeatures:
            self.indexedFeatures.extend(self.oneHotEncodedFeaturesList)
            predictionData = predictionData.drop(*self.indexedFeatures)
        else:
            predictionData = predictionData

        #overWriting the original dataset

        '''this step is needed to write because of the nature of spark to not read or write whole data at once
        it only takes limited data to memory and another problem was lazy evaluation of spark.
        so overwriting the same dataset which is already in the memory is not possible'''
        emptyUserId = ''
        fileNameWithPathTemp = self.locationAddress + emptyUserId + self.datasetName + "_temp.parquet"
        predictionData.write.parquet(fileNameWithPathTemp, mode="overwrite")
        predictionDataReadAgain = self.spark.read.parquet(fileNameWithPathTemp)

        predictionTableData = \
            PredictiveUtilities.writeToParquet(fileName=self.datasetName,
                                                       locationAddress=self.locationAddress,
                                                       userId=emptyUserId,
                                                       data=predictionDataReadAgain)        
        return predictionTableData

