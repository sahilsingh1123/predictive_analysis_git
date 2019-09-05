import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.functions import abs as absSpark, sqrt as sqrtSpark, mean as meanSpark, stddev as stddevSpark
from scipy.stats import norm

from PredictionAlgorithms.PredictiveDataTransformation import PredictiveDataTransformation


#for future
# from PredictionAlgorithms.PredictiveRegressionModel import *



class PredictiveUtilities():
    def __init__(self):
        pass

    def ETLOnDataset(self,datasetAdd,featuresColmList,labelColmList,
                     relationshipList,relation,trainDataRatio,spark):

        dataset = spark.read.parquet(datasetAdd)
        # changing the relationship of the colm
        dataTransformationObj = PredictiveDataTransformation(dataset=dataset)
        dataset = \
            dataTransformationObj.colmTransformation(
                colmTransformationList=relationshipList) if relation == "non_linear" else dataset
        # transformation
        dataTransformationObj = PredictiveDataTransformation(dataset=dataset)
        dataTransformationResult = dataTransformationObj.dataTranform(labelColm=labelColmList,
                                                                      featuresColm=featuresColmList)
        dataset = dataTransformationResult["dataset"]
        categoricalFeatures = dataTransformationResult["categoricalFeatures"]
        numericalFeatures = dataTransformationResult["numericalFeatures"]
        maxCategories = dataTransformationResult["maxCategories"]
        categoryColmStats = dataTransformationResult["categoryColmStats"]
        indexedFeatures = dataTransformationResult["indexedFeatures"]
        idNameFeaturesOrdered=dataTransformationResult["idNameFeaturesOrdered"]
        oneHotEncodedFeaturesList=dataTransformationResult.get("oneHotEncodedFeaturesList")
        label = dataTransformationResult.get("label")
        featuresColm = "features"

        if trainDataRatio is not None:
            trainData, testData = dataset.randomSplit([trainDataRatio, (1 - trainDataRatio)],
                                                      seed=40)
            ETLOnDatasetStat = {"featuresColm": featuresColm, "labelColm": label,
                                "trainData": trainData, "testData": testData,
                                "idNameFeaturesOrdered": idNameFeaturesOrdered,
                                "dataset": dataset}
        else:
            ETLOnDatasetStat = {"featuresColm": featuresColm, "labelColm": label,
                                "idNameFeaturesOrdered": idNameFeaturesOrdered,
                                "dataset": dataset}

        return ETLOnDatasetStat


    def summaryTable(self,featuresName,featuresStat):
        statDict={}
        for name, stat in zip(featuresName.values(),
                              featuresStat.values()):
            statDict[name]=stat
        return statDict

    def writeToParquet(self,fileName,locationAddress,userId,data):
        extention=".parquet"
        fileName=fileName.upper()
        userId = userId.upper()
        fileNameWithPath=locationAddress+userId+fileName+extention
        data.write.parquet(fileNameWithPath,mode="overwrite")
        onlyFileName = userId+fileName
        result = {"fileNameWithPath":fileNameWithPath,
                  "onlyFileName":onlyFileName}
        return result

    def scaleLocationGraph(self,label,predictionTargetData,residualsData):

        predictionTrainingWithTarget = \
            predictionTargetData.select(label, "prediction",
                                        sqrtSpark(absSpark(predictionTargetData[label])).alias("sqrtLabel"))

        predictionTrainingWithTargetIndexing = \
            predictionTrainingWithTarget.withColumn("row_index",
                                                    F.monotonically_increasing_id())
        residualsTrainingIndexing = \
            residualsData.withColumn("row_index",
                                     F.monotonically_increasing_id())
        residualsPredictiveLabelDataTraining = \
            predictionTrainingWithTargetIndexing.join(residualsTrainingIndexing,
                                                      on=["row_index"]).sort("row_index").drop("row_index")
        residualsPredictiveLabelDataTraining.show()
        stdResiduals = \
            residualsPredictiveLabelDataTraining.select("sqrtLabel", "prediction",
                                                                   (residualsPredictiveLabelDataTraining["residuals"] /
                                                                    residualsPredictiveLabelDataTraining[
                                                                        "sqrtLabel"]).alias("stdResiduals"))
        sqrtStdResiduals = \
            stdResiduals.select("stdResiduals", "prediction",
                                sqrtSpark(absSpark(stdResiduals["stdResiduals"])).alias(
                                                  "sqrtStdResiduals"))
        sqrtStdResiduals=sqrtStdResiduals.select("stdResiduals", "prediction")
        sqrtStdResiduals.show()
        sqrtStdResiduals.na.drop()
        return sqrtStdResiduals

    def residualsFittedGraph(self,residualsData,predictionData):
        predictionData=predictionData.select("prediction")
        residualsTrainingIndexing = residualsData.withColumn("row_index",
                                                             F.monotonically_increasing_id())
        predictionTrainingIndexing = predictionData.withColumn("row_index",
                                                               F.monotonically_increasing_id())
        residualsPredictiveDataTraining = \
            predictionTrainingIndexing.join(residualsTrainingIndexing,
                                            on=["row_index"]).sort("row_index").drop("row_index")
        residualsPredictiveDataTraining.na.drop()
        residualsPredictiveDataTraining.show()
        return residualsPredictiveDataTraining

    def quantileQuantileGraph(self,residualsData,spark):
        sortedResiduals = residualsData.sort("residuals")
        residualsCount = sortedResiduals.count()
        quantile = []
        for value in range(0, residualsCount):
            quantile.append((value - 0.5) / residualsCount)
        zTheory = []
        for value in quantile:
            zTheory.append(norm.ppf(abs(value)))

        meanStdDev = []
        stat = \
            sortedResiduals.select(meanSpark("residuals"), stddevSpark("residuals"))
        for rows in stat.rdd.toLocalIterator():
            for row in rows:
                meanStdDev.append(row)
        meanResiduals = meanStdDev[0]
        stdDevResiduals = meanStdDev[1]
        zPractical = []
        for rows in sortedResiduals.rdd.toLocalIterator():
            for row in rows:
                zPractical.append((row - meanResiduals) / stdDevResiduals)
        print(zTheory, zPractical)
        quantileTheoryPractical = []
        for theory, practical in zip(zTheory, zPractical):
            quantileTheoryPractical.append([round(theory, 5),
                                            round(practical, 5)])
        '''
        #for future
        schemaQuantile=StructType([StructField("theoryQuantile",DoubleType(),True),
                                   StructField("practicalQuantile",DoubleType(),True)])
        quantileDataframe=spark.createDataFrame(quantileTheoryPractical,schema=schemaQuantile)
        '''
        quantileQuantileData = \
            pd.DataFrame(quantileTheoryPractical, columns=["theoryQuantile",
                                                           "practicalQuantile"])
        quantileQuantileData = spark.createDataFrame(quantileQuantileData)
        quantileQuantileData.na.drop()

        return quantileQuantileData

