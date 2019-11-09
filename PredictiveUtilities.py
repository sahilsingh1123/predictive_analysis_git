import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.functions import abs as absSpark, sqrt as sqrtSpark, mean as meanSpark, stddev as stddevSpark
from scipy.stats import norm

from PredictionAlgorithms.PredictiveDataTransformation import PredictiveDataTransformation
from PredictionAlgorithms.PredictiveConstants import PredictiveConstants


#for future
# from PredictionAlgorithms.PredictiveRegressionModel import *


class PredictiveUtilities():
    # def __init__(self):
    #     pass

    def ETLOnDataset(datasetAdd,featuresColmList,labelColmList,
                     relationshipList,relation,trainDataRatio,spark,userId):

        dataset = spark.read.parquet(datasetAdd)
        # changing the relationship of the colm
        dataTransformationObj = PredictiveDataTransformation(dataset=dataset)
        dataset = \
            dataTransformationObj.colmTransformation(
                colmTransformationList=relationshipList) if relation == PredictiveConstants.NON_LINEAR else dataset
        # transformation
        dataTransformationObj = PredictiveDataTransformation(dataset=dataset)
        dataTransformationResult = dataTransformationObj.dataTranform(labelColm=labelColmList,
                                                                      featuresColm=featuresColmList,userId=userId)
        dataset = dataTransformationResult[PredictiveConstants.DATASET]
        categoricalFeatures = dataTransformationResult.get(PredictiveConstants.CATEGORICALFEATURES)
        numericalFeatures = dataTransformationResult.get(PredictiveConstants.NUMERICALFEATURES)
        maxCategories = dataTransformationResult.get(PredictiveConstants.MAXCATEGORIES)
        categoryColmStats = dataTransformationResult.get(PredictiveConstants.CATEGORYCOLMSTATS)
        indexedFeatures = dataTransformationResult.get(PredictiveConstants.INDEXEDFEATURES)
        idNameFeaturesOrdered=dataTransformationResult.get(PredictiveConstants.IDNAMEFEATURESORDERED)
        oneHotEncodedFeaturesList=dataTransformationResult.get(PredictiveConstants.ONEHOTENCODEDFEATURESLIST)
        label = dataTransformationResult.get(PredictiveConstants.LABEL)
        featuresColm = dataTransformationResult.get(PredictiveConstants.VECTORFEATURES)
        # featuresColm = "features"

        if trainDataRatio is not None:
            trainData, testData = dataset.randomSplit([trainDataRatio, (1 - trainDataRatio)],
                                                      seed=40)
            ETLOnDatasetStat = {PredictiveConstants.FEATURESCOLM: featuresColm, PredictiveConstants.LABELCOLM: label,
                                PredictiveConstants.TRAINDATA: trainData, PredictiveConstants.TESTDATA: testData,
                                PredictiveConstants.IDNAMEFEATURESORDERED: idNameFeaturesOrdered,
                                PredictiveConstants.DATASET: dataset,
                                PredictiveConstants.INDEXEDFEATURES:indexedFeatures,
                                PredictiveConstants.ONEHOTENCODEDFEATURESLIST:oneHotEncodedFeaturesList}
        else:
            ETLOnDatasetStat = {PredictiveConstants.FEATURESCOLM: featuresColm, PredictiveConstants.LABELCOLM: label,
                                PredictiveConstants.IDNAMEFEATURESORDERED: idNameFeaturesOrdered,
                                PredictiveConstants.DATASET: dataset,
                                PredictiveConstants.INDEXEDFEATURES:indexedFeatures,
                                PredictiveConstants.ONEHOTENCODEDFEATURESLIST:oneHotEncodedFeaturesList}

        return ETLOnDatasetStat


    def summaryTable(featuresName,featuresStat):
        statDict={}
        for name, stat in zip(featuresName.values(),
                              featuresStat.values()):
            statDict[name]=stat
        return statDict

    def writeToParquet(fileName,locationAddress,userId,data):
        extention=".parquet"
        fileName=fileName.upper()
        userId = userId.upper()
        fileNameWithPath=locationAddress+userId+fileName+extention
        data.write.parquet(fileNameWithPath,mode="overwrite")
        onlyFileName = userId+fileName
        result = {"fileNameWithPath":fileNameWithPath,
                  "onlyFileName":onlyFileName}
        return result

    def scaleLocationGraph(label,predictionTargetData,residualsData,modelSheetName):

        predictionTrainingWithTarget = \
            predictionTargetData.select(label, modelSheetName,
                                        sqrtSpark(absSpark(predictionTargetData[label])).alias("sqrtLabel"))

        predictionTrainingWithTargetIndexing = \
            predictionTrainingWithTarget.withColumn(PredictiveConstants.ROW_INDEX,
                                                    F.monotonically_increasing_id())
        residualsTrainingIndexing = \
            residualsData.withColumn(PredictiveConstants.ROW_INDEX,
                                     F.monotonically_increasing_id())
        residualsPredictiveLabelDataTraining = \
            predictionTrainingWithTargetIndexing.join(residualsTrainingIndexing,
                                                      on=[PredictiveConstants.ROW_INDEX]).sort(PredictiveConstants.ROW_INDEX).drop(PredictiveConstants.ROW_INDEX)
        residualsPredictiveLabelDataTraining.show()
        stdResiduals = \
            residualsPredictiveLabelDataTraining.select("sqrtLabel", modelSheetName,
                                                                   (residualsPredictiveLabelDataTraining["residuals"] /
                                                                    residualsPredictiveLabelDataTraining[
                                                                        "sqrtLabel"]).alias("stdResiduals"))
        sqrtStdResiduals = \
            stdResiduals.select("stdResiduals", modelSheetName,
                                sqrtSpark(absSpark(stdResiduals["stdResiduals"])).alias(
                                                  "sqrtStdResiduals"))
        sqrtStdResiduals=sqrtStdResiduals.select("stdResiduals", modelSheetName)
        sqrtStdResiduals.show()
        sqrtStdResiduals.na.drop()
        return sqrtStdResiduals

    def residualsFittedGraph(residualsData,predictionData,modelSheetName):
        predictionData=predictionData.select(modelSheetName)
        residualsTrainingIndexing = residualsData.withColumn(PredictiveConstants.ROW_INDEX,
                                                             F.monotonically_increasing_id())
        predictionTrainingIndexing = predictionData.withColumn(PredictiveConstants.ROW_INDEX,
                                                               F.monotonically_increasing_id())
        residualsPredictiveDataTraining = \
            predictionTrainingIndexing.join(residualsTrainingIndexing,
                                            on=[PredictiveConstants.ROW_INDEX]).sort(PredictiveConstants.ROW_INDEX).drop(PredictiveConstants.ROW_INDEX)
        residualsPredictiveDataTraining.na.drop()
        residualsPredictiveDataTraining.show()
        return residualsPredictiveDataTraining

    def quantileQuantileGraph(residualsData,spark):
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

