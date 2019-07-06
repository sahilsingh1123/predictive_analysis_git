import json

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorIndexer, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.types import *

from PredictionAlgorithms.chi_sqr_importance import chi_square_test
from PredictionAlgorithms.relationship import Relationship

spark = SparkSession.builder.appName("predictive analysis").master("local[*]").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

class RandomClassifierModel():
    def randomClassifier(dataset_add, feature_colm, label_colm, relation_list, relation,userId):
        try:
            dataset = spark.read.parquet(dataset_add)
            label = ''
            for y in label_colm:
                label = y

            Schema = dataset.schema
            stringFeatures = []
            numericalFeatures = []
            for x in Schema:
                if (str(x.dataType) == "StringType" or str(x.dataType) == "TimestampType" or str(
                        x.dataType) == "DateType" or str(x.dataType) == "BooleanType" or str(x.dataType) == "BinaryType"):
                    for y in feature_colm:
                        if x.name == y:
                            dataset = dataset.withColumn(y, dataset[y].cast(StringType()))
                            stringFeatures.append(x.name)
                else:
                    for y in feature_colm:
                        if x.name == y:
                            numericalFeatures.append(x.name)

            if relation == 'linear':
                dataset = dataset
            if relation == 'non_linear':
                dataset = Relationship(dataset, relation_list)
            summaryList = ['mean', 'stddev', 'min', 'max']
            summaryDict = {}

            import pyspark.sql.functions as F
            import builtins
            round = getattr(builtins, 'round')
            for colm in numericalFeatures:
                summaryListTemp = []
                for value in summaryList:
                    summ = list(dataset.select(colm).summary(value).toPandas()[colm])
                    summaryListSubTemp = []
                    for val in summ:
                        summaryListSubTemp.append(round(float(val), 4))
                    summaryListTemp.append(summaryListSubTemp)
                summaryDict[colm] = summaryListTemp
            summaryList.extend(['skewness','kurtosis', 'variance'])
            summaryDict['summaryName'] = summaryList
            summaryDict['categoricalColumn'] = stringFeatures
            skewnessList = []
            kurtosisList = []
            varianceList = []
            skewKurtVarDict = {}
            for colm in numericalFeatures:
                skewness = (dataset.select(F.skewness(dataset[colm])).toPandas())
                for i, row in skewness.iterrows():
                    for j, column in row.iteritems():
                        skewnessList.append(round(column, 4))
                kurtosis = (dataset.select(F.kurtosis(dataset[colm])).toPandas())
                for i, row in kurtosis.iterrows():
                    for j, column in row.iteritems():
                        kurtosisList.append(round(column, 4))
                variance = (dataset.select(F.variance(dataset[colm])).toPandas())
                for i, row in variance.iterrows():
                    for j, column in row.iteritems():
                        varianceList.append(round(column, 4))

            for skew, kurt, var, colm in zip(skewnessList, kurtosisList, varianceList, numericalFeatures):
                print(skew, kurt, var)
                skewKurtVarList = []
                skewKurtVarList.append(skew)
                skewKurtVarList.append(kurt)
                skewKurtVarList.append(var)
                skewKurtVarDict[colm] = skewKurtVarList

            for (keyOne, valueOne), (keyTwo, valueTwo) in zip(summaryDict.items(), skewKurtVarDict.items()):
                print(keyOne, valueOne, keyTwo, valueTwo)
                if keyOne == keyTwo:
                    valueOne.extend(valueTwo)
                    summaryDict[keyOne] = valueOne

            categoryColmList = []
            categoryColmListFinal = []
            categoryColmListDict = {}
            countOfCategoricalColmList = []
            for value in stringFeatures:
                categoryColm = value
                listValue = value
                listValue = []
                categoryColm = dataset.groupby(value).count()
                countOfCategoricalColmList.append(categoryColm.count())
                categoryColmJson = categoryColm.toJSON()
                for row in categoryColmJson.collect():
                    categoryColmSummary = json.loads(row)
                    listValue.append(categoryColmSummary)
                categoryColmListDict[value] = listValue

            if not stringFeatures:
                maxCategories = 5
            else:
                maxCategories = max(countOfCategoricalColmList)
            if relation == 'linear':
                dataset = dataset
            if relation == 'non_linear':
                dataset = Relationship(dataset, relation_list)

            dataset.show()
            for x in Schema:
                if (str(x.dataType) == "StringType" and x.name == label):
                    for labelkey in label_colm:
                        label_indexer = StringIndexer(inputCol=label, outputCol='indexed_' + label, handleInvalid="skip").fit(dataset)
                        dataset = label_indexer.transform(dataset)
                        label = 'indexed_' + label
                else:
                    label = label
            indexed_features = []
            for colm in stringFeatures:
                indexer = StringIndexer(inputCol=colm, outputCol='indexed_' + colm, handleInvalid="skip").fit(dataset)
                indexed_features.append('indexed_' + colm)
                dataset = indexer.transform(dataset)
            final_features = numericalFeatures + indexed_features
            response_chi_test = chi_square_test(dataset=dataset, features=indexed_features, label_col=label,
                                                stringFeatures=stringFeatures)

            featureassembler = VectorAssembler(
                inputCols=final_features,
                outputCol="features", handleInvalid="skip")
            dataset = featureassembler.transform(dataset)
            dataset.show()
            vec_indexer = VectorIndexer(inputCol='features', outputCol='vec_indexed_features', maxCategories=maxCategories, handleInvalid="skip").fit(
                dataset)
            categorical_features = vec_indexer.categoryMaps
            print("Choose %d categorical features: %s" %
                  (len(categorical_features), ", ".join(str(k) for k in categorical_features.keys())))
            vec_indexed = vec_indexer.transform(dataset)
            vec_indexed.show()
            finalized_data = vec_indexed.select(label, 'vec_indexed_features')
            train_data, test_data = finalized_data.randomSplit([0.75, 0.25], seed=40)
            rf = RandomForestClassifier(labelCol=label, featuresCol='vec_indexed_features', numTrees=10, maxBins=3000)
            model = rf.fit(train_data)
            predictions = model.transform(test_data)
            print(model.featureImportances)
            feature_importance = model.featureImportances.toArray().tolist()
            print(feature_importance)
            import pyspark.sql.functions as F
            import builtins
            round = getattr(builtins, 'round')
            feature_importance = model.featureImportances.toArray().tolist()
            print(feature_importance)
            featureImportance = []
            for x in feature_importance:
                featureImportance.append(round(x, 4))
            print(featureImportance)

            features_column_for_user = numericalFeatures + stringFeatures
            feature_imp = {'feature_importance': featureImportance, "feature_column": features_column_for_user}
            response_dict = {
                'feature_importance': feature_imp,
                'ChiSquareTestData': response_chi_test,
                'summaryDict': summaryDict,
                'categoricalSummary': categoryColmListDict

            }
            return response_dict
        except Exception as e:
            print("exception is  = " + str(e))
