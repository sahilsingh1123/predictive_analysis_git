from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql import SparkSession

from PredictionAlgorithms.pearson_test_importance import Correlation_test_imp
from PredictionAlgorithms.relationship import Relationship

spark = SparkSession.builder.appName("predictive analysis").master("local[*]").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

def randomClassifier(dataset_add, feature_colm, label_colm,relation_list, relation):
    try:
        dataset = spark.read.parquet(dataset_add)
        label = ''
        for y in label_colm:
            label = y

        Schema = dataset.schema
        stringFeatures = []
        numericalFeatures = []
        for x in Schema:
            if (str(x.dataType) == "StringType" ):
                for y in feature_colm:
                    if x.name == y:
                        stringFeatures.append(x.name)
            else:
                for y in feature_colm:
                    if x.name == y:
                        numericalFeatures.append(x.name)

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
                    summaryListSubTemp.append(round(float(val), 2))
                # print(summaryListSubTemp)
                summaryListTemp.append(summ)
            # varianceListTemp = list(dataset.select(variance(col(colm)).alias(colm)).toPandas()[colm])
            # summaryListTemp.append(varianceListTemp)
            summaryDict[colm] = summaryListTemp
        # summaryList.append('variance')
        summaryDict['summaryName'] = summaryList
        summaryDict['categoricalColumn'] = stringFeatures
        if relation == 'linear':
            dataset = dataset
        if relation == 'non_linear':
            dataset = Relationship(dataset, relation_list)
        response_pearson_test = Correlation_test_imp(dataset=dataset, features = numericalFeatures, label_col= label)
        indexed_features = []
        for colm in stringFeatures:
            indexer = StringIndexer(inputCol=colm, outputCol='indexed_' + colm).fit(dataset)
            indexed_features.append('indexed_'+colm)
            dataset = indexer.transform(dataset)
        final_features = numericalFeatures + indexed_features
        featureassembler = VectorAssembler(
            inputCols=final_features,
            outputCol="features")
        dataset = featureassembler.transform(dataset)
        dataset.show()
        vec_indexer = VectorIndexer(inputCol='features', outputCol='vec_indexed_features', maxCategories=4).fit(dataset)
        categorical_features = vec_indexer.categoryMaps
        print("Choose %d categorical features: %s" %
                    (len(categorical_features), ", ".join(str(k) for k in categorical_features.keys())))
        vec_indexed = vec_indexer.transform(dataset)
        vec_indexed.show()
        finalized_data = vec_indexed.select(label, 'vec_indexed_features')
        train_data, test_data = finalized_data.randomSplit([0.75, 0.25], seed=40)
        rf=RandomForestRegressor(labelCol=label,featuresCol='vec_indexed_features',numTrees=10)
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
        # feature_importance = [round(x,4) for x in feature_importance]
        featureImportance = []
        for x in feature_importance:
            featureImportance.append(round(x, 4))
        print(featureImportance)

        features_column_for_user = numericalFeatures + stringFeatures
        feature_imp = { 'feature_importance': feature_importance,"feature_column" : features_column_for_user}
        response_dict = {
            'feature_importance': feature_imp,
            'pearson_test_data': response_pearson_test,
            'summaryDict' : summaryDict
        }
        return response_dict
    except Exception as e :
        print("exception is  = " + str(e))
