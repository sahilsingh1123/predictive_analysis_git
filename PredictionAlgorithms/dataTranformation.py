from pyspark.ml.feature import VectorIndexer, VectorAssembler,StringIndexer
import json
from pyspark.sql.types import *
class DataTransformation():
    def __init__(self,dataset,labelColm,featuresColm,categoricalFeatures,numericalFeatures):
        self.dataset=None if dataset==None else dataset
        self.labelColm=None if labelColm==None else labelColm
        self.featuresColm=None if featuresColm==None else featuresColm
        self.categoricalFeatures=None if categoricalFeatures==None else categoricalFeatures
        self.numericalFeatures=None if numericalFeatures==None else numericalFeatures
    #data transformation
    def dataTranform(self):
        dataset=self.dataset
        schemaData=dataset.schema
        categoricalFeatures=[]
        numericalFeatures=[]
        for schemaVal in schemaData:
            if (str(schemaVal.dataType) == "StringType" or str(schemaVal.dataType) == "TimestampType" or str(
                    schemaVal.dataType) == "DateType" or str(schemaVal.dataType) == "BooleanType" or str(schemaVal.dataType) == "BinaryType"):
                for y in self.featuresColm:
                    if schemaVal.name == y:
                        dataset = dataset.withColumn(y, dataset[y].cast(StringType()))
                        categoricalFeatures.append(schemaVal.name)
            else:
                for y in self.featuresColm:
                    if schemaVal.name == y:
                        numericalFeatures.append(schemaVal.name)

        for schemaVal in schemaData:
            if (str(schemaVal.dataType) == "StringType" and schemaVal.name == label):
                for labelkey in self.labelColm:
                    label_indexer = StringIndexer(inputCol=label, outputCol='indexed_' + label,
                                                  handleInvalid="skip").fit(dataset)
                    dataset = label_indexer.transform(dataset)
                    label = 'indexed_' + label
            else:
                label = label
        indexedFeatures=[]
        for colm in categoricalFeatures:
            indexer = StringIndexer(inputCol=colm, outputCol='indexed_' + colm, handleInvalid="skip").fit(dataset)
            indexedFeatures.append('indexed_' + colm)
            dataset = indexer.transform(dataset)
        combinedFeatures= numericalFeatures + indexedFeatures
        categoryColmListDict = {}
        countOfCategoricalColmList = []
        for value in categoricalFeatures:
            # categoryColm = value
            # listValue = value
            listValue = []
            categoryColm = dataset.groupby(value).count()
            countOfCategoricalColmList.append(categoryColm.count())
            categoryColmJson = categoryColm.toJSON()
            for row in categoryColmJson.collect():
                categoryColmSummary = json.loads(row)
                listValue.append(categoryColmSummary)
            categoryColmListDict[value] = listValue
        self.numericalFeatures=numericalFeatures
        self.categoricalFeatures=categoricalFeatures
        if not categoricalFeatures:
            maxCategories = 5
        else:
            maxCategories = max(countOfCategoricalColmList)
        
        featureassembler = VectorAssembler(
            inputCols=combinedFeatures,
            outputCol="features", handleInvalid="skip")
        dataset = featureassembler.transform(dataset)
        vec_indexer = VectorIndexer(inputCol='features', outputCol='vec_indexed_features', maxCategories=maxCategories,
                                    handleInvalid="skip").fit(dataset)
        categorical_features = vec_indexer.categoryMaps
        print("Choose %d categorical features: %s" %
              (len(categorical_features), ", ".join(str(k) for k in categorical_features.keys())))
        dataset= vec_indexer.transform(dataset)
        return dataset,categoricalFeatures,numericalFeatures


    #stats of the each colm
    def dataStatistics(self):
        self.dataTranform()
        summaryList = ['mean', 'stddev', 'min', 'max']
        summaryDict = {}
        dataset=self.dataset
        import pyspark.sql.functions as F
        import builtins
        round = getattr(builtins, 'round')
        for colm in self.numericalFeatures:
            summaryListTemp = []
            for value in summaryList:
                summ = list(dataset.select(colm).summary(value).toPandas()[colm])
                summaryListSubTemp = []
                for val in summ:
                    summaryListSubTemp.append(round(float(val), 4))
                summaryListTemp.append(summaryListSubTemp)
            summaryDict[colm] = summaryListTemp
        summaryList.extend(['skewness', 'kurtosis', 'variance'])
        summaryDict['summaryName'] = summaryList
        summaryDict['categoricalColumn'] = self.categoricalFeatures
        skewnessList = []
        kurtosisList = []
        varianceList = []
        skewKurtVarDict = {}
        for colm in self.numericalFeatures:
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

        for skew, kurt, var, colm in zip(skewnessList, kurtosisList, varianceList, self.numericalFeatures):
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