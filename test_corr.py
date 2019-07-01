
feature_colm = ["CYLINDERS", "WEIGHT" , "HORSEPOWER","ACCELERATION", "DISPLACEMENT", "MODELYEAR"]
label_colm = ["MPG"]

label = ''
for y in label_colm:
    label = y

print(label)


import pyspark.sql.functions as f
residual_graph = residual_graph.withColumn('row_index', f.monotonically_increasing_id())
lr_prediction_onlypred = lr_prediction_onlypred.withColumn('row_index', f.monotonically_increasing_id())

finaldataframe = residual_graph.join(lr_prediction_onlypred, on=["row_index"]).sort("row_index").drop("row_index")

finaldataframe.show()