# Databricks notebook source
ACCESS_KEY = "" # your access key
SECRET_KEY = "" # your secret key
ENCODED_SECRET_KEY = SECRET_KEY.replace("/", "%2F")
AWS_BUCKET_NAME = "comp4651-project-data"
MOUNT_NAME = "data"

#dbutils.fs.mount("s3a://%s:%s@%s" % (ACCESS_KEY, ENCODED_SECRET_KEY, AWS_BUCKET_NAME), "/mnt/%s" % MOUNT_NAME)
display(dbutils.fs.ls("/mnt/%s" % MOUNT_NAME))

#val mdf = spark.read.option("multiline", "true").json("train.json")
#mdf.show(false)

#df = spark.read.text("/mnt/%s/test.json" % MOUNT_NAME)

# COMMAND ----------

display(dbutils.fs.ls("/mnt/%s" % MOUNT_NAME))

# COMMAND ----------

df = spark.read.option("multiline", "true").json("dbfs:/mnt/data/train.json")
display(df)

# COMMAND ----------

# occur = rdd.flatMap(lambda (cuisine,_,ingredients): [(cuisine, ing) for ing in ingredients])
# occur.collect()
# freq=occur.map(lambda (cuisine,ingredient): ((cuisine,ingredient),1)).reduce(lambda x,y:x+y)
# freq.collect()

# COMMAND ----------

# get the cuisine dictionary
cuisine_list = df.select('cuisine').distinct().collect() # list'
cuisines = [row['cuisine'] for row in cuisine_list]
cuisine_dict = {}
for i,cuisine in enumerate(cuisines):
  cuisine_dict[cuisine] = i

print(cuisine_dict)

# COMMAND ----------

# get the cuisine dictionary

occur = df.rdd.flatMap(lambda (cuisine,_,ingredients): [(cuisine, ing) for ing in ingredients])
occur = occur.map(lambda (cuisine, ing): (ing, 1)).reduceByKey(lambda a,b: a+b)
ingredient_list = occur.collect()

ingredients = [row[0] for row in ingredient_list]
ingredient_dict = {}
for i,ingredient in enumerate(ingredients):
  ingredient_dict[ingredient] = i

# print(ingredient_dict)

# COMMAND ----------

tuple_length = len(ingredient_dict) + 1 # 1 for the cuisine/label
def parseRow(row):
  cuisine, ingredients = row[0], row[2]
  one_positions = [ingredient_dict[ing] for ing in ingredients]
  row_vals = [0] * tuple_length
  for one in one_positions:
    row_vals[one] = 1
  row_vals[-1] = cuisine_dict[cuisine]
  return tuple(row_vals)

vals = df.rdd.map(parseRow).collect()

# COMMAND ----------

(sample_df, _) = df.randomSplit([0.0001,0.9999],1900009193L) # sample_df.count()=4
sample_vals = sample_df.rdd.map(parseRow).collect()
# display(sample_df)
# for key, value in cuisine_dict.items():
#   if value == sample_vals[0][-1]:
#     print(key)
  
# for ing in ["capers","eggplant","chopped fresh thyme","olive oil","marinara sauce","chopped garlic","vegetable oil cooking spray","zucchini","onions","fresh basil","monkfish fillets","bell pepper"]:
#   print(sample_vals[0][ingredient_dict[ing]])

# COMMAND ----------

from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

columns = map(str, range(0, len(ingredients)+1))
parsed_df = sqlContext.createDataFrame(vals, columns)
#parsed_df = sqlContext.createDataFrame(sample_vals, columns)
display(parsed_df)

# COMMAND ----------

parsed_df.write.save('/mnt/data/parsed_df.csv', format='csv', header=True, mode="overwrite")

# COMMAND ----------

parsed_df=df = spark.read.option("multiline", "true").json("dbfs:/mnt/data/train.json")

# COMMAND ----------

(split15DF, split85DF) = parsed_df.randomSplit([0.15,0.85],1900009193L)
testSetDF = split15DF.cache()
trainingSetDF = split85DF.cache()

# COMMAND ----------

parsed_df.printSchema()

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
vectorizer = VectorAssembler()
vectorizer.setInputCols(columns[:-2])
vectorizer.setOutputCol("features")

# COMMAND ----------

from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml import Pipeline

# Create a DecisionTreeRegressor
dt = DecisionTreeRegressor()

dt.setPredictionCol("Prediction_cuisine")\
  .setLabelCol("6714")\
  .setFeaturesCol("features")\
  .setMaxBins(100)

# Create a Pipeline
dtPipeline = Pipeline()

# Set the stages of the Pipeline
dtPipeline.setStages([vectorizer,dt])

# Let's first train on the entire dataset to see what we get
dtModel = dtPipeline.fit(trainingSetDF)

# COMMAND ----------

resultsDtDf = dtModel.transform(testSetDF)
resultsDtDf.write.save('/mnt/data/resultsDtDf.parquet', format='parquet', header=True, mode="overwrite")

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator

# Create an RMSE evaluator using the label and predicted columns
regEval = RegressionEvaluator(predictionCol="Prediction_cuisine", labelCol="6714", metricName="rmse")

# We can reuse the RegressionEvaluator, regEval, to judge the model based on the best Root Mean Squared Error
# Let's create our CrossValidator with 3 fold cross validation
crossval = CrossValidator(estimator=dtPipeline, evaluator=regEval, numFolds=3)

# Let's tune over our dt.maxDepth parameter on the values 2 and 3, create a paramter grid using the ParamGridBuilder
paramGrid = (ParamGridBuilder()
             .addGrid(dt.maxDepth, [2,3])
             .build())

# Add the grid to the CrossValidator
crossval.setEstimatorParamMaps(paramGrid)

# Now let's find and return the best model
dtModelBest = crossval.fit(trainingSetDF).bestModel

# COMMAND ----------

# Now let's use dtModel to compute an evaluation metric for our test dataset: testSetDF
resultsBestDtDf = dtModelBest.transform(testSetDF)
resultsBestDtDf.write.save('/mnt/data/resultsBestDtDf.parquet', format='parquet', header=True, mode="overwrite")

# COMMAND ----------



# COMMAND ----------

from pyspark.ml.regression import RandomForestRegressor

# Create a RandomForestRegressor
rf = RandomForestRegressor()

rf.setPredictionCol("Prediction_cuisine")\
  .setLabelCol("6714")\
  .setFeaturesCol("features")\
  .setSeed(190088121L)\
  .setMaxDepth(8)\
  .setNumTrees(25)

# Create a Pipeline
rfPipeline = Pipeline()

# Set the stages of the Pipeline
rfPipeline.setStages([vectorizer, rf])

# Let's first train on the entire dataset to see what we get
rfModel = rfPipeline.fit(trainingSetDF)

# COMMAND ----------

resultsRfDf = rfModel.transform(testSetDF)
resultsRfDf.write.save('/mnt/data/resultsRfDf.parquet', format='parquet', header=True, mode="overwrite")

# COMMAND ----------

# Let's just reuse our CrossValidator with the new rfPipeline,  RegressionEvaluator regEval, and 3 fold cross validation
crossval.setEstimator(rfPipeline)

# Let's tune over our rf.maxBins parameter on the values 50 and 100, create a paramter grid using the ParamGridBuilder
paramGrid = (ParamGridBuilder()
             .addGrid(rf.maxBins, [50,100])
             .build())

# Add the grid to the CrossValidator
crossval.setEstimatorParamMaps(paramGrid)

# Now let's find and return the best model
rfModelBest = crossval.fit(trainingSetDF).bestModel

# COMMAND ----------

resultsBestRfDf = rfModelBest.transform(testSetDF)
resultsBestRfDf.write.save('/mnt/data/resultsBestRfDf.parquet', format='parquet', header=True, mode="overwrite")
