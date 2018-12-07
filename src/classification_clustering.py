# Databricks notebook source
parsed_df= spark.read.csv("dbfs:/mnt/data/parsed_df.csv",header = True)

# COMMAND ----------

#display(parsed_df)
parsed_df.count()

# COMMAND ----------

all_columns_count = 6715
columns = map(str, range(0, all_columns_count))

len(columns[:-1])

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
vectorizer = VectorAssembler()
feature_cols = columns[:-1]
vectorizer.setInputCols(feature_cols)
vectorizer.setOutputCol("features")

# COMMAND ----------

from pyspark.sql.functions import *

expr = [col(c).cast('Double').alias(c) for c in columns]
parsed_df = parsed_df.select(*expr)
df2 = vectorizer.transform(parsed_df)

# COMMAND ----------

# df2.show()
df3 = df2.select('features')
display(df3)


# COMMAND ----------

# joining two dataframes
df_1 = parsed_df.withColumnRenamed('6714', 'label').select('label').withColumn('id', monotonically_increasing_id())
df_2 = df3.select('features').withColumn('id', monotonically_increasing_id())

df_ = df_1.join(df_2, 'id', 'outer').drop('id')
display(df_)

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(df_)
# Automatically identify categorical features, and index them.
# We specify maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=20).fit(df_)

(trainingData, testData) = df_.randomSplit([0.85,0.15],1900009193L)

# Train a DecisionTree model.
dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")

# Chain indexers and tree in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("prediction", "indexedLabel", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g " % (1.0 - accuracy))
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="f1")
print("F1 Score = " + str(evaluator.evaluate(predictions)))

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# We can reuse the RegressionEvaluator, regEval, to judge the model based on the best Root Mean Squared Error
# Let's create our CrossValidator with 3 fold cross validation
crossval = CrossValidator(estimator=pipeline, evaluator=evaluator, numFolds=3)

# We'll create a paramter grid using the ParamGridBuilder, and add the grid to the CrossValidator
paramGrid = (ParamGridBuilder()
             .addGrid(dt.maxDepth, [2,6,10])
             .addGrid(dt.maxBins,[20, 32, 40])
             .build())
crossval.setEstimatorParamMaps(paramGrid)
# Now let's find and return the best model
cvModel = crossval.fit(trainingData).bestModel

# Make predictions.
predictions = cvModel.transform(testData)

# Select example rows to display.
predictions.select("prediction", "indexedLabel", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g " % accuracy)
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="f1")
print("F1 Score = " + str(evaluator.evaluate(predictions)))

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Load and parse the data file, converting it to a DataFrame.
# data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(df_)

# Automatically identify categorical features, and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=20).fit(df_)

# Split the data into training and test sets (30% held out for testing)
# (trainingData, testData) = df_.randomSplit([0.7, 0.3])

# Train a RandomForest model.
rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)

# Convert indexed labels back to original labels.
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)

# Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="f1")
score = evaluator.evaluate(predictions)
print("F1 score = %g" % score)

evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g" % accuracy)

# COMMAND ----------

# We can reuse the RegressionEvaluator, regEval, to judge the model based on the best Root Mean Squared Error
# Let's create our CrossValidator with 3 fold cross validation
crossval = CrossValidator(estimator=pipeline, evaluator=evaluator, numFolds=3)

# We'll create a paramter grid using the ParamGridBuilder, and add the grid to the CrossValidator
paramGrid = (ParamGridBuilder()
             .addGrid(rf.maxDepth, [2,6,10])
             .addGrid(rf.numTrees,[10,30])
             .build())
crossval.setEstimatorParamMaps(paramGrid)
# Now let's find and return the best model
cvModel = crossval.fit(trainingData).bestModel

# Make predictions.
predictions = cvModel.transform(testData)

# Select example rows to display.
predictions.select("prediction", "indexedLabel", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g " % accuracy)
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="f1")
print("F1 Score = " + str(evaluator.evaluate(predictions)))

# COMMAND ----------

from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Split the data into train and test
#(trainingData, testData) = df_.randomSplit([0.15,0.85],1900009193L)

# specify layers for the neural network:
# input layer of size 4 (features), two intermediate of size 5 and 4
# and output of size 3 (classes)
layers = [6714, 500, 20]

# create the trainer and set its parameters
trainer = MultilayerPerceptronClassifier(maxIter=1000, layers=layers, blockSize=128, seed=1234)

# train the model
model = trainer.fit(trainingData)

# compute accuracy on the test set
result = model.transform(testData)
predictionAndLabels = result.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))

# COMMAND ----------

evaluator = MulticlassClassificationEvaluator(metricName="f1")
print("F1 Score = " + str(evaluator.evaluate(predictionAndLabels)))

# COMMAND ----------

from pyspark.ml.clustering import KMeans

# Trains a k-means model.
model = KMeans().setK(20).setSeed(1).fit(df_)

# Evaluate clustering by computing Within Set Sum of Squared Errors.
wssse = model.computeCost(df_)
print("Within Set Sum of Squared Errors = " + str(wssse))

# Shows the result.
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)

# COMMAND ----------

from pyspark.ml.feature import PCA as PCAml
from pyspark.ml.linalg import Vectors  # Pre 2.0 pyspark.mllib.linalg

pca = PCAml(k=2, inputCol="features", outputCol="pca")
model = pca.fit(df_)
df_pca = model.transform(df_)


# COMMAND ----------

model.explainedVariance

# COMMAND ----------

display(df_pca)

# COMMAND ----------

from pyspark.ml.clustering import KMeans

# Trains a k-means model.
model = KMeans().setParams(featuresCol="pca", k=20, seed=1).fit(df_pca)

# Evaluate clustering by computing Within Set Sum of Squared Errors.
wssse = model.computeCost(df_pca)
print("Within Set Sum of Squared Errors = " + str(wssse))

# Shows the result.
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)

# COMMAND ----------

df_dis = df_pca.select('pca').withColumnRenamed('pca', 'features')
display(model, df_dis)

# COMMAND ----------

df_dis = df_pca.select('pca', 'label').withColumnRenamed('pca', 'features')
display(df_dis)

# COMMAND ----------

dis = df_dis.rdd.collect()

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
# ax.yaxis.set_ticks_position('left')
# ax.xaxis.set_ticks_position('bottom')

for l in range(20):
  filtered = [rec for rec in dis if rec['label']==l]
  x = [rec['features'][0] for rec in filtered]
  y = [rec['features'][1] for rec in filtered]
  ax.plot(x, y, 'o',markersize=3)

display(fig)

# COMMAND ----------

fig, ax = plt.subplots()

x = [c[0] for c in centers[0:1]]
y = [c[1] for c in centers[0:1]]

ax.plot(x, y, 'o')
display(fig)

# COMMAND ----------


