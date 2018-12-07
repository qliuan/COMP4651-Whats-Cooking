# Databricks notebook source
ACCESS_KEY = ""
SECRET_KEY = ""
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

rdd = df.rdd
rdd.collect()

# COMMAND ----------

occur = rdd.flatMap(lambda (cuisine,_,ingredients): [(cuisine, ing) for ing in ingredients])
occur.collect()

# COMMAND ----------

#To make rdd to have ingredients as key and country as values
inverseOccur = occur.map(lambda (cuisine, ingredient): (ingredient, cuisine))

# COMMAND ----------

# Inverse Custom Schema for cuisine and ingredients
from pyspark.sql.types import *
inverseCustomSchema = StructType([ \
    StructField("ingredient", StringType(), True), \
    StructField("cuisines", StringType(), True)])

# COMMAND ----------

#'occur' rdd: (cuisine, ingredient)
#'cuisineCount' rdd: (cuisine, count)
#First look at the case for each cuisine
#Calculate the number for each cuisine
cuisineCount = (occur
                .map(lambda (cui, ing): (cui, 1))
                .countByKey())
print cuisineCount

# COMMAND ----------

import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt


# COMMAND ----------

print df[["cuisine", "Ingredients"]].take(20)

# COMMAND ----------

def toString (x, y):
  return x + '_' + y;

# COMMAND ----------

#to make all ingredients matched to the cuisine, prepare for the dataframe
preDFRDD2 = occur.reduceByKey(toString).sortByKey()

# COMMAND ----------

from pyspark.sql.types import *
# Custom Schema for cuisine and ingredients
customSchema = StructType([ \
    StructField("cuisine", StringType(), True), \
    StructField("ingredients", StringType(), True)])

# COMMAND ----------

#change the preDrRDD to dataframe
cuiIngDF2 = spark.createDataFrame(preDFRDD2, customSchema)

# COMMAND ----------

display(cuiIngDF2)

# COMMAND ----------

cuiIngDF2.select('cuisine').collect()

# COMMAND ----------

#brazilian
pdsDF = cuiIngDF2.toPandas()
text00 = pdsDF.ingredients[0]
wordcloud00 = WordCloud(collocations = False, regexp = r"[^_]+").generate(text00)
plt.figure(figsize=(10,8))
plt.imshow(wordcloud00, interpolation='bilinear')
plt.axis("off")
plt.show()
display()

# COMMAND ----------

sorted(wordcloud00.words_.items(), key = lambda x : -x[1])

# COMMAND ----------

#chinese
text03 = pdsDF.ingredients[3]
wordcloud03 = WordCloud(collocations = False, background_color = "white", regexp = r"[^_]+").generate(text03)
plt.figure(figsize=(10,8))
plt.imshow(wordcloud03, interpolation='bilinear')
plt.axis("off")
plt.show()
display()

# COMMAND ----------

sorted(wordcloud03.words_.items(), key = lambda x : -x[1])

# COMMAND ----------

type(wordcloud03.words_.items())

# COMMAND ----------

ingRDD = sc.parallelize(wordcloud03.words_.items(), numSlices = 4)

# COMMAND ----------

ingRDD.takeOrdered(100, key = lambda x : -x[1])

# COMMAND ----------

sumFreq = ingRDD.map(lambda (ing, relFreq): relFreq). sum()
freqRDD = ingRDD.map(lambda (ing, relFreq): (ing, relFreq/sumFreq)).sortBy(lambda x: x[1], False)
freqRDD.take(100)

# COMMAND ----------

ingFreqSchema = StructType([ \
    StructField("ingredients", StringType(), True), \
    StructField("frequency", DoubleType(), True)])
ingFreqDF = spark.createDataFrame(freqRDD, ingFreqSchema)
partIngFreqDF = ingFreqDF.limit(20)

# COMMAND ----------

#matplot plt.bar
pdsIngFreqDF = partIngFreqDF.toPandas()
plt.figure(figsize=(30,20))
pdsIngFreqDF.plot.bar()
plt.xticks(rotation=50)
plt.xlabel("Ingredient")
plt.ylabel("Frequency")
fig = plt.show()
display(fig)

# COMMAND ----------

import seaborn as sns
sns.set()
cmap = sns.cubehelix_palette(20)

# COMMAND ----------

#Python plt.bar
pdsIngFreqDF = partIngFreqDF.toPandas()
pdsIngFreqDF.plot.bar(color = cmap)
plt.xticks(rotation=50)
plt.xlabel("Ingredient")
plt.ylabel("Frequency")
fig = plt.show()
display(fig)

# COMMAND ----------

#seaborn, color = cmap/
pdsIngFreqDF = partIngFreqDF.toPandas()
plt.figure(figsize=(16,11))
ax = sns.barplot("ingredients", y="frequency", data=pdsIngFreqDF, palette = "Blues_d")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize = "medium")
plt.xlabel("Ingredients")
plt.ylabel("Frequency")
fig = plt.show()
display(fig)

# COMMAND ----------

#Before each wordcloud is generated, run cmd 31-35 and change the file name in 32. Here we just run the cell that are shown in our reporr
ACCESS_KEY_IMG = ""
SECRET_KEY_IMG = ""
ENCODED_SECRET_KEY_IMG = SECRET_KEY.replace("/", "%2F")
AWS_BUCKET_NAME_IMG = "xyangaq"
MOUNT_NAME_IMG = "image1"

#dbutils.fs.mount("s3a://%s:%s@%s" % (ACCESS_KEY, ENCODED_SECRET_KEY, AWS_BUCKET_NAME), "/mnt/%s" % MOUNT_NAME)
display(dbutils.fs.ls("/mnt/%s" % MOUNT_NAME_IMG))

#val mdf = spark.read.option("multiline", "true").json("train.json")
#mdf.show(false)

#df = spark.read.text("/mnt/%s/test.json" % MOUNT_NAME)

# COMMAND ----------

image = sc.binaryFiles("dbfs:/mnt/image1/SouthKoreaFlag.jpg")

# COMMAND ----------

imgList = image.values().collect()
rawdata = imgList[0]

# COMMAND ----------

from StringIO import StringIO
mask = np.asarray(Image.open(StringIO(rawdata)))
mask

# COMMAND ----------

ingredients = occur.map(lambda (cui,ing): ("world", ing))
ingredients.collect()

# COMMAND ----------

IngRDD2 = ingredients.reduceByKey(toString)

# COMMAND ----------

IngDF2 = spark.createDataFrame(preDFRDD2, customSchema)

# COMMAND ----------

#world
worldPdsDF = IngDF2.toPandas()
worldText = worldPdsDF.ingredients[0]
worldImg = WordCloud(background_color="white", mode="RGBA", max_words=5000, mask=mask, contour_color='gold', collocations = False, regexp = r"[^_]+")

# Generate a wordcloud
worldImg.generate(worldText)
image_colors = ImageColorGenerator(mask)


# show
plt.imshow(worldImg.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")

world = plt.show()
display(world)

# COMMAND ----------

#00: Brazilian
# Create a word cloud image
plateImg = WordCloud(background_color="white", max_words=3000, mask=mask,
               contour_width=1, contour_color='gold', collocations = False, regexp = r"[^_]+")

# Generate a wordcloud
plateImg.generate(text00)

# show
plt.imshow(plateImg, interpolation='bilinear')
plt.axis("off")

plate = plt.show()
display(plate)

# COMMAND ----------

#03: Chinese
# Generate a word cloud image
flagImg1 = WordCloud(background_color="white", mode="RGBA", max_words=2000, mask=mask, collocations = False, regexp = r"[^_]+").generate(text03)

# create coloring from image
image_colors = ImageColorGenerator(mask)
plt.imshow(flagImg1.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")

# store to file
flag = plt.show()
display(flag)

# COMMAND ----------

#british
text01 = pdsDF.ingredients[1]
wordcloud01 = WordCloud(collocations = False, regexp = r"[^_]+").generate(text01)
# Generate a word cloud image
flagImg2 = WordCloud(background_color="white", mode="RGBA", max_words=2000, mask=mask, collocations = False, regexp = r"[^_]+").generate(text01)

# create coloring from image
image_colors = ImageColorGenerator(mask)

plt.imshow(flagImg2.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")

# store to file
flag = plt.show()
display(flag)

# COMMAND ----------

#south korea
text012 = pdsDF.ingredients[12]
wordcloud012 = WordCloud(collocations = False, regexp = r"[^_]+").generate(text012)
# Generate a word cloud image
flagImg3 = WordCloud(background_color="white", mode="RGBA", max_words=2000, mask=mask, collocations = False, regexp = r"[^_]+").generate(text012)

# create coloring from image
image_colors = ImageColorGenerator(mask)

plt.imshow(flagImg3.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")

# store to file
flag = plt.show()
display(flag)

# COMMAND ----------

#french
text05 = pdsDF.ingredients[5]
wordcloud05 = WordCloud(collocations = False, regexp = r"[^_]+").generate(text05)
# Generate a word cloud image
flagImg4 = WordCloud(background_color="white", mode="RGBA", max_words=2000, mask=mask, collocations = False, regexp = r"[^_]+").generate(text05)

# create coloring from image
image_colors = ImageColorGenerator(mask)

plt.imshow(flagImg4.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")

# store to file
flag = plt.show()
display(flag)

# COMMAND ----------

#india
text07 = pdsDF.ingredients[7]
wordcloud07 = WordCloud(collocations = False, regexp = r"[^_]+").generate(text07)
# Generate a word cloud image
flagImg5 = WordCloud(background_color="white", mode="RGBA", max_words=2000, mask=mask, collocations = False, regexp = r"[^_]+").generate(text07)

# create coloring from image
image_colors = ImageColorGenerator(mask)

plt.imshow(flagImg5.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")

# store to file
flag = plt.show()
display(flag)

# COMMAND ----------

#filipino
text04 = pdsDF.ingredients[4]
wordcloud04 = WordCloud(collocations = False, regexp = r"[^_]+").generate(text04)
# Generate a word cloud image
flagImg6 = WordCloud(background_color="white", mode="RGBA", max_words=2000, mask=mask, collocations = False, regexp = r"[^_]+").generate(text04)

# create coloring from image
image_colors = ImageColorGenerator(mask)

plt.imshow(flagImg6.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")
# store to file
flag = plt.show()
display(flag)

# COMMAND ----------

#italian
text09 = pdsDF.ingredients[9]
wordcloud09 = WordCloud(collocations = False, regexp = r"[^_]+").generate(text09)
# Generate a word cloud image
flagImg7 = WordCloud(background_color="white", mode="RGBA", max_words=2000, mask=mask, collocations = False, regexp = r"[^_]+").generate(text09)

# create coloring from image
image_colors = ImageColorGenerator(mask)

plt.imshow(flagImg7.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")
# store to file
flag = plt.show()
display(flag)

# COMMAND ----------

#greek
text06 = pdsDF.ingredients[6]
wordcloud06 = WordCloud(collocations = False, regexp = r"[^_]+").generate(text06)
# Generate a word cloud image
flagImg8 = WordCloud(background_color="white", mode="RGBA", max_words=2000, mask=mask, collocations = False, regexp = r"[^_]+").generate(text06)

# create coloring from image
image_colors = ImageColorGenerator(mask)

plt.imshow(flagImg8.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")
# store to file
flag = plt.show()
display(flag)

# COMMAND ----------

#irish
text08 = pdsDF.ingredients[8]
wordcloud08 = WordCloud(collocations = False, regexp = r"[^_]+").generate(text08)
# Generate a word cloud image
flagImg9 = WordCloud(background_color="white", mode="RGBA", max_words=2000, mask=mask, collocations = False, regexp = r"[^_]+").generate(text08)

# create coloring from image
image_colors = ImageColorGenerator(mask)

plt.imshow(flagImg9.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")
# store to file
flag = plt.show()
display(flag)

# COMMAND ----------

#jamanican
text10 = pdsDF.ingredients[10]
wordcloud10 = WordCloud(collocations = False, regexp = r"[^_]+").generate(text10)
# Generate a word cloud image
flagImg10 = WordCloud(background_color="white", mode="RGBA", max_words=2000, mask=mask, collocations = False, regexp = r"[^_]+").generate(text10)

# create coloring from image
image_colors = ImageColorGenerator(mask)

plt.imshow(flagImg10.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")
# store to file
flag = plt.show()
display(flag)

# COMMAND ----------

#japan
text11 = pdsDF.ingredients[11]
wordcloud11 = WordCloud(collocations = False, regexp = r"[^_]+").generate(text11)
# Generate a word cloud image
flagImg11 = WordCloud(background_color="white", mode="RGBA", max_words=2000, mask=mask, collocations = False, regexp = r"[^_]+").generate(text11)

# create coloring from image
image_colors = ImageColorGenerator(mask)

plt.imshow(flagImg11.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")
# store to file
flag = plt.show()
display(flag)

# COMMAND ----------

#mexican
text13 = pdsDF.ingredients[13]
wordcloud13 = WordCloud(collocations = False, regexp = r"[^_]+").generate(text13)
# Generate a word cloud image
flagImg12 = WordCloud(background_color="white", mode="RGBA", max_words=2000, mask=mask, collocations = False, regexp = r"[^_]+").generate(text13)

# create coloring from image
image_colors = ImageColorGenerator(mask)

plt.imshow(flagImg12.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")
# store to file
flag = plt.show()
display(flag)

# COMMAND ----------

#morrocan
text14 = pdsDF.ingredients[14]
wordcloud14 = WordCloud(collocations = False, regexp = r"[^_]+").generate(text14)
# Generate a word cloud image
flagImg13 = WordCloud(background_color="white", mode="RGBA", max_words=2000, mask=mask, collocations = False, regexp = r"[^_]+").generate(text14)

# create coloring from image
image_colors = ImageColorGenerator(mask)

plt.imshow(flagImg13.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")
# store to file
flag = plt.show()
display(flag)

# COMMAND ----------

#russian
text15 = pdsDF.ingredients[15]
wordcloud15 = WordCloud(collocations = False, regexp = r"[^_]+").generate(text15)
# Generate a word cloud image
flagImg14 = WordCloud(background_color="white", mode="RGBA", max_words=2000, mask=mask, collocations = False, regexp = r"[^_]+").generate(text15)

# create coloring from image
image_colors = ImageColorGenerator(mask)

plt.imshow(flagImg14.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")
# store to file
flag = plt.show()
display(flag)

# COMMAND ----------

#southern_us
text16 = pdsDF.ingredients[16]
wordcloud16 = WordCloud(collocations = False, regexp = r"[^_]+").generate(text16)
# Generate a word cloud image
flagImg15 = WordCloud(background_color="white", mode="RGBA", max_words=2000, mask=mask, collocations = False, regexp = r"[^_]+").generate(text16)

# create coloring from image
image_colors = ImageColorGenerator(mask)

plt.imshow(flagImg15.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")
# store to file
flag = plt.show()
display(flag)

# COMMAND ----------

#spain
text17 = pdsDF.ingredients[17]
wordcloud17 = WordCloud(collocations = False, regexp = r"[^_]+").generate(text17)
# Generate a word cloud image
flagImg16 = WordCloud(background_color="white", mode="RGBA", max_words=2000, mask=mask, collocations = False, regexp = r"[^_]+").generate(text17)

# create coloring from image
image_colors = ImageColorGenerator(mask)
plt.figure(figsize=[15,8])
plt.imshow(flagImg16.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")
# store to file
flag = plt.show()
display(flag)

# COMMAND ----------

#thailand
text18 = pdsDF.ingredients[18]
wordcloud18 = WordCloud(collocations = False, regexp = r"[^_]+").generate(text18)
# Generate a word cloud image
flagImg17 = WordCloud(background_color="white", mode="RGBA", max_words=2000, mask=mask, collocations = False, regexp = r"[^_]+").generate(text18)

# create coloring from image
image_colors = ImageColorGenerator(mask)
plt.figure(figsize=[15,8])
plt.imshow(flagImg17.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")
# store to file
flag = plt.show()
display(flag)

# COMMAND ----------

#vietnam
text19 = pdsDF.ingredients[19]
wordcloud19 = WordCloud(collocations = False, regexp = r"[^_]+").generate(text19)
# Generate a word cloud image
flagImg18 = WordCloud(background_color="white", mode="RGBA", max_words=2000, mask=mask, collocations = False, regexp = r"[^_]+").generate(text19)

# create coloring from image
image_colors = ImageColorGenerator(mask)
plt.figure(figsize=[15,8])
plt.imshow(flagImg18.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")
# store to file
flag = plt.show()
display(flag)

# COMMAND ----------

#brazilian
text00 = pdsDF.ingredients[0]
wordcloud00 = WordCloud(collocations = False, regexp = r"[^_]+").generate(text00)
# Generate a word cloud image
flagImg19 = WordCloud(background_color="white", mode="RGBA", max_words=2000, mask=mask, collocations = False, regexp = r"[^_]+").generate(text00)

# create coloring from image
image_colors = ImageColorGenerator(mask)
plt.figure(figsize=[15,8])
plt.imshow(flagImg19.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")
# store to file
flag = plt.show()
display(flag)
