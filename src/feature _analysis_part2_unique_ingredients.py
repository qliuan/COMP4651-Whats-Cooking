# Databricks notebook source
#mount the dataset from S3
ACCESS_KEY = 
SECRET_KEY = 
ENCODED_SECRET_KEY = SECRET_KEY.replace("/", "%2F")
AWS_BUCKET_NAME = "comp4651-project-data"
MOUNT_NAME = "data"

#dbutils.fs.mount("s3a://%s:%s@%s" % (ACCESS_KEY, ENCODED_SECRET_KEY, AWS_BUCKET_NAME), "/mnt/%s" % MOUNT_NAME)
display(dbutils.fs.ls("/mnt/%s" % MOUNT_NAME))

#val mdf = spark.read.option("multiline", "true").json("train.json")
#mdf.show(false)

#df = spark.read.text("/mnt/%s/test.json" % MOUNT_NAME)

# COMMAND ----------

#Transform the original JSON dataset into dataframe
df = spark.read.option("multiline", "true").json("dbfs:/mnt/data/train.json")
display(df)

# COMMAND ----------

#Transfrom the dataframe into an RDD
rdd = df.rdd
occur = rdd.flatMap(lambda (cuisine,_,ingredients): [(cuisine, ing) for ing in ingredients])
occur.collect()

# COMMAND ----------

#self-defined function as first reduce function in MapReduce for getting unique ingredients
def counting(v):
  result=list()
  temp=set(v)
  for i in temp:
      result.append([i,v.count(i)])
  return result

# COMMAND ----------

#first round of map reduce to find the unique ingredients
uniqueIngredients=occur.map(lambda (k,v):(v,[k])).reduceByKey(lambda a,b:a+b).map(lambda (k,v): (k,counting(v))).filter(lambda (k,v): len(v)==1).filter(lambda (k,v):v[0][1]>=5)
uniqueIngredients.collect()

# COMMAND ----------

#second self-defined function for MapReduce to find unique ingredients
def integration(a):
  result=list()
  for i in range(0,len(a)/2):
    result.append([a[2*i],a[2*i+1]])
  result.sort(key=(lambda a:a[1]),reverse=True)
  return result

# COMMAND ----------

#unique ingredients
combined=uniqueIngredients.map(lambda (k,v):(v[0][0],[k,v[0][1]])).reduceByKey(lambda a,b:a+b)
combined.collect()

# COMMAND ----------

#unique ingredients
combined.map(lambda (k,v):(k,integration(v))).collect()
