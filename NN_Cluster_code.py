#!/usr/bin/env python
# coding: utf-8

# <h1 align="center"><strong>DS/CMPSC 410 - Mini Project</strong></h1>
# <h2 align="center"><strong>Crime Analysis in Chicago City.</strong></h2>
# 
# ## Instructor: Professor Romit Maulik
# 
# ## Team Members:
# ### - Sai Sanwariya Narayan
# ### - Nikhil Melligeri
# ### - Shafwat Mustafa 
# ### - Rohan Singh 
# ### - Shengdi You
# ### - Daniel Gao
# ### - Nathan Quint

# In[2]:


import pyspark
import pandas as pd
import numpy as np
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, StringType, LongType, IntegerType, FloatType
from pyspark.sql.functions import col, column, when, countDistinct
from pyspark.sql.functions import expr
from pyspark.sql.functions import split
import matplotlib.pyplot as plt
from pyspark.sql import Row
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, IndexToString
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from decision_tree_plot.decision_tree_parser import decision_tree_parse
from decision_tree_plot.decision_tree_plot import plot_trees
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import StandardScaler
from pyspark.ml.classification import MultilayerPerceptronClassifier


crime=SparkSession.builder.appName("CrimeDataAnalysis").getOrCreate()


crime.sparkContext.setCheckpointDir("~/scratch")

Data22 = crime.read.csv("./Crimes_-_2022_20231016.csv", header=True, inferSchema=True)


Data23 = crime.read.csv("./Crimes_-_2023_20231016.csv", inferSchema = True, header = True)


Data19 = crime.read.csv("./Crimes_-_2019_20231112.csv", header=True, inferSchema=True)
Data20 = crime.read.csv("./Crimes_-_2020_20231112.csv", header=True, inferSchema=True)
Data21 = crime.read.csv("./Crimes_-_2021_20231112.csv", header=True, inferSchema=True)


Data22_23 = Data22.union(Data23).union(Data19).union(Data20).union(Data21)



df = Data22_23.select("Date", "Block", "Primary Type", "Description", "Location Description", "Arrest", "Domestic", 
                     "Beat", "District", "Ward", "Community Area", "Year")


df_clean = df.dropna(how = 'any')

dt_df = df_clean.select("Primary Type", "Description", "Location Description", "Arrest", "Domestic", "Ward", "Year")


dt_df = dt_df.withColumn("Arrest", when(col("Arrest") == "true", 1).otherwise(0))
dt_df = dt_df.withColumn("Domestic", when(col("Domestic") == "true", 1).otherwise(0))

desc_ct = dt_df.groupBy("Description").count() 
desc_ct_sorted = desc_ct.sort(desc_ct['count'], ascending = False)
desc_top_32 = desc_ct_sorted.limit(32) # limit to 32 bins 
desc_list = desc_top_32.select("Description").rdd.flatMap(lambda x: x).collect() # make a list of top 32 descriptions 
loc_ct = dt_df.groupBy('Location Description').count()
loc_ct_sorted = loc_ct.sort(loc_ct['count'], ascending = False)
loc_top_32 = loc_ct_sorted.limit(32)
loc_list = loc_top_32.select("Location Description").rdd.flatMap(lambda x: x).collect()
filtered_df_1 = dt_df.filter(col("Description").isin(desc_list))
filtered_df_2 = filtered_df_1.filter(col("Location Description").isin(loc_list))

inputs = ["Primary Type", "Description", "Location Description"]
outputs = ["index1","index2","index3"]

indexer = StringIndexer(inputCols = inputs, outputCols = outputs).fit(filtered_df_2)

transformed_data = indexer.transform(filtered_df_2)

input_features = ['index1', 'index2', 'index3', 'Domestic', 'Ward', 'Year']
assembler = VectorAssembler(inputCols=input_features, outputCol="features")
vectorized_data = assembler.transform(transformed_data)


df_dt_data = vectorized_data.select("features",'Arrest')


# scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
# scaler_model = scaler.fit(df_dt_data)
# data = scaler_model.transform(df_dt_data)


# (training_data, test_data) = data.randomSplit([0.8, 0.2], seed=123)

# layers = [6, 16, 8, 10]  

# trainer = MultilayerPerceptronClassifier(maxIter=100, 
#                                          layers=layers, 
#                                          featuresCol='scaled_features',
#                                          labelCol='Arrest', blockSize=128, seed=1234)

# model = trainer.fit(training_data)


# predictions = model.transform(test_data)


# evaluator = MulticlassClassificationEvaluator(labelCol='Arrest',predictionCol='prediction',metricName='f1')
# ann_f1 = evaluator.evaluate(predictions)


scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
scaler_model = scaler.fit(df_dt_data)
data = scaler_model.transform(df_dt_data)
hyperparams_eval_df = pd.DataFrame( columns = ['blockSize', 'layer', 'training f1','validation f1', 'Best Model'] )
blockSize = [32, 64, 128, 256]
layers = [[6, 16, 8, 10],[6, 12, 8, 10],[6, 10, 10, 10], [6, 20, 20, 10], [6, 16, 16, 10], 
          [6, 8, 8, 10], [6, 8, 10, 10], [6, 12, 12, 10], [6, 16, 20, 10], [6, 8, 20, 10]]
index =0
highest_validation_f1 = 0
(training_data, test_data, validation_data) = data.randomSplit([0.6, 0.2, 0.2], seed=123)
training_data.persist()
test_data.persist()
validation_data.persist()
for block in blockSize:
    for layer in layers:
        trainer = MultilayerPerceptronClassifier(maxIter=100, 
                                         layers=layer, 
                                         featuresCol='scaled_features',
                                         labelCol='Arrest', blockSize=block, seed=1234)
        model = trainer.fit(training_data)
        training_predictions = model.transform(training_data)
        validation_predictions = model.transform(validation_data)
        evaluator = MulticlassClassificationEvaluator(labelCol='Arrest',predictionCol='prediction',metricName='f1')
        training_f1 = evaluator.evaluate(training_predictions)
        validation_f1 = evaluator.evaluate(validation_predictions)
        hyperparams_eval_df.loc[index] = [block, str(layer), training_f1, validation_f1, 0]
        index += 1
        if validation_f1 > highest_validation_f1 :
            best_model = model
            best_evaluator = evaluator 
            best_block = block
            best_layer = str(layer)
            best_index = index -1
            highest_validation_f1 = validation_f1
            highest_training_f1 = training_f1
    
            
testing_predictions = best_model.transform(test_data)
testing_f1 = best_evaluator.evaluate(testing_predictions)
print("Testing accuracy f1 score: ", testing_f1)

print('best block: ', best_block, 'best layer:', best_layer)

hyperparams_eval_df.loc[best_index]=[best_block, best_layer, highest_training_f1, highest_validation_f1, 1000]
output_path = "./NNHPT_cluster.csv"
hyperparams_eval_df.to_csv(output_path)  


crime.stop()

