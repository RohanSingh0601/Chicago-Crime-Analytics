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

# ## Importing Packages

# In[ ]:


import pyspark
import pandas as pd
import seaborn as sns
import numpy as np
from pyspark.sql import Row
import plotly.express as px
from datetime import datetime
import matplotlib.pyplot as plt
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.sql.functions import col, column
from pyspark.sql.types import StructField, StructType, StringType, LongType, IntegerType, FloatType
from pyspark.sql.functions import col, column, when, countDistinct
from pyspark.sql.functions import expr
from pyspark.sql.functions import split
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, IndexToString
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from decision_tree_plot.decision_tree_parser import decision_tree_parse
from decision_tree_plot.decision_tree_plot import plot_trees
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


# In[ ]:


crime=SparkSession.builder.master("local").appName("CrimeDataAnalysis").getOrCreate()


# In[ ]:


crime.sparkContext.setCheckpointDir("~/scratch")


# ## Uploading 2019-2023 Crime Data

# In[ ]:


Data19 = crime.read.csv("./Crimes_-_2019_20231112.csv", header=True, inferSchema=True)
Data20 = crime.read.csv("./Crimes_-_2020_20231112.csv", header=True, inferSchema=True)
Data21 = crime.read.csv("./Crimes_-_2021_20231112.csv", header=True, inferSchema=True)
Data22 = crime.read.csv("./Crimes_-_2022_20231016.csv", header=True, inferSchema=True)
Data23 = crime.read.csv("./Crimes_-_2023_20231016.csv", header=True, inferSchema=True)


# ## Data Merging with 2022 - 2023

# In[ ]:


Data19_23 = Data19.union(Data20).union(Data21).union(Data22).union(Data23)
Data19_23.show(10)


# In[ ]:


print(f"Total Entries in 2019 to 2023: {Data19_23.count()}")


# ## Removing rows with null values

# In[ ]:


df = Data19_23.select("Date", "Block", "Primary Type", "Description", "Location Description", "Arrest", "Domestic",
                     "Beat", "District", "Ward", "Community Area", "Year")


# In[ ]:


df_clean = df.dropna(how = 'any')


# In[ ]:


print(f"Total Entries after cleaning in 2019 to 2023: {df_clean.count()}")


# In[ ]:


df_clean.columns


# In[ ]:


# Display the first two rows of the dataset
first_two_rows = df_clean.take(2)

# Print each row with column names for clarity
for row in first_two_rows:
    for col_name in df_clean.columns:
        print(f"{col_name}: {row[col_name]}")
    print("\n---\n")  # Separator between rows


# In[ ]:


from pyspark.sql.functions import countDistinct

# List of columns to check for cardinality
columns_to_check = ["Primary Type", "Description", "Location Description", "Ward", "Year"]

# Query to count distinct values in each column
for column in columns_to_check:
    distinct_count = df_clean.select(countDistinct(col(column)).alias(column)).collect()[0][column]
    print(f"Distinct count in {column}: {distinct_count}")


# ## Map-Reduce for data analysis

# In[ ]:


mapped_primary_type = df_clean.rdd.map(lambda row: (row["Primary Type"], 1))
reduced_primary_type = mapped_primary_type.reduceByKey(lambda a, b: a + b)
sorted_primary_type = reduced_primary_type.sortBy(lambda x: x[1], ascending=False)
primary_type_counts_sorted = sorted_primary_type.collect()

primary = [] #empty list for visualization
counts = [] #empty list for visualization

for primary_type, count in primary_type_counts_sorted:
    print(f"{primary_type}: {count}")
    primary.append(primary_type)  #the first column (primary type)
    counts.append(int(count))  #the second column (counts)


# ## Random Forest Classifier

# In[ ]:


from pyspark.sql.functions import col, when

# Cardinality reduction for 'Description' column
desc_ct = df_clean.groupBy("Description").count()
desc_ct_sorted = desc_ct.sort(desc_ct['count'], ascending=False)
desc_top_32 = desc_ct_sorted.limit(32)
desc_list = desc_top_32.select("Description").rdd.flatMap(lambda x: x).collect()

# Update 'Description' column to keep only top 32 categories
df_clean = df_clean.withColumn("Description", when(col("Description").isin(desc_list), col("Description")).otherwise("Other"))

# Cardinality reduction for 'Location Description' column
loc_ct = df_clean.groupBy("Location Description").count()
loc_ct_sorted = loc_ct.sort(loc_ct['count'], ascending=False)
loc_top_32 = loc_ct_sorted.limit(32)
loc_list = loc_top_32.select("Location Description").rdd.flatMap(lambda x: x).collect()

filtered_df = df_clean.filter(col("Description").isin(desc_list) & col("Location Description").isin(loc_list))


# #### Model Setup

# In[ ]:


from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

df_model = filtered_df.select("Primary Type", "Description", "Location Description", "Arrest", "Domestic", "Year")
df_model = df_model.withColumn("Arrest", when(col("Arrest") == "true", 1).otherwise(0))
df_model = df_model.withColumn("Domestic", when(col("Domestic") == "true", 1).otherwise(0))

# Indexing and assembling
inputs = ["Primary Type", "Description", "Location Description"]
outputs = [input_col + "_index" for input_col in inputs]
indexers = [StringIndexer(inputCol=input_col, outputCol=output_col).fit(df_model) for input_col, output_col in zip(inputs, outputs)]
assembler = VectorAssembler(inputCols=[indexer.getOutputCol() for indexer in indexers] + ["Domestic"], outputCol="features")

# Random Forest Classifier
rf = RandomForestClassifier(labelCol="Arrest", featuresCol="features")
# Pipeline
pipeline = Pipeline(stages=indexers + [assembler, rf])

# Splitting data
train_data, validation_data, test_data = df_model.randomSplit([0.60, 0.20, 0.20], seed=17)

train_data.cache()
test_data.cache()

# Parameter grid for model tuning
paramGrid = ParamGridBuilder()     .addGrid(rf.numTrees, [10, 15])     .addGrid(rf.maxDepth, [5, 7])     .addGrid(rf.maxBins, [32, 40])     .build()

# Evaluator
evaluator = MulticlassClassificationEvaluator(labelCol="Arrest", predictionCol="prediction", metricName="accuracy")

# Cross-validation
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=3)

# Fit model using training data
cvModel = crossval.fit(train_data)

# Use validation data for hyperparameter tuning
validation_predictions = cvModel.transform(validation_data)

# Select the best model
best_model = cvModel.bestModel

# Extract best hyperparameters
best_numTrees = best_model.stages[-1]._java_obj.getNumTrees()
best_maxDepth = best_model.stages[-1]._java_obj.getMaxDepth()
best_maxBins = best_model.stages[-1]._java_obj.getMaxBins()

# Evaluate the best model on validation data
validation_accuracy = evaluator.evaluate(validation_predictions)

# Create a DataFrame for hyperparameters and their evaluation metrics
import pandas as pd
hyperparams_eval_df = pd.DataFrame({
    "numTrees": [best_numTrees],
    "maxDepth": [best_maxDepth],
    "maxBins": [best_maxBins],
    "validation_accuracy": [validation_accuracy]
})

# Save the DataFrame to a CSV file
output_path = "./RF_Hyperparameters_Evaluation.csv"
hyperparams_eval_df.to_csv(output_path, index=False)

# Use the best model to make predictions on the test data
predictions = best_model.transform(test_data)

# Evaluate accuracy
accuracy = evaluator.evaluate(predictions)
print("Test Accuracy = %g" % accuracy)

train_data.unpersist()
test_data.unpersist()


# #### Model Interpretation

# In[ ]:


best_rf_model = cvModel.bestModel.stages[-1]

importances = best_rf_model.featureImportances


feature_names = [indexer.getOutputCol() for indexer in indexers] + ["Domestic"]


importances_with_names = [(feature_names[i], importance) for i, importance in enumerate(importances)]


sorted_importances = sorted(importances_with_names, key=lambda x: x[1], reverse=True)


names, values = zip(*sorted_importances)

plt.figure(figsize=(10, 6))
plt.barh(names, values)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importances in Random Forest Model')
plt.gca().invert_yaxis()  # To display the highest importance at the top
plt.show()


# In[ ]:


from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Precision
precision_evaluator = MulticlassClassificationEvaluator(
    labelCol="Arrest", predictionCol="prediction", metricName="weightedPrecision")
precision = precision_evaluator.evaluate(predictions)
print("Precision = %g" % precision)

# Recall
recall_evaluator = MulticlassClassificationEvaluator(
    labelCol="Arrest", predictionCol="prediction", metricName="weightedRecall")
recall = recall_evaluator.evaluate(predictions)
print("Recall = %g" % recall)

# F1 Score
f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="Arrest", predictionCol="prediction", metricName="f1")
f1_score = f1_evaluator.evaluate(predictions)
print("F1 Score = %g" % f1_score)

