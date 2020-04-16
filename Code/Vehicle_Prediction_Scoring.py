import os
import sys
import json
import pandas as pd
import subprocess
from datetime import date
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, MinMaxScaler, MinMaxScalerModel, PCA, PCAModel
from pyspark.ml.regression import LinearRegression, LinearRegressionModel
from pyspark.ml.regression import DecisionTreeRegressionModel, GBTRegressionModel, RandomForestRegressionModel
from pyspark import SparkContext
from pyspark.sql import SparkSession

os.environ["PYSPARK_PYTHON"]="/usr/bin/python3"

spark = SparkSession.builder.getOrCreate()

'''
 {
    "use_scoring" : true,
    "scoring_args" : {
        "manufacturer": "ford",
        "make": "escape",
        "condition": "excellent",
        "cylinders": 4,
        "fuel": "gas",
        "odometer": 35475,
        "title_status": "clean",
        "transmission": "automatic",
        "drive": "4wd",
        "size": "full-size",
        "type": "SUV",
        "paint_color": "grey",
        "output_file_name": "prediction_1.csv"
    }
} 
'''
with open('/bd-fs-mnt/Spark_RA3/data/Craigslist_Vehicles/LabelEncoding.json', 'r') as file:
    encoding = json.load(file)

#Reading from commandline from deployment as json
cli_input = json.loads(sys.argv[1])

#Encoding input 
for feature in cli_input:
    if feature in encoding:
        for e in encoding[feature]:
            if cli_input[feature] == e:
                cli_input[feature] = encoding[feature][e]

#Obtaining file name and removing from dictionary
output_file_name = cli_input["output_file_name"]
del cli_input["output_file_name"]

#Converting to dataframe objects
vehicle_full_df = pd.DataFrame(cli_input, index=[0])
vehicle_full_df = pd.concat([vehicle_full_df, vehicle_full_df])
vehicle_full_df = spark.createDataFrame(vehicle_full_df)

#Vectorizing input 
vectorAssembler = VectorAssembler(inputCols = ['manufacturer', 'make', 'condition', 'cylinders',
       'fuel', 'odometer', 'title_status', 'transmission', 'drive', 'size',
       'type', 'paint_color'], outputCol = 'og_features')
vector_vehicle_df = vectorAssembler.transform(vehicle_full_df)
vector_vehicle_df = vector_vehicle_df.select(['og_features'])

#Load scaled models 
scalerModel = MinMaxScalerModel.load('file:///bd-fs-mnt/Spark_RA3/models/Spark_Vehicles/Scaler.model')
scaledData = scalerModel.transform(vector_vehicle_df)

NoScale_Pca = PCAModel.load('file:///bd-fs-mnt/Spark_RA3/models/Spark_Vehicles/NoScale_Pca.model')
Scaled_Pca = PCAModel.load('file:///bd-fs-mnt/Spark_RA3/models/Spark_Vehicles/Scaled_Pca.model')

NoScale_Pca = NoScale_Pca.transform(vector_vehicle_df).select(["og_features", "features"])
Scaled_Pca = Scaled_Pca.transform(scaledData).select(["og_features", "features"])

#Loading models 
lr_model = LinearRegressionModel.load('file:///bd-fs-mnt/Spark_RA3/models/Spark_Vehicles/lr_model.model')
dtr_model = DecisionTreeRegressionModel.load('file:///bd-fs-mnt/Spark_RA3/models/Spark_Vehicles/dtr_model.model')
gbt_model = GBTRegressionModel.load('file:///bd-fs-mnt/Spark_RA3/models/Spark_Vehicles/gbt_model.model')
rf_model = RandomForestRegressionModel.load('file:///bd-fs-mnt/Spark_RA3/models/Spark_Vehicles/rfr_model.model')

#Generate prediction
lr_pred = lr_model.transform(NoScale_Pca).select('prediction').collect()[0]['prediction']
dtr_pred = dtr_model.transform(Scaled_Pca).select('prediction').collect()[0]['prediction']
gbt_pred = gbt_model.transform(Scaled_Pca).select('prediction').collect()[0]['prediction']
rfr_pred = rf_model.transform(NoScale_Pca).select('prediction').collect()[0]['prediction'
]
#Prepare output df to output predictions
output_df = pd.DataFrame()
output_df['Algorithm'] = ['Linear Regression', 'Decision Tree', 'Gradient Boosted Tree', 'Random Forest']
output_df['Result'] = [round(lr_pred,2), round(dtr_pred, 2), round(gbt_pred, 2), round(rfr_pred, 2)]
output_dict = {} 
output_dict['Linear_Regression'] = round(lr_pred, 2)
output_dict['Decision_Tree'] = round(dtr_pred, 2)
output_dict['Gradient_Boosted_Tree'] = round(gbt_pred, 2)
output_dict['Random_Forest'] = round(rfr_pred, 2)


#Saving output to project repo 
file_path = "/bd-fs-mnt/Spark_RA3/Vehicle_Price_Predictions/" + output_file_name
output_df.to_csv(file_path)