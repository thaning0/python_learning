from LR import *
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import cm, gridspec
from matplotlib import pyplot as plt
from sklearn import metrics
from tensorflow.python.data import Dataset

california_housing_dataframe = pd.read_csv("https://download.mlcc.google.cn/mledu-datasets/california_housing_train.csv", sep=",")
california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe["median_house_value"] /= 1000.0

california_housing_dataframe["rooms_per_person"] =california_housing_dataframe["total_rooms"] /california_housing_dataframe["population"]
calibration_data = train_model(learning_rate=0.1,steps=100,batch_size=10, input_feature="rooms_per_person", Dataframe=california_housing_dataframe)

plt.subplot(1, 2, 1)
plt.scatter(calibration_data["predictions"], calibration_data["targets"])
plt.subplot(1, 2, 2)
_ = california_housing_dataframe["rooms_per_person"].hist()
plt.show()
plt.close()

# 截断数据
california_housing_dataframe["clipped_feature"] = california_housing_dataframe["rooms_per_person"].apply(lambda val: min(val, 5))
calibration_data = train_model(learning_rate=0.1,steps=100,batch_size=100, input_feature="clipped_feature", Dataframe=california_housing_dataframe)

plt.subplot(1, 2, 1)
plt.scatter(calibration_data["predictions"], calibration_data["targets"])
plt.subplot(1, 2, 2)
_ = california_housing_dataframe["clipped_feature"].hist()
plt.show()
plt.close()