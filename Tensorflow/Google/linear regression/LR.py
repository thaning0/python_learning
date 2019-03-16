import math
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import cm, gridspec
from matplotlib import pyplot as plt
from sklearn import metrics
from tensorflow.python.data import Dataset


# 定于输入函数，对数据进行预处理, 并创建iterator, 返回数据和标签的元组
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
  """
  Trains a linear regression model of one feature.
  Args:
    features: pandas DataFrame of features
    targets: pandas DataFrame of targets
    batch_size: Size of batches to be passed to the model
    shuffle: True or False. Whether to shuffle the data.
    num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
  Returns:
    Tuple of (features, labels) for next data batch
  """
  # 把 pandas 数据集转换为 numpy 字典 (dict comprehensions 字典推导式)
  features = {key:np.array(value) for key,value in dict(features).items()}                                          
  # 构建tenserflow数据集，并将数据做拆分重复处理
  # from_tensor_slices() creates a Dataset whose elements are slices of the given tensors.
  ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
  ds = ds.batch(batch_size).repeat(num_epochs)
  
  # 对数据进行随机处理
  if shuffle:
    ds = ds.shuffle(buffer_size=10000)
  
  # 返回数据
  features, labels = ds.make_one_shot_iterator().get_next()
  return features, labels

def train_model(learning_rate, steps, batch_size, input_feature, Dataframe):
  """Trains a linear regression model.
  
  Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    input_feature: A `string` specifying a column from `Dataframe`
      to use as input feature.
    Dataframe: A pandas Dataframe
      
  Returns:
    A Pandas `DataFrame` containing targets and the corresponding predictions done
    after training the model.
  """
  
  periods = 10
  steps_per_period = steps / periods

  my_feature = input_feature
  my_feature_data = Dataframe[[my_feature]].astype('float32')
  my_label = "median_house_value"
  targets = Dataframe[my_label].astype('float32')

  # Create input functions.
  training_input_fn = lambda: my_input_fn(my_feature_data, targets, batch_size=batch_size)
  predict_training_input_fn = lambda: my_input_fn(my_feature_data, targets, num_epochs=1, shuffle=False)
  
  # Create feature columns.
  feature_columns = [tf.feature_column.numeric_column(my_feature)]
    
  # Create a linear regressor object.
  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  linear_regressor = tf.estimator.LinearRegressor(
      feature_columns=feature_columns,
      optimizer=my_optimizer
  )

  # Set up to plot the state of our model's line each period.
  plt.figure(figsize=(15, 6))
  plt.subplot(1, 2, 1)
  plt.title("Learned Line by Period")
  plt.ylabel(my_label)
  plt.xlabel(my_feature)
  sample = Dataframe.sample(n=300)
  plt.scatter(sample[my_feature], sample[my_label])
  colors = [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print("Training model...")
  print("RMSE (on training data):")
  root_mean_squared_errors = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    linear_regressor.train(
        input_fn=training_input_fn,
        steps=steps_per_period,
    )
    # Take a break and compute predictions.
    predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
    predictions = np.array([item['predictions'][0] for item in predictions])
    
    # Compute loss.
    root_mean_squared_error = math.sqrt(
      metrics.mean_squared_error(predictions, targets))
    # Occasionally print the current loss.
    print("  period %02d : %0.2f" % (period, root_mean_squared_error))
    # Add the loss metrics from this period to our list.
    root_mean_squared_errors.append(root_mean_squared_error)
    # Finally, track the weights and biases over time.
    # Apply some math to ensure that the data and line are plotted neatly.
    y_extents = np.array([0, sample[my_label].max()])
    
    weight = linear_regressor.get_variable_value('linear/linear_model/%s/weights' % input_feature)[0]
    bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')
    
    x_extents = (y_extents - bias) / weight
    x_extents = np.maximum(np.minimum(x_extents,
                                      sample[my_feature].max()),
                           sample[my_feature].min())
    y_extents = weight * x_extents + bias
    plt.plot(x_extents, y_extents, color=colors[period]) 
  print("Model training finished.")

  # Output a graph of loss metrics over periods.
  plt.subplot(1, 2, 2)
  plt.ylabel('RMSE')
  plt.xlabel('Periods')
  plt.title("Root Mean Squared Error vs. Periods")
  plt.tight_layout()
  plt.plot(root_mean_squared_errors)
  plt.savefig('LR.png')
  plt.close()

  # Create a table with calibration data.
  calibration_data = pd.DataFrame()
  calibration_data["predictions"] = pd.Series(predictions)
  calibration_data["targets"] = pd.Series(targets)
  print(calibration_data.describe())

  print("Final RMSE (on training data): %0.2f" % root_mean_squared_error)
  
  return calibration_data

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.ERROR)
  pd.options.display.max_rows = 10
  pd.options.display.float_format = '{:.1f}'.format
  
  # 加载数据集（加利福尼亚州 1990 年的人口普查数据）
  california_housing_dataframe = pd.read_csv("https://download.mlcc.google.cn/mledu-datasets/california_housing_train.csv", sep=",")
  
  # 随机打乱数据顺序并把 median_house_value 的单位调整为千
  california_housing_dataframe = california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))
  california_housing_dataframe["median_house_value"] /= 1000.0
  california_housing_dataframe.describe()
  
  # 用 total_rooms 预测 median_house_value （线性回归）
  # 定义输入的特征
  my_feature = california_housing_dataframe[["total_rooms"]]
  
  # 定义特征和目标
  feature_columns = [tf.feature_column.numeric_column("total_rooms")]
  targets = california_housing_dataframe["median_house_value"]
  
  # 使用最速下降训练模型
  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
  
  # 裁剪梯度, 防止梯度太大
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0) 
  
  # 设置线性回归模型
  linear_regressor = tf.estimator.LinearRegressor(feature_columns=feature_columns,optimizer=my_optimizer)
  
  # 训练模型 参考https://www.tensorflow.org/guide/premade_estimators
  # _ Ignore the outupt ot linear_regressor.train
  _ = linear_regressor.train(input_fn = lambda:my_input_fn(my_feature, targets, batch_size=10),steps=1000)
  
  # Create an input function for predictions.
  # Note: Since we're making just one prediction for each example, we don't 
  # need to repeat or shuffle the data here.
  prediction_input_fn =lambda: my_input_fn(my_feature, targets, num_epochs=1, shuffle=False)
  
  # Call predict() on the linear_regressor to make predictions. Returns a python iterable, yielding a dictionary.
  predictions = linear_regressor.predict(input_fn=prediction_input_fn)
  
  # Format predictions as a NumPy array, so we can calculate error metrics.
  predictions = np.array([item['predictions'][0] for item in predictions])
  
  # Print Mean Squared Error and Root Mean Squared Error.
  mean_squared_error = metrics.mean_squared_error(predictions, targets)
  root_mean_squared_error = math.sqrt(mean_squared_error)
  print("Mean Squared Error (on training data): %0.3f" % mean_squared_error)
  print("Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error)
  
  calibration_data = pd.DataFrame()
  calibration_data["predictions"] = pd.Series(predictions)
  calibration_data["targets"] = pd.Series(targets)
  print(calibration_data.describe())
  
  sample = california_housing_dataframe.sample(n=300)
  
  # Get the min and max total_rooms values.
  x_0 = sample["total_rooms"].min()
  x_1 = sample["total_rooms"].max()
  
  # Retrieve the final weight and bias generated during training.
  weight = linear_regressor.get_variable_value('linear/linear_model/total_rooms/weights')[0]
  bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')
  
  # Get the predicted median_house_values for the min and max total_rooms values.
  y_0 = weight * x_0 + bias 
  y_1 = weight * x_1 + bias
  
  # Plot our regression line from (x_0, y_0) to (x_1, y_1).
  plt.plot([x_0, x_1], [y_0, y_1], c='r')
  
  # Label the graph axes.
  plt.ylabel("median_house_value")
  plt.xlabel("total_rooms")
  
  # Plot a scatter plot from our data sample.
  plt.scatter(sample["total_rooms"], sample["median_house_value"])
  
  # save graph.
  plt.savefig("linearregression.png")

