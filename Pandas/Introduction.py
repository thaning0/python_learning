import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# Main data strcture in pandas: DataFrame and Series
# Create Series
city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])

# Create DataFrame
cities = pd.DataFrame({'City name': city_names, 'Population': population})

# Inport csv as DataFrame
california_housing_dataframe = pd.read_csv("/Users/t/Project/python_learning/Pandas/California housing Data.csv", sep=",")

# describe shows some statistics about DataFrame
california_housing_dataframe.describe()

# head displays the first few records of a DataFrame
california_housing_dataframe.head()

# hist plots the distribution of values
california_housing_dataframe.hist('housing_median_age')
plt.show()

# Accessing Data (familiar to python dict/list)
print(type(cities['City name']))
cities['City name']

print(type(cities['City name'][1]))
cities['City name'][1]

print(type(cities[0:2]))
cities[0:2]

# Manipulating Data
# python basic arithmetic operations
population_1000 = population/1000
print(population_1000)

# Numpy functions
population_log = np.log(population)

# Series.apply apply lambda function to each value of Series
population.apply(lambda x: x > 1000000)

# DataFrames is similar
cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
cities['Population density'] = cities['Population']/cities['Area square miles']

# Example 
# Modify the cities table by adding a new boolean column that is True if and only if both of the following are True: The city is named after a saint. The city has an area greater than 50 square miles.
# When performing logical and between Series use & instead.
cities['New column'] = (cities['Area square miles'] > 50) & cities['City name'].apply(lambda name: 'San' in name)

# Indexes
city_names.index
cities.index
cities.reindex([2, 0, 1])

# reindex with numpy
cities.reindex(np.random.permutation(cities.index))

cities.reindex([0, 1, 2, 3])
