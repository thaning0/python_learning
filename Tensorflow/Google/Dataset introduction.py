# 参考 https://zhuanlan.zhihu.com/p/30751039 作者:何之源
# Hello world
import tensorflow as tf
import tensorflow.contrib.eager as tfe
# enable eager mode
# tfe.enable_eager_execution()

hello = ['h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd']

# Create dataset
# Create a dataset by a list, dataset also can be create by numpy.array, numpy.matrix, tuple, dict
# there are some other way to create datasets, see document 
dataset = tf.data.Dataset.from_tensor_slices(hello)

# Transform Dataset to a new dataset (map, batch, shuffle, repeat)
dataset = tf.data.Dataset.from_tensor_slices([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

# map() maps the dataset by a function
dataset_map = dataset.map(lambda x: x + 1) # 2.0, 3.0, 4.0, 5.0, 6.0 7.0

# batch() batchs dataset 
dataset_batch = dataset.batch(2)

# shuffle() randomly resorts dataset (similar to the random reindex in pandas Dataframe)
dataset_shuffle = dataset.shuffle(100000)

# repeat(n) repeats dataset n times, repeat() repeats dataset infinite times 
dataset_repeat = dataset.repeat(2) # 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 1.0, 2.0, 3.0, 4.0, 5.0, 6.0


# Extract data from dataset
# Iterator
# A one shot iterator returns a value to dataset
# There are also some other iterator to extract data
iterator = dataset.make_one_shot_iterator()
iterator1 = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
test = iterator.initializer()
# non eager mode
with tf.Session() as sess:
    sess.run(iterator.initializer())
    for i in range(11):
        print(sess.run(one_element))

# eager mode
for one_element in tfe.Iterator(dataset):
    print(one_element)


