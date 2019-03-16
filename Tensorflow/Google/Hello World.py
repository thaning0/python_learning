import tensorflow as tf

try:
  tf.contrib.eager.enable_eager_execution()
except ValueError:
  pass  # enable_eager_execution errors after its first call

tensor = tf.constant('Hello, world!')
tensor_value = tensor.numpy() # bytes object 
print(tensor_value.decode()) # Use decode() convert bytes to string object