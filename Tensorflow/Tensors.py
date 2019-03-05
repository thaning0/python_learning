import tensorflow as tf
try:
    # With eager execution enabled, TensorFlow functions execute operations immediately (as opposed to adding to a graph to be executed later in a tf.Session) and return concrete values (as opposed to symbolic references to a node in a computational graph).
    tf.contrib.eager.enable_eager_execution()
    print("TF imported with eager execution!")
except ValueError:
    print("TF already imported with eager execution!")

# Mathemattical operation on tensors (see tf.math)
# A primes vector containing prime numbers.
primes = tf.constant([2, 3, 5, 7, 11, 13], dtype=tf.int32)
print("primes:", primes)

# A ones vector containing all 1 values.
ones = tf.ones([6], dtype=tf.int32)
print("ones:", ones)

# A vector created by performing element-wise addition over the first two vectors.
just_beyond_primes = tf.add(primes, ones)
print("just_beyond_primes:", just_beyond_primes)

# A vector created by doubling the elements in the primes vector.
twos = tf.constant([2, 2, 2, 2, 2, 2], dtype=tf.int32)
primes_doubled = primes * twos
print("primes_doubled:", primes_doubled)

# calling the value of the tensor as a numpy array
primes_value = primes.numpy()
print(primes_value)

# A zero scalar (0-D tensor).
scalar = tf.zeros([])

# A zero vector with 3 elements.
vector = tf.zeros([3])

# A zero matrix with 2 rows and 3 columns.
matrix = tf.zeros([2, 3])

# calling the shape of the tensor as a numpy array
matrix_dim = matrix.get_shape()
print(matrix_dim)

# TensorFlow supports broadcasting (a concept borrowed from numpy)

one = tf.constant(1, dtype=tf.int32)
print("one:", one)

# adds one to primes using broadcasting
just_beyond_primes = tf.add(primes, one)
print("just_beyond_primes:", just_beyond_primes)

# primes times two using broadcasting
two = tf.constant(2, dtype=tf.int32)
primes_doubled = primes * two
print("primes_doubled:", primes_doubled)

# create a vector such that the ith element equals square of the ith element of primes minus 1
def solution(primes):
    primes_square = tf.pow(primes, 2)
    just_under_primes_squared = tf.subtract(primes_square,one)
    return just_under_primes_squared

just_under_primes_squared = solution(primes)
print("just_under_primes_squared:", just_under_primes_squared)

# matrix multiplication
# A 3x4 matrix (2-d tensor).
x = tf.constant([[5, 2, 4, 3], [5, 1, 6, -2], [-1, 3, -1, -2]],
                dtype=tf.int32)

# A 4x2 matrix (2-d tensor).
y = tf.constant([[2, 2], [3, 5], [4, 5], [1, 6]], dtype=tf.int32)

# Multiply `x` by `y`; result is 3x2 matrix.
matrix_multiply_result = tf.matmul(x, y)
print(matrix_multiply_result)

# Reshape tensors
# Create an 8x2 matrix (2-D tensor).
matrix = tf.constant(
    [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]],
    dtype=tf.int32)

reshaped_2x8_matrix = tf.reshape(matrix, [2, 8])
reshaped_4x4_matrix = tf.reshape(matrix, [4, 4])

print("Original matrix (8x2):")
print(matrix.numpy())
print("Reshaped matrix (2x8):")
print(reshaped_2x8_matrix.numpy())
print("Reshaped matrix (4x4):")
print(reshaped_4x4_matrix.numpy())

reshaped_2x2x4_tensor = tf.reshape(matrix, [2, 2, 4])
one_dimensional_vector = tf.reshape(matrix, [16])

print("Original matrix (8x2):")
print(matrix.numpy())
print("Reshaped 3-D tensor (2x2x4):")
print(reshaped_2x2x4_tensor.numpy())
print("1-D vector:")
print(one_dimensional_vector.numpy())

# Reshape  vectors into compatible operands for matrix multiplication
a = tf.constant([5, 3, 2, 7, 1, 4])
b = tf.constant([4, 6, 3])
a_reshape = tf.reshape(a, [2, 3])
b_reshape = tf.reshape(b, [3, 1])
axb = tf.matmul(a_reshape, b_reshape)
print(axb)

# Variables
# Create a scalar variable with the initial value 3.
v = tf.contrib.eager.Variable([3])

# Create a vector variable of shape [1, 4], with random initial values,
# sampled from a normal distribution with mean 1 and standard deviation 0.35.
w = tf.contrib.eager.Variable(tf.random_normal([1, 4], mean=1.0, stddev=0.35))

print("v:", v.numpy())
print("w:", w.numpy())

tf.assign(v, [7])
print(v.numpy())

v.assign([5])
print(v.numpy())

v = tf.contrib.eager.Variable([[1, 2, 3], [4, 5, 6]])
print(v.numpy())

try:
  print("Assigning [7, 8, 9] to v")
  v.assign([7, 8, 9])
except ValueError as e:
  print("Exception:", e)

# Simulate 10 rolls of two dice. Use tf.random.uniform to generate random int in [1,2,3,4,5,6]
dice = tf.random.uniform([10,2], minval = 1, maxval = 7, dtype = tf.int32)

# tf.reduce_sum computes the sum of elements across dimensions
sum = tf.reduce_sum(dice, 1)
sum = tf.reshape(sum, [10,1])

# tf.concat Concatenates tensors along one dimension
ans = tf.concat([dice, sum], 1)
