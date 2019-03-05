import tensorflow as tf

# Tensorflow concepts
# Tensors: constant or variable arrays of arbitrary dimensionality (scalar, vector, matrix ...)
# Operations: Programs that manipulate tensors
# Graphs: A graph data structure constructed by operation (notdes) and tensors (edges)

# define a constant tensor
x = tf.constant(3.14)

# define a variable tensor 
y = tf.Variable([5])

# assign a value to varibles 
y = y.assign([1])

# Graphs must run within a TensorFlow session, which holds the state for the graph(s) it runs. And variables must be initialized by calling tf.global_variables_initializer at the start of session.

# Create a graph.
g = tf.Graph()

# Establish the graph as the "default" graph.
with g.as_default():
  # Assemble a graph consisting of the following three operations:
  #   * Two tf.constant operations to create the operands.
  #   * One tf.add operation to add the two operands.
  x = tf.constant(8, name="x_const")
  y = tf.constant(5, name="y_const")
  my_sum = tf.add(x, y, name="x_y_sum")

  z = tf.constant(4)
  new_sum = tf.add(my_sum, z)

  # Now create a session.
  # The session will run the default graph.
  with tf.Session() as sess:
    print(my_sum.eval()) # t.eval() is a shortcut for calling tf.get_default_session().run(t)
    print(new_sum.eval()) 

