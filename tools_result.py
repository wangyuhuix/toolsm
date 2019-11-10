import tensorflow as tf

sess = tf.InteractiveSession()
a = tf.constant(1.)
tf.global_variables_initializer().run()