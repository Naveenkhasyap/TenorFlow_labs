import tensorflow as tf 

sess = tf.InteractiveSession()

A = tf.constant([4],int32,name="A")
x = tf.placeholder(tf.int32,name="x")

y = A * x

#y.eval = tf.get_default_session().run()
y.eval(feed_dict = {x:[5]})

print(y.eval(feed_dict = {x:[5]}))

sess.close()