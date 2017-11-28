import tensorflow as tf 

g1 = tf.Graph()

with g1.as_default():
    with tf.Session() as sess:
        a = tf.constant([6,5],tf.int32,name="constant_a")
        b = tf.placeholder(tf.int32,name="constant_b")
        c = tf.constant([3,2],tf.int32,name="constant_c")
       
        y = a * b + c

        print(sess.run(y,feed_dict={b:[10,100]}))

        assert y.graph is g1

g2 = tf.Graph()

with g2.as_default():
    with tf.Session() as sess:
        a = tf.constant([1,3],tf.int32,name="constant_a")
        b = tf.placeholder(tf.int32,name="constant_b")
        c = tf.constant([2,8],tf.int32,name="constant_c")
       
        y = a*b+c

        print(sess.run(y,feed_dict={b:[13,11]}))

        assert y.graph is g2

default_graph = tf.get_default_graph()
with tf.Session() as sess:
        a = tf.constant([1,3],tf.int32,name="constant_a")
        b = tf.placeholder(tf.int32,name="constant_b")
        #c = tf.constant([2,8],tf.int32,name="constant_c")
       
        y = a+b

        print(sess.run(y,feed_dict={b:[13,11]}))

        assert y.graph is default_graph