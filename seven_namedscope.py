import tensorflow as tf 

a = tf.constant([6],tf.int32,name="constant_a")
b = tf.constant([5],tf.int32,name="constant_b")
c = tf.constant([3],tf.int32,name="constant_c")

x = tf.placeholder(tf.int32,name="x")

# y = Ax^2 + Bx + c
with tf.name_scope("Equation_1"):
    Ax2 = tf.multiply(a,tf.pow(x,2),name="Ax2")
    Bx = tf.multiply(b,x,name="Bx")
    y1 = tf.add_n([Ax2,Bx,c],name="y1")

# y = Ax^2 + Bx^2
with tf.name_scope("Equation_2"):
    Ax2 = tf.multiply(a,tf.pow(x,2),name="Ax2")
    Bx2 = tf.multiply(b,tf.pow(x,2),name="Bx2")
    y2 = tf.add_n([Ax2,Bx2],name="y2")

with tf.name_scope("Final_sum"):
    y = y1+y2
    

with tf.Session() as sess:
    print(sess.run(y,feed_dict={x:[10]}))

    writer = tf.summary.FileWriter('./seven_board',sess.graph)
    writer.close()  
