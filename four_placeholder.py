import tensorflow as tf 

#y  = Wx + b 
W= tf.constant([10,100],name='const_W')

#these placeholders can hold tensors of any shape
x = tf.placeholder(tf.int32,name ="x")
b = tf.placeholder(tf.int32,name ="b")

Wx = tf.multiply(W,x,name="Wx")

y = tf.add(Wx,b,name="y")


#y_ = x - b
y_ = tf.subtract(x,b,name="y_")


with tf.Session() as sess:
    
    print("Intermediate result Wx : ",sess.run(Wx,feed_dict={x:[3,33]}))
    print("Final result Wx + b: ",sess.run(y,feed_dict={Wx:[5,50],b:[1,2]}))
    print("Intermediate specified Wx + b = ",\
    sess.run(fetches=y, feed_dict={Wx:[10,20] , b:[3,4]}))
    print("two results : [Wx+b , x-b] =  ",\
    sess.run(fetches=[y, y_], feed_dict={x:[5,50] , b:[1,2]}))


writer = tf.summary.FileWriter('./four_board',sess.graph)
writer.close()

 
 