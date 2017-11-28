import tensorflow as tf 

#y  = Wx + b 
W= tf.Variable([2.5,4.0],tf.float32,name='var_W')

#these placeholders can hold tensors of any shape
x = tf.placeholder(tf.float32,name ="x")
b = tf.Variable([5.0,10.0],tf.float32,name ="var_b")

y = W * x + b
#initialize all defined variable
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print("Final result Wx+b = ",sess.run(y,feed_dict = {x:[10,100]}))

#s = Wx
s= W*x 

#initialize only requrired variables
init = tf.variables_initializer([W])

with tf.Session() as sess:
    sess.run(init)
    #this does not work as we are using unitialized variable b
    #print("will thi work ? Wx+b = ",sess.run(y,feed_dict{x:[10,100]}))
    print("Results Wx = ",sess.run(s,feed_dict = {x:[10,100]}))

number = tf.Variable(2)
multiplier = tf.Variable(1)

init = tf.global_variables_initializer()

result = number.assign(tf.multiply(number,multiplier))

with tf.Session() as sess:
    sess.run(init)

    for i in range(10):
        print("final result ",sess.run(result))
        print("Increment multiplier ",sess.run(multiplier.assign_add(1)))

writer = tf.summary.FileWriter('./five_board',sess.graph)
writer.close()