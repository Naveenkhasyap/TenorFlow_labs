import tensorflow as tf 

x = tf.placeholder(tf.int32,shape=[3],name ="x")
y = tf.placeholder(tf.int32,shape=[3],name ="y")

sum_x = tf.reduce_sum(x,name ="sum_x")
prod_y = tf.reduce_prod(y,name="prod_y")

final_div = tf.div(sum_x,prod_y,name="final_div")
final_mean = tf.reduce_mean([sum_x,prod_y],name="final_mean") 

sess = tf.Session()



print("sum(x): ",sess.run(sum_x,feed_dict={x:[100,200,300]}))
print("prod_y: ",sess.run(prod_y,feed_dict={y:[1,2,3]}))
print("sum_x/prod_y: ",sess.run(final_div,feed_dict={x:[1,2,3] , y:[3,4,5]}))
print("mean: ",sess.run(final_mean,feed_dict={x:[1000,2000,3000] , y:[3,4,5]}))

writer = tf.summary.FileWriter('./three_board',sess.graph)
writer.close()
sess.close() 

 
 