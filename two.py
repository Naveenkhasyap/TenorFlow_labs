import tensorflow as tf 

x = tf.constant([100,200,300],name ="x")
y = tf.constant([1,2,3],name ="y")

sum_x = tf.reduce_sum(x,name ="sum_x")
prod_y = tf.reduce_prod(y,name="prod_y")

final_div = tf.div(sum_x,prod_y,name="final_div")
final_mean = tf.reduce_mean([sum_x,prod_y],name="final_mean") 

sess = tf.Session()

print("x: ",sess.run(x))
print("y: ",sess.run(y))

print("sum(x): ",sess.run(sum_x))
print("prod_y: ",sess.run(prod_y))
print("sum_x/prod_y: ",sess.run(final_div))
print("mean: ",sess.run(final_mean))

writer = tf.summary.FileWriter('./two_board',sess.graph)
writer.close()
sess.close() 

 
 