import tensorflow as tf 
import matplotlib.image as mp_img
import matplotlib.pyplot as plot 
import os 


filename = "./one_dandelion.JPG"

image = mp_img.imread(filename)


print("Image shape : ",image.shape) #shape of the image [length,width,RGB]
print("Image array : ",image) #pixel array values

plot.imshow(image)  # to load the image 
plot.show()     # to display image to screen

x = tf.Variable(image,name="x")

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    #Method 1 Generic matrix transposing function perm parametes tell x,y,z 
    #which axis to transform
    transpose = tf.transpose(x,perm=[1,0,2])
    
    #Method 2 inbuilt transpose of image in tensorflow
    #transpose = tf.image.transpose_image(x)
    
    result = sess.run(transpose)

    print("transposed image shape:", result)
    plot.imshow(result)
    plot.show()