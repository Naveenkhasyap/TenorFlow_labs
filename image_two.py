import tensorflow as tf 
from PIL import Image

original_image_list = ['./one_flower.jpg',
                        './two_flower.jpg',
                        './three_flower.jpg',
                        './four_flower.jpg']

#make a queue of file names including all the 
#images specified above
filename_queue = tf.train.string_input_producer(original_image_list)

#Read an entire image file
image_reader = tf.WholeFileReader()

with tf.Session() as sess:
    #Coordinate the loading of image files 
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    image_list = []
    for i in range(len(original_image_list)):
        #_ is file name and image_file is actual file
        _, image_file = image_reader.read(filename_queue)

        image = tf.image.decode_jpeg(image_file)

        image = tf.image.resize_images(image,[224,224])
        image.set_shape([224,224,3])

        #get an image tensor and print it
        image_array = sess.run(image)
        print(image_array.shape)

        Image.fromarray(image_array.astype('uint8'),'RGB').show()

        #expand the dimension of the image 
        image_list.append(tf.expand_dims(image_array,0))
    
    coord.request_stop()
    coord.join(threads)

    index = 0
    summary_writer = tf.summary.FileWriter('./image_two',graph=sess.graph)
    
    for image_tensor in image_list:
        summary_str = sess.run(tf.summary.image("image-"+str(index),image_tensor))
        summary_writer.add_summary(summary_str)
        index+=1

    summary_writer.close()