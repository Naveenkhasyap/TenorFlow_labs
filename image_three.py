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

        #to flip an image upside down
        image = tf.image.flip_up_down(image)

        #to crop an image at center 
        #image = tf.image.central_crop(image,central_fraction=0.5)
        #get an image tensor and print it
        image_array = sess.run(image)
        print(image_array.shape)

        image_tensor = tf.stack(image_array)

        #expand the dimension of the image 
        print(image_tensor)
        image_list.append(image_tensor)
    
    coord.request_stop()
    coord.join(threads)


    image_tensor  = tf.stack(image_list)
    print(image_tensor)
    # index = 0
    summary_writer = tf.summary.FileWriter('./image_three',graph=sess.graph)
    
    # for image_tensor in image_list:
    summary_str = sess.run(tf.summary.image("image",image_tensor,max_outputs=4))
    summary_writer.add_summary(summary_str)
    #     index+=1

    summary_writer.close()