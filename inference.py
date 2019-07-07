"""Translate an image to another image
An example of command-line usage is:
python export_graph.py --model pretrained/apple2orange.pb \
                       --input input_sample.jpg \
                       --output output_sample.jpg \
                       --image_size 256
"""

import tensorflow as tf
import os
import utils
from time import time


def inference():
    #path_src_test_pic = r"./Train_input"
    #path_dst_test_dir = r"./Train_output"
    path_src_test_pic = r"./Test_input"
    path_dst_test_dir = r"./Test_output"
    temp_img_list = os.listdir(path_src_test_pic)
    image_size = 400
    path_model_dir = r"./pretrained"
    model_name_list = os.listdir(path_model_dir)
    Total_count_model = len(model_name_list)
    Total_count_img = len(temp_img_list)

    count_model = 1
    #################################################################################
    for model_name in model_name_list:

        path_model_now_use = os.path.join(path_model_dir, model_name)
        print("Model Progress :" + str(count_model) + "/" + str(Total_count_model))
        print("Now use Model is " + model_name)
        count_model += 1
        count_img = 1
        model_name = model_name.split(".")
        model_name = model_name[0]
        output_by_model_dir = os.path.join(path_dst_test_dir, model_name)

        if not os.path.isdir(output_by_model_dir):
            os.mkdir(output_by_model_dir)

        print("Image will save in the path of directory :  "+output_by_model_dir)


        for img in temp_img_list:
            graph = tf.Graph()
            start=time()


            with graph.as_default():
                temp_input = os.path.join(path_src_test_pic, img)
                temp_output = os.path.join(output_by_model_dir, img)


                with tf.gfile.FastGFile(temp_input, 'rb') as f:
                    image_data = f.read()
                    input_image = tf.image.decode_jpeg(image_data, channels=3)
                    input_image = tf.image.resize_images(input_image, size=(image_size, image_size))
                    input_image = utils.convert2float(input_image)
                    input_image.set_shape([image_size, image_size, 3])

                with tf.gfile.FastGFile(path_model_now_use, 'rb') as model_file:
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(model_file.read())



                    [output_image] = tf.import_graph_def(graph_def,
                                                         input_map={'input_image': input_image},
                                                         return_elements=['output_image:0'],
                                                         name='output')

                with tf.Session(graph=graph) as sess:
                    generated = output_image.eval()
                    with open(temp_output, 'wb') as f:
                        f.write(generated)

                print(str(count_img) + "/" + str(Total_count_img))

                End=time()
                t=End-start
                print(str(t)+" sec")
                count_img += 1








def main(unused_argv):
  inference()

if __name__ == '__main__':
  tf.app.run()
