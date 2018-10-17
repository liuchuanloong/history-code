import os,time,cv2, sys, math
import tensorflow as tf
import argparse
import numpy as np

from utils import utils, helpers
from builders import model_builder

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default=None, required=False, help='The image you want to predict on. ')
parser.add_argument('--checkpoint_path', type=str, default=None, required=True, help='The path to the latest checkpoint weights for your model.')
parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped input image to network')
parser.add_argument('--model', type=str, default="RefineNet", required=False, help='The model you are using')
parser.add_argument('--dataset', type=str, default="CamVid", required=False, help='The dataset you are using')
parser.add_argument('--frontend', type=str, default="ResNet50", help='The frontend you are using. See frontend_builder.py for supported models')
args = parser.parse_args()

class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))
pre_input_names = utils.prepare_predict_data(args.dataset)
pre_indices=range(0,len(pre_input_names))

num_classes = len(label_values)

print("\n***** Begin prediction *****")
print("Dataset -->", args.dataset)
print("Model -->", args.model)
print("Crop Height -->", args.crop_height)
print("Crop Width -->", args.crop_width)
print("Num Classes -->", num_classes)
print("Image -->", args.image)

# Initializing network
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)

net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes]) 

network, _ = model_builder.build_model(model_name=args.model, frontend=args.frontend, net_input=net_input, num_classes=num_classes, crop_width=args.crop_width + 27, crop_height=args.crop_height + 27, is_training=False)

sess.run(tf.global_variables_initializer())

print('Loading model checkpoint weights')
saver=tf.train.Saver(max_to_keep=1000)
saver.restore(sess, args.checkpoint_path)

for ind in pre_indices:
    print("Testing image " + pre_input_names[ind])

    loaded_image = utils.load_image(pre_input_names[ind])
    padding_image = cv2.copyMakeBorder(loaded_image, 14, 13, 14, 13, cv2.BORDER_REFLECT)
    input_image = np.expand_dims(np.float32(padding_image),axis=0)/255.0

    st = time.time()
    output_image = sess.run(network,feed_dict={net_input:input_image})

    run_time = time.time()-st

    output_image = np.array(output_image[0,:,:,:])
    output_image = helpers.reverse_one_hot(output_image)

    out_vis_image = helpers.colour_code_segmentation(output_image, label_values)[14:-13,14:-13,:]
    file_name = utils.filepath_to_name(pre_input_names[ind])
    if not os.path.isdir("%s/predict/%s"%(args.dataset,args.model)):
        os.makedirs("%s/predict/%s"%(args.dataset,args.model))
    cv2.imwrite("%s/predict/%s/%s_pred.png"%(args.dataset,args.model,file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))

    print("Wrote image " + "%s/predict/%s/%s_pred.png"%(args.dataset,args.model,file_name))

print("")
print("Finished!")