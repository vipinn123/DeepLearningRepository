import numpy as np
import tensorflow as tf
import scipy.misc
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *




#Read files
def read_content_and_style_inputs(content_path, stype_path):

    content_image = scipy.misc.imread(content_path)
    content_image = reshape_and_normalize_image(content_image) #TODO - To check the mean
    print(" CONTENT IMAGE SHAPE : "+str(content_image.shape))
    #imshow(content_image)


    style_image = scipy.misc.imread(stype_path)
    style_image = reshape_and_normalize_image(style_image) #TODO - To check the mean
    print("STYLE IMAGE SHAPE : "+ str(style_image.shape))
    #imshow(style_image)

    return content_image,style_image


def compute_content_loss(a_C,a_G):

    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape a_C and a_G (≈2 lines)
    a_C_unrolled = tf.reshape(a_C,[n_H*n_W,n_C])
    a_G_unrolled = tf.reshape(a_G,[n_H*n_W,n_C])
    
    # compute the cost with tensorflow (≈1 line)
    return tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled,a_G_unrolled)))/(4*n_H*n_W*n_C)
    

def compute_layer_style_cost(a_S, a_G):

    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape the images to have them of shape (n_C, n_H*n_W) (≈2 lines)
    a_S = tf.transpose(tf.reshape(a_S,[n_H*n_W,n_C]))
    a_G = tf.transpose(tf.reshape(a_G,[n_H*n_W,n_C]))


    GS = tf.matmul(a_S,tf.transpose(a_S))
    GG = tf.matmul(a_G,tf.transpose(a_G))
    
    return  tf.reduce_sum(tf.square(tf.subtract(GS,GG)))/(4*(n_H*n_W)*(n_H*n_W)*n_C*n_C)

def compute_style_loss(STYLE_LAYERS_ACTIVAIONS,model):

    STYLE_LAYERS_COEFICIENT = {
        'conv1_1': 0.2,
        'conv2_1' :0.2,
        'conv1_1': 0.2,
        'conv2_1' :0.2,
        'conv2_1' :0.2
    }
   
    J_style = 0

    #input_image_activation = model[content_layer]

    for layer_name in STYLE_LAYERS_ACTIVAIONS:
        #input_image_activation = model[layer_name]
        J_style_Layer=compute_layer_style_cost(STYLE_LAYERS_ACTIVAIONS[layer_name], model[layer_name] )
        J_style+=STYLE_LAYERS_COEFICIENT[layer_name]*J_style_Layer


    return J_style


def get_style_activations(model):

    STYLE_LAYERS_ACTIVAIONS = {
        'conv1_1': None,
        'conv2_1' :None,
        'conv1_1': None,
        'conv2_1' :None,
        'conv2_1' :None
    }

    """STYLE_LAYERS_ACTIVAIONS = [
    ('conv1_1', None ),
    ('conv2_1', None),
    ('conv3_1', None),
    ('conv4_1', None),
    ('conv5_1', None)]"""


    for layer_name in STYLE_LAYERS_ACTIVAIONS:
        STYLE_LAYERS_ACTIVAIONS[layer_name] = model[layer_name]

    return STYLE_LAYERS_ACTIVAIONS



def initialize_activations(CONTENT_PATH,STYLE_PATH,model,CONTENT_LAYER):

    content_image, style_image = read_content_and_style_inputs(CONTENT_PATH,STYLE_PATH)

    content_image_vgg = model['input'].assign(content_image)
    content_image_activations = model['conv4_2']

    style_image_vgg=model['input'].assign(style_image)
    style_activations=get_style_activations(model)

    return content_image_activations, style_activations

def compute_total_cost(content_activations,style_activations,model,CONTENT_LAYER,content_weight=0.5,style_weight=0.5):

    J=0

    J_content = compute_content_loss(content_activations,model[CONTENT_LAYER])

    J_style = compute_style_loss(style_activations,model)

    J = content_weight*J_content+style_weight*J_style

    return J,J_content,J_style

def main(argv):

    CONTENT_PATH="my_content.jpg"
    STYLE_PATH="my_style.jpg"
    VGG_MODEL_PATH="vgg19/imagenet-vgg-verydeep-19"
    CONTENT_LAYER='conv4_2'
    num_iterations = 200

    model = load_vgg_model(VGG_MODEL_PATH)

    content_activations, style_activations = initialize_activations(CONTENT_PATH,STYLE_PATH,model,CONTENT_LAYER)

    # TODO - The output size must sync with content image. Also why not initialize noise to average of content
    input_image = generate_noise_image(reshape_and_normalize_image(scipy.misc.imread(CONTENT_PATH)))

    model['input'].assign(input_image)

    J = compute_total_cost(content_activations,style_activations,model,CONTENT_LAYER)

    optimizer = tf.train.AdamOptimizer(2.0)

    

    for i in range(num_iterations):
        train_step = optimizer.minimize(J)
        generated_image = model['input']

                # Print every 20 iteration.
        if i%20 == 0:
            J, Jc, Js = compute_total_cost(content_activations,style_activations,model,CONTENT_LAYER)
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(J))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
            
            # save current generated image in the "/output" directory
            save_image("output/" + str(i) + ".png", generated_image)
    
    # save last generated image
    save_image('output/generated_image.jpg', generated_image)

    #print(out)

    #print(model)

    return





if __name__ == "__main__":
    
    tf.enable_eager_execution()
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)


    pass