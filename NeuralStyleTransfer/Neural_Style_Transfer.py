import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
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

    a_C = a_C[0]
    a_G = a_G[0]
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape a_C and a_G (≈2 lines)
    a_C_unrolled = tf.reshape(a_C,[n_H*n_W,n_C])
    a_G_unrolled = tf.reshape(a_G,[n_H*n_W,n_C])
    
    # compute the cost with tensorflow (≈1 line)

    J_content = tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled,a_G_unrolled)))/(4*n_H*n_W*n_C)

    #print("########J Content")
    #print(a_C)
    #print(a_G[0][0][0])
    #print(J_content)
    return J_content
    

def compute_layer_style_cost(a_S, a_G):

    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape the images to have them of shape (n_C, n_H*n_W) (≈2 lines)
    a_S = tf.transpose(tf.reshape(a_S,[n_H*n_W,n_C]))
    a_G = tf.transpose(tf.reshape(a_G,[n_H*n_W,n_C]))


    GS = tf.matmul(a_S,tf.transpose(a_S))
    GG = tf.matmul(a_G,tf.transpose(a_G))
    
    return  tf.reduce_sum(tf.square(tf.subtract(GS,GG)))/(4*(n_H*n_W)*(n_H*n_W)*n_C*n_C)

def compute_style_loss(style_activations,input_image_activations):


    STYLE_LAYERS_COEFICIENT = [0.2,0.2,0.2,0.2,0.2]

    style_loss_tuple_set = set(zip(style_activations,input_image_activations,STYLE_LAYERS_COEFICIENT))


   
    J_style = 0

    #input_image_activation = model[content_layer]

    for style_layer,input_layer,coef in style_loss_tuple_set:
        #input_image_activation = model[layer_name]
        J_style_Layer=compute_layer_style_cost(style_layer, input_layer)
        J_style+=coef*J_style_Layer


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


    """for layer_name in STYLE_LAYERS_ACTIVAIONS:
        STYLE_LAYERS_ACTIVAIONS[layer_name] = tf.identity(model[layer_name])"""

    return STYLE_LAYERS_ACTIVAIONS

    



def initialize_activations(CONTENT_PATH,STYLE_PATH,model,STYLE_LAYERS,CONTENT_LAYER):

    content_image, style_image = read_content_and_style_inputs(CONTENT_PATH,STYLE_PATH)

    """content_image_vgg = model['input'].assign(content_image)
    content_image_activations = tf.identity(model['conv4_2'])"""

    print(content_image.dtype)
    content_image_vgg=model(tf.cast(content_image,tf.float32))

    content_image_activations = content_image_vgg[len(STYLE_LAYERS):]



    #print(content_image_activations[0][0][0])

    """style_image_vgg=model['input'].assign(style_image)
    style_activations=get_style_activations(model)"""

    style_image_vgg=model(tf.cast(style_image,tf.float32))
    #style_activations=get_style_activations(model)
    style_activations = style_image_vgg[:len(STYLE_LAYERS)]



    return content_image_activations, style_activations

def compute_total_cost(content_activations,style_activations,input_image_vgg,STYLE_LAYERS,CONTENT_LAYER,content_weight=0.5,style_weight=0.5):

    J=0

    #print("####### Check Content Activiation and Model ")
    J_content = compute_content_loss(content_activations,input_image_vgg[len(STYLE_LAYERS):])

    J_style = compute_style_loss(style_activations,input_image_vgg[:len(STYLE_LAYERS)])

    J = content_weight*J_content+style_weight*J_style

    return J,J_content,J_style


def main(argv):

    CONTENT_PATH="my_content.jpg"
    STYLE_PATH="my_style.jpg"
    VGG_MODEL_PATH="imagenet-vgg-verydeep-19"
    CONTENT_LAYER='block5_conv2'
    STYLE_LAYERS = ['block1_conv1','block2_conv1','block3_conv1', 'block4_conv1', 'block5_conv1']
    num_iterations = 5000

    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means 

    #model = load_vgg_model(VGG_MODEL_PATH)
    model = get_vgg_model(STYLE_LAYERS,CONTENT_LAYER)

    content_activations, style_activations = initialize_activations(CONTENT_PATH,STYLE_PATH,model,STYLE_LAYERS,CONTENT_LAYER)

    
    # TODO - The output size must sync with content image. Also why not initialize noise to average of content
    input_image = generate_noise_image(reshape_and_normalize_image(scipy.misc.imread(CONTENT_PATH)),0.6)
    input_image = tfe.Variable(input_image,dtype=tf.float32)
    
    
    
    optimizer = tf.train.AdamOptimizer(2.0)
    
 
    for i in range(num_iterations):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(input_image)
            
            input_image_vgg = model(tf.cast(input_image,tf.float32))
            
            J,J_content,J_style = compute_total_cost(content_activations,style_activations,input_image_vgg,STYLE_LAYERS, CONTENT_LAYER,0.1,0.9)
            
            grads = tape.gradient(J, input_image)

            clipped = tf.clip_by_value(input_image, min_vals, max_vals)

            input_image.assign(clipped)

            assert grads != None

            #print("###Loss J ######" + str(J))
            #print("###Loss J ######" + str(J_style))
            #print("###Loss J ######" + str(J_content))
            
            optimizer.apply_gradients([(grads,input_image)])

            
            # Print every 20 iteration.
            if i%20 == 0:
                #J, Jc, Js = compute_total_cost(content_activations,style_activations,model,CONTENT_LAYER)
                print("Iteration " + str(i) + " :")
                print("total cost = " + str(J))
                print("content cost = " + str(J_content))
                print("style cost = " + str(J_style))
                
                # save current generated image in the "/output" directory
                save_image("output/" + str(i) + ".png", input_image)

    
    # save last generated image
    #generated_image = deprocess_img(input_image.numpy())
    save_image('output/generated_image.jpg', input_image)

    #print(out)

    #print(model)
    
    return





if __name__ == "__main__":
    
    tf.enable_eager_execution()
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)


    pass