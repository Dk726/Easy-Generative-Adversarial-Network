#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import all dependencies
from matplotlib import pyplot as plt
import cv2
import numpy as np
import glob
import random
import pandas as pd
import albumentations as A
from sklearn.model_selection import train_test_split
from skimage.metrics import structural_similarity as ssi
import tensorflow as tf
from tensorflow.keras import mixed_precision
import keras
from numpy.random import randn
from numpy.random import randint
from keras.optimizers import Adam, RMSprop
from keras.models import Model
from keras.layers import Input, Conv2DTranspose, UpSampling2D, LayerNormalization, Add, Layer, LeakyReLU, BatchNormalization, MaxPooling2D
from keras.layers import Dense, Reshape, Flatten, Conv2D, Dropout, Concatenate, add, MaxPool2D, SpectralNormalization

#Random fix
from numpy.random import seed
seed(19)
import tensorflow as tf 
tf.random.set_seed(19)

##############################################################################################################################################
# Read Image dataset 
def process_images(path): #This function assumes 1024x640 image size and outputs resized-cropped images of size 128x160
    images = []
    for img_path in sorted(glob.glob(path)):
        img = cv2.imread(img_path, 0)
        img = cv2.resize(img, (0, 0), fx = 0.5, fy = 0.5)
        for i in range(0, 320, 160):  # height
            for j in range(0, 512, 128):  # width
                patch = img[i:i+160, j:j+128]
                if patch.shape == (160, 128):  # Ensure correct size
                    images.append((patch - 127.5) / 127.5)
    return images

# Define paths and process
path = "Original Images/*.*" #Please insert the image path. The given path is examplenary
imgs = process_images(path)
real_img = np.array(imgs)

##############################################################################################################################################
# Define all functions
def load_real_samples(real_img, half_batch):
    select = randint(0, np.shape(real_img)[0], half_batch)
    X_real = real_img[select]
    return X_real

def generate_latent_points(W, H, n_samples, M=0, SD=10): #Generates random noise Z
    z_input = []
    for i in range(0, n_samples):
        x_input = np.random.normal(M, SD, size=(W,H))
        z_input.append(x_input)
    z_input = np.array(z_input)
    return z_input

#Load trained Super-Noise-Generator model as GSN_model
GSN_model = keras.saving.load_model('')
def generate_mZs(GSN_model, W, H, n_samples, M=0, SD=10): #Generates synthetic structured noise meZ
    z_input = []
    for i in range(0, n_samples):
        x_input = np.random.normal(M, SD, size=(W,H))
        z_input.append(x_input)
    z_input = np.array(z_input)
    z_input = GSN_model.predict(z_input)
    return z_input #meZ

def generate_mI(GSN_model, g_model, W, H, n_samples): #Generates synthetic images mI
    z_input = generate_meZ(eZ_model, W, H, n_samples)
    images = g_model.predict(z_input)
    return images #mI

def discriminator_loss(real_output, fake_output):
    real_loss = tf.reduce_mean(real_output)
    fake_loss = tf.reduce_mean(fake_output)
    total_loss = -real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    fake_loss = tf.reduce_mean(fake_output)
    return -fake_loss

#For Perceptual loss
def feature_matching_loss(real_features, fake_features):
    return tf.reduce_mean(tf.abs(real_features - fake_features))

def gradient_penalty(d_model, real_images, fake_images):
    batch_size = tf.shape(real_images)[0]

    if len(real_images.shape) == 3:
        real_images = tf.expand_dims(real_images, -1)
    if len(fake_images.shape) == 3:
        fake_images = tf.expand_dims(fake_images, -1)

    real_images = tf.cast(real_images, tf.float32)
    fake_images = tf.cast(fake_images, tf.float32)

    alpha = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0., maxval=1.)
    alpha = tf.cast(alpha, real_images.dtype) 

    interpolated = real_images + alpha * (fake_images - real_images)

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = d_model(interpolated, training=True)

    grads = tape.gradient(pred, [interpolated])[0]
    grads_norm = tf.norm(tf.reshape(grads, [batch_size, -1]), axis=1)
    gp = tf.reduce_mean((grads_norm - 1.0) ** 2)
    return gp
    
##############################################################################################################################################
#Build SL-Generator and SL-Descriminator. Varient; SLGAN-Dense-Wloss. **For Patch varient, uncomment patch output in Descriminator and comment FCL block**
def Generator(W, H, n_classes=3): #eGenerator
    in_lat = Input(shape=(W, H, 1)) 

    C31 = Conv2D(128, (5,5), padding='same', kernel_initializer='he_normal')(in_lat)
    C3 = LeakyReLU(alpha=0.3)(C31)
    C3 = Conv2D(128, (5,5), padding='same', kernel_initializer='he_normal')(C3)
    C3 = Add()([C31, C3])
    C3 = LeakyReLU(alpha=0.3)(C3)
    
    C3 = UpSampling2D(size=(2,2), interpolation="nearest")(C3)
    C3 = Conv2D(128, (5,5), padding='same', kernel_initializer='he_normal')(C3)
    C3 = LeakyReLU(alpha=0.3)(C3)
    
    C31 = Conv2D(128, (5,5), padding='same', kernel_initializer='he_normal')(C3)
    C3 = LeakyReLU(alpha=0.3)(C31)
    C3 = Conv2D(128, (5,5), padding='same', kernel_initializer='he_normal')(C3)
    C3 = Add()([C31, C3])
    C3 = LeakyReLU(alpha=0.3)(C3)

    C3 = UpSampling2D(size=(2,2), interpolation="nearest")(C3)
    C3 = Conv2D(256, (5,5), padding='same', kernel_initializer='he_normal')(C3)
    C3 = LeakyReLU(alpha=0.3)(C3)
    
    C31 = Conv2D(256, (5,5), padding='same', kernel_initializer='he_normal')(C3)
    C3 = LeakyReLU(alpha=0.3)(C31)
    C3 = Conv2D(256, (5,5), padding='same', kernel_initializer='he_normal')(C3)
    C3 = Add()([C31,C3])
    C3 = LeakyReLU(alpha=0.2)(C3)

    C31 = Conv2D(512, (5,5), padding='same', kernel_initializer='he_normal')(C3)
    C3 = LeakyReLU(alpha=0.3)(C31)
    C3 = Conv2D(512, (5,5), padding='same', kernel_initializer='he_normal')(C3)
    C3 = Add()([C31,C3])
    C3 = LeakyReLU(alpha=0.2)(C3)
    
    out_layer = Conv2D(1, (5,5), activation='tanh', padding='same')(C3) #320x256x1

    model = Model(in_lat, out_layer)
    return model 

def Discriminator(input_shape): #eDescriminator
    inp = Input(shape=input_shape)
    
    X = SpectralNormalization(Conv2D(64, 4, padding='same'))(inp)
    x = LeakyReLU(0.2)(X)
    x = SpectralNormalization(Conv2D(64, 4, padding='same'))(x)
    x = Add()([X,x])
    x = LeakyReLU(0.2)(x)
    x = SpectralNormalization(Conv2D(64, 4, strides=2, padding='same'))(x)
    
    X = SpectralNormalization(Conv2D(128, 4, padding='same'))(x)
    x = LeakyReLU(0.2)(X)
    x = SpectralNormalization(Conv2D(128, 4, padding='same'))(x)
    x = Add()([X,x])
    x = LeakyReLU(0.2)(x)
    x = SpectralNormalization(Conv2D(128, 4, strides=2, padding='same'))(x)
    
    X = SpectralNormalization(Conv2D(256, 4, padding='same'))(x) 
    x = LeakyReLU(0.2)(X)
    x = SpectralNormalization(Conv2D(256, 4, padding='same'))(x)
    x = Add()([X,x])
    x = LeakyReLU(0.2)(x)
    x = SpectralNormalization(Conv2D(256, 4, strides=2, padding='same'))(x)
    
    X = SpectralNormalization(Conv2D(512, 4, padding='same'))(x)
    x = LeakyReLU(0.2)(X)
    x = SpectralNormalization(Conv2D(512, 4, padding='same'))(x)
    x = Add()([X,x])
    x = LeakyReLU(0.2)(x)
    x = SpectralNormalization(Conv2D(512, 4, strides=2, padding='same'))(x)
    
    #Patch output
    #x = Conv2D(1, 4, strides=1, padding='same')(x)  
    
    #FCL
    x = Flatten()(x)
    x = SpectralNormalization(Dense(16))(x)
    x = LeakyReLU(0.2)(x)
    x = SpectralNormalization(Dense(4))(x)
    x = LeakyReLU(0.2)(x)
    x = SpectralNormalization(Dense(1))(x)
    return tf.keras.Model(inp, x)
    
##############################################################################################################################################
#Initilize SL-Generator and SL-Descriminator
g_model = Generator(40, 32, 1)
g_model.summary()

d_model = Discriminator((160,128,1))
d_model.summary()

##############################################################################################################################################
#Main SL-GAN training loop. Varient: SLGAN-Dense-Wloss. ***For All-loss varient, uncomment all code line from here***
D_optimizer = Adam(learning_rate=0.00005, beta_1=0.0, beta_2=0.9)
G_optimizer = Adam(learning_rate=0.00001, beta_1=0.0, beta_2=0.9)
d_total_loss = []
G_total_loss = []
#G_W_loss = []
#G_FML_loss = []
#E_loss = []
mixed_precision.set_global_policy('mixed_float16')

#vgg = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(160, 128, 3))
#vgg.trainable = False
data = pd.DataFrame(columns = ['d_total_loss', 'g_total_loss'])#, 'W_loss', 'FML_loss', 'E_loss'])
ext_model.trainable = False
def train_FL(g_model, d_model, GSN_model, real_img, W, H, n_epochs=100, n_batch=20, last_chk_point=0, folder='SLGAN'):
    bat_per_epo = int(real_img.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    #feature_extractor = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv2').output, trainable=False) 

####Train Descriminator--------------------------------------------------------------------------------------------------------------   
    for i in range(last_chk_point, n_epochs):        
        for j in range(bat_per_epo):
            d_model.trainable = True  

            X_real = load_real_samples(real_img, half_batch)
            X_fake = generate_mI(GSN_model, g_model, W, H, half_batch)

            for _ in range(6):
                with tf.GradientTape() as discriminator_tape:
                    d_real_pred = d_model(X_real, training=True)
                    d_fake_pred = d_model(X_fake, training=True)
                    d_loss = discriminator_loss(d_real_pred, d_fake_pred)
                    
                    #Gradient penalty
                    gp = gradient_penalty(d_model, X_real, X_fake)
                    lambda_gp = 10.0
                    d_loss = d_loss + (lambda_gp * gp)
    
                discriminator_grads = discriminator_tape.gradient(d_loss, d_model.trainable_variables)
                D_optimizer.apply_gradients(zip(discriminator_grads, d_model.trainable_variables))

####Train Generator--------------------------------------------------------------------------------------------------------------             
            X_real = load_real_samples(real_img, n_batch)
            h_input = generate_mZs(GSN_model, W, H, n_batch)

            with tf.GradientTape() as tape:
                d_model.trainable = False
                g_pred = g_model(h_input, training=True)
                gan_pred = d_model(g_pred, training=False)
                
                Wloss = generator_loss(gan_pred)
                #Wloss = Wloss*0.5 # Lambda_1

                #d_real_features = feature_extractor(tf.repeat(np.expand_dims(X_real, axis=-1), 3, axis=-1))  #tf.repeat(np.expand_dims(X_real, axis=-1), 3, axis=-1)
                #d_fake_features = feature_extractor(tf.repeat(g_pred, 3, axis=-1)) 
                #fl = feature_matching_loss(d_real_features, d_fake_features)
                #fl = tf.cast(fl, tf.float32)
                #fl = fl*0.1 #Lambda_2

                #edges_real = tf.image.sobel_edges(tf.cast(tf.expand_dims(X_real, axis=-1), tf.float32))
                #edges_fake = tf.image.sobel_edges(tf.cast(g_pred, tf.float32))
                #edges_real_mag = tf.sqrt(tf.reduce_sum(tf.square(edges_real), axis=-1) + 1e-6)
                #edges_fake_mag = tf.sqrt(tf.reduce_sum(tf.square(edges_fake), axis=-1) + 1e-6)
                #edge_loss = (tf.abs(tf.reduce_mean(edges_real_mag) - tf.reduce_mean(edges_fake_mag)))
                #edge_loss = edge_loss*10 #Lambda_3
                
                #print('Wloss:  ', float(Wloss))
                #print('PLoss: ', float(fl))
                #print('ELoss:  ', float(edge_loss))

                g_losses = Wloss #+ fl #+ edge_loss #Total generator losses

            loss_grads = tape.gradient(g_losses, g_model.trainable_variables)
            G_optimizer.apply_gradients(zip(loss_grads, g_model.trainable_variables))
            
#####Training monitor------------------------------------------------------------------------------------------------------------------
                 
            d_total_loss.append(d_loss)
            G_total_loss.append(g_losses)
            #G_W_loss.append(Wloss)
            #G_FML_loss.append(fl)
            #E_loss.append(edge_loss)

            print('Epoch>%d, Batch %d/%d, d_loss=%.3f, g_loss=%.3f' %
                (i+1, j+1, bat_per_epo, d_loss, g_losses))
            data.loc[str(i)+'_'+str(j)] = [float(d_loss), float(g_losses)]#, float(Wloss), float(fl)]#, float(edge_loss)]
            
        if (i+1)%25 == 0:
            d_model.save(date+'/D_model_date_'+date+'_epoch_'+str(i+1)+'.keras')
            g_model.save(date+'/G_model_date_'+date+'_epoch_'+str(i+1)+'.keras')
            
        if (i+1)%5 == 0:
            data.to_csv(date+'/log.csv', index=True)
            z_input = generate_mZs(GSN_model, W, H, 10)
            X  = g_model.predict(z_input)
            plt.imsave(date+'/'+str(i+1)+'_epochs_image.png', np.squeeze(X[0], axis=-1), cmap='grey')

            eps = range(1, len(np.array(d_total_loss)) + 1)
            plt.rcParams['figure.figsize'] = [20, 8]
            plt.plot(eps, d_total_loss, 'b', label='d_loss')
            plt.plot(eps, G_total_loss, 'r', label='G_loss')
            #plt.plot(eps, G_W_loss, 'c', label='W_loss')
            #plt.plot(eps, G_FML_loss, 'y', label='FM_loss')
            #plt.plot(eps, TV_loss, 'g', label='TV_loss')
            plt.title('Losses '+str(i+1)+' Epochs')
            plt.xlabel('Epochs*'+str(bat_per_epo))
            plt.ylabel('Losses')
            plt.legend()
            plt.savefig(date+'/'+str(i+1)+'_epochs_all_losses.png')
            plt.show()

##############################################################################################################################################
#Start Training
W = 40
H = 32
n_epochs= 3000
n_batch=50
last_chk_point=0
folder='SLGAN_Dense_Wloss' #Text given here is used for creating new folder name where the models will be saved

train_FL(g_model, d_model, GSN_model, real_img, W, H, n_epochs, n_batch, last_chk_point, folder)

