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

def generate_latent_points(W, H, n_samples, M=0, SD=10):
    z_input = []
    for i in range(0, n_samples):
        x_input = np.random.normal(M, SD, size=(W,H))
        z_input.append(x_input)
    z_input = np.array(z_input)
    return z_input

def generate_fake_samples(g_model, W, H, n_samples):
    z_input = generate_latent_points(W, H, n_samples)
    images = g_model.predict(z_input)
    return images

def generate_helper_latents(h_model, real_imgs):
    h_imgs = h_model.predict(real_imgs)
    return h_imgs

def discriminator_loss(real_output, fake_output):
    real_loss = tf.reduce_mean(real_output)
    fake_loss = tf.reduce_mean(fake_output)
    total_loss = -real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    fake_loss = tf.reduce_mean(fake_output)
    return -fake_loss

def gradient_penalty(d_model, real_images, fake_images):
    batch_size = tf.shape(real_images)[0]

    if len(real_images.shape) == 3:
        real_images = tf.expand_dims(real_images, -1)
    if len(fake_images.shape) == 3:
        fake_images = tf.expand_dims(fake_images, -1)

    real_images = tf.cast(real_images, tf.float32)
    fake_images = tf.cast(fake_images, tf.float32)

    alpha = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0., maxval=1.)
    alpha = tf.cast(alpha, real_images.dtype)  # <--- important!

    interpolated = real_images + alpha * (fake_images - real_images)

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = d_model(interpolated, training=True)

    grads = tape.gradient(pred, [interpolated])[0]
    grads_norm = tf.norm(tf.reshape(grads, [batch_size, -1]), axis=1)
    gp = tf.reduce_mean((grads_norm - 1.0) ** 2)
    return gp

##############################################################################################################################################
#Build eZ-Generator, eZ-Descriminator and Extractor
def Generator(W, H): #eZ-Generator
    in_lat = Input(shape=(W, H, 1)) 

    C3 = Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal')(in_lat)
    C3 = LeakyReLU(alpha=0.3)(C3)
    C3 = Conv2D(128, (3,3), strides=2, padding='same', kernel_initializer='he_normal')(C3)
    C3 = LeakyReLU(alpha=0.3)(C3)

    C3 = Conv2D(256, (3,3), padding='same', kernel_initializer='he_normal')(C3)
    C3 = LeakyReLU(alpha=0.3)(C3)
    C3 = Conv2D(256, (3,3), strides=2, padding='same', kernel_initializer='he_normal')(C3)
    C3 = LeakyReLU(alpha=0.3)(C3)

    C3 = Conv2D(512, (3,3), padding='same', kernel_initializer='he_normal')(C3)
    C3 = LeakyReLU(alpha=0.3)(C3)
    
    C3 = UpSampling2D(size=(2,2), interpolation="nearest")(C3)
    C3 = Conv2D(256, (3,3), padding='same', kernel_initializer='he_normal')(C3)
    C3 = LeakyReLU(alpha=0.3)(C3)
    C3 = Conv2D(256, (3,3), padding='same', kernel_initializer='he_normal')(C3)
    C3 = LeakyReLU(alpha=0.3)(C3)

    C3 = UpSampling2D(size=(2,2), interpolation="nearest")(C3)
    C3 = Conv2D(256, (3,3), padding='same', kernel_initializer='he_normal')(C3)
    C3 = LeakyReLU(alpha=0.3)(C3)
    C3 = Conv2D(256, (3,3), padding='same', kernel_initializer='he_normal')(C3)
    C3 = LeakyReLU(alpha=0.3)(C3)

    out_layer = Conv2D(1, (3,3), activation='tanh', padding='same')(C3) #320x256x1

    model = Model(in_lat, out_layer)
    return model 

def Discriminator(in_shape=(40,32,1)): #eZ-Descriminator
    in_image = Input(shape=in_shape) 
    
    fe = Conv2D(64, (3,3), strides=(2,2), padding='same')(in_image)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe) 
    fe = Flatten()(fe) 
    fe = Dense(16)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    out_layer = Dense(1)(fe)
     
    model = Model(in_image, out_layer)
    return model

def Extractor(W, H):
    img = Input(shape=(W, H, 1))
    
    C1 = Conv2D(32, (5,5), padding='same')(img) #320,256
    C1 = LeakyReLU(alpha=0.2)(C1)
    C1 = Conv2D(32, (4,4), strides=(2,2), padding='same')(C1)

    C2 = Conv2D(64, (5,5), padding='same')(C1) #160,128
    C2 = LeakyReLU(alpha=0.2)(C2)
    C2 = Conv2D(64, (4,4), strides=(2,2), padding='same')(C2)

    #C3 = Conv2D(64, (5,5), padding='same')(C2) #80,64
    #C3 = LeakyReLU(alpha=0.2)(C3)
    #C3 = Conv2D(64, (4,4), strides=(2,2), padding='same')(C3)

    C4 = Conv2D(128, (5,5), padding='same')(C2) #40,32
    C4 = LeakyReLU(alpha=0.2)(C4)

    out = Conv2D(1, (5,5), activation='tanh', padding='same')(C4)
    model = Model(img, out)
    return model

##############################################################################################################################################
#Initialize all models
g_model = Generator(40,32)
g_model.summary()

d_model = Discriminator((40,32,1))
d_model.summary()

e_model = Extractor(160, 128)
e_model.summary()

#For always loading the same extractor model. This is important to keep constant weights.
e_model.save('baseline ext_model.keras')
e_model = keras.saving.load_model('baseline ext_model.keras')
##############################################################################################################################################
#Employ Extractor to creat eZ. eZ will be the real samples for producing synthetic structured noise µeZ. High Importance!!!
real_img = generate_helper_latents(e_model, real_img)

#############################################################################################################################################
# Main training loop for eZ-Generator and ez-Descriminator
D_optimizer = Adam(learning_rate=0.00005, beta_1=0.0, beta_2=0.9)
G_optimizer = Adam(learning_rate=0.00005, beta_1=0.0, beta_2=0.9)
d_total_loss = []
G_total_loss = []

mixed_precision.set_global_policy('mixed_float16')
data = pd.DataFrame(columns = ['d_total_loss', 'g_total_loss'])
def train_FL(g_model, d_model, real_img, W, H, n_epochs=100, n_batch=20, last_chk_point=0, folder='eZ_generator'):
    bat_per_epo = int(real_img.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    
####Train Descriminator 
    for i in range(last_chk_point, n_epochs):        
        for j in range(bat_per_epo):
            d_model.trainable = True  

            X_real = load_real_samples(real_img, half_batch)
            X_fake = generate_fake_samples(g_model, W, H, half_batch)

            for _ in range(5):
                with tf.GradientTape() as discriminator_tape:
                    d_real_pred = d_model(X_real, training=True)
                    d_fake_pred = d_model(X_fake, training=True)
                    d_loss = discriminator_loss(d_real_pred, d_fake_pred)
    
                    gp = gradient_penalty(d_model, X_real, X_fake)
                    lambda_gp = 10.0
                    d_loss = d_loss + (lambda_gp * gp)
    
                discriminator_grads = discriminator_tape.gradient(d_loss, d_model.trainable_variables)
                D_optimizer.apply_gradients(zip(discriminator_grads, d_model.trainable_variables))

####Train Generator             
            z_input = generate_latent_points(W, H, half_batch)
            with tf.GradientTape() as tape:
                d_model.trainable = False
                g_pred_lat = g_model(z_input, training=True)
                gan_pred_lat = d_model(g_pred_lat, training=False) 
                g_loss = generator_loss(gan_pred_lat)

            loss_grads = tape.gradient(g_loss, g_model.trainable_variables)
            G_optimizer.apply_gradients(zip(loss_grads, g_model.trainable_variables))

#####Training monitor      
            d_total_loss.append(d_loss)
            G_total_loss.append(g_loss)

            print('Epoch>%d, Batch %d/%d, d_loss=%.3f, g_loss=%.3f' %
                (i+1, j+1, bat_per_epo, d_loss, g_loss))
            data.loc[str(i)+'_'+str(j)] = [d_loss, g_loss]
            
        if (i+1)%10 == 0:
            d_model.save(date+'/D_model_date_'+date+'_epoch_'+str(i+1)+'.keras')
            g_model.save(date+'/G_model_date_'+date+'_epoch_'+str(i+1)+'.keras')
            
        if (i+1)%5 == 0:
            data.to_csv(date+'/log.csv', index=True)
            z_input = generate_latent_points(W, H, n_batch)
            X  = g_model.predict(z_input)
            plt.imsave(date+'/'+str(i+1)+'_epochs_image.png', np.squeeze(X[0], axis=-1), cmap='grey')
            fig, ax = plt.subplots(1, 4, figsize=(12, 4))
            ax[0].imshow(X[0], cmap='grey')
            ax[1].imshow(X[1], cmap='grey')
            ax[2].imshow(X_real[0], cmap='grey')
            ax[3].imshow(X_real[1], cmap='grey')
            plt.show()

            eps = range(1, len(np.array(d_total_loss)) + 1)
            plt.rcParams['figure.figsize'] = [20, 8]
            plt.plot(eps, d_total_loss, 'b', label='d_loss')
            plt.plot(eps, G_total_loss, 'r', label='G_total_loss')
            plt.title('Losses '+str(i+1)+' Epochs')
            plt.xlabel('Epochs*'+str(bat_per_epo))
            plt.ylabel('Losses')
            plt.legend()
            plt.savefig(date+'/'+str(i+1)+'_epochs_losses.png')
            plt.show()

#############################################################################################################################################
# Start training
W = 40
H = 32
n_epochs=3001
n_batch=64
last_chk_point=0
folder='eZ_Generator' #Text given here is used for creating new folder name where the models will be saved

train_FL(g_model, d_model, real_img, W, H, n_epochs, n_batch, last_chk_point, folder)

