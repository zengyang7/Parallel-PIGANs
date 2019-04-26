#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 13:18:34 2018

@author: zengyang
"""

import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
import os, time, random
import math

tf.reset_default_graph()
np.random.seed(1)

# parameter need to be changed
cons_value = 0
lam_cons = 0.2
train_epoch = 120
lr_setting = 0.0005

# number of mesh
n_mesh = 32
n_label = 3
batch_size = 100

print('cons: %.3f lam: %.3f lr: %.6f ep: %.3f' %(cons_value, lam_cons, lr_setting, train_epoch))

# load normalization parameter
nor = np.loadtxt('NormalizedParameter')
nor_max_v = nor[0]
nor_min_v = nor[1]
nor_max_p = nor[2]
nor_min_p = nor[3]

x = [-0.5, 0.5]
y = [-0.5, 0.5]

x_mesh = np.linspace(x[0], x[1], int(n_mesh))
y_mesh = np.linspace(y[0], y[1], int(n_mesh))

# For all samples, X and Y are the same (on a same mesh)
X, Y = np.meshgrid(x_mesh, y_mesh)

# use to calculate divergence
d_x = X[:,1:]-X[:,:-1]
d_y = Y[1:,:]-Y[:-1,:]
d_x_ = np.tile(d_x, (batch_size, 1)).reshape([batch_size, n_mesh, n_mesh-1])
d_y_ = np.tile(d_y, (batch_size, 1)).reshape([batch_size, n_mesh-1, n_mesh])

# use to filter divergence
filter = np.ones((n_mesh-1, n_mesh-1))
filter[13:18,13:18] = 0
filter_batch = np.tile(filter, (batch_size, 1)).reshape([batch_size, n_mesh-1, n_mesh-1])

#----------------------------------------------------------------------------#
#GANs
def read_tfrecord(filename_queue):
    '''
    The function is used to read the tfrecord
    Inputs: 
        filename_queue -queue of file names
    Outputs:
        image
        label
    '''
    features = tf.parse_single_example(
            filename_queue,
            features={
                    'image':tf.FixedLenFeature([], tf.string),
                    'label':tf.FixedLenFeature([], tf.string)
                    })
    
    image = tf.decode_raw(features['image'], tf.float64)
    label = tf.decode_raw(features['label'], tf.float64)
    
    image = tf.reshape(image, [n_mesh, n_mesh, 3])
    label = tf.reshape(label, [3])
    
    return image, label
    
# leak_relu
def lrelu(X, leak=0.2):
    f1 = 0.5*(1+leak)
    f2 = 0.5*(1+leak)
    return f1*X+f2*tf.abs(X)

# G(z)
def generator(z, isTrain=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        # initializer
        w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
        b_init = tf.constant_initializer(0.0)
        
        deconv1 = tf.layers.conv2d_transpose(z, 256, [4, 4], strides=(1, 1), padding='valid', 
                                             kernel_initializer=w_init, bias_initializer=b_init)
        lrelu1 = lrelu(tf.layers.batch_normalization(deconv1, training=isTrain), 0.2)
        # 2nd hidden layer
        deconv2 = tf.layers.conv2d_transpose(lrelu1, 128, [5, 5], strides=(2, 2), padding='same', 
                                             kernel_initializer=w_init, bias_initializer=b_init)
        lrelu2 = lrelu(tf.layers.batch_normalization(deconv2, training=isTrain), 0.2)
        
        # 3rd layer
        deconv3 = tf.layers.conv2d_transpose(lrelu2, 64, [5, 5], strides=(2, 2), padding='same', 
                                             kernel_initializer=w_init, bias_initializer=b_init)
        lrelu3 = lrelu(tf.layers.batch_normalization(deconv3, training=isTrain), 0.2)
        
        # output layer
        deconv4 = tf.layers.conv2d_transpose(lrelu3, 3, [5, 5], strides=(2, 2), padding='same', 
                                             kernel_initializer=w_init, bias_initializer=b_init)
        o = tf.nn.tanh(deconv4)
        return o

# D(x)
def discriminator(x, isTrain=True, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        # initializer
        w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
        b_init = tf.constant_initializer(0.0)

        # 1st hidden layer
        conv1 = tf.layers.conv2d(x, 64, [5, 5], strides=(2, 2), padding='same', 
                                 kernel_initializer=w_init, bias_initializer=b_init)
        lrelu1 = lrelu(tf.layers.batch_normalization(conv1, training=isTrain), 0.2)

        # 2nd hidden layer
        conv2 = tf.layers.conv2d(lrelu1, 128, [5, 5], strides=(2, 2), padding='same', 
                                 kernel_initializer=w_init, bias_initializer=b_init)
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)
        
        # 3rd hidden layer
        conv3 = tf.layers.conv2d(lrelu2, 256, [5, 5], strides=(2, 2), padding='same', 
                                 kernel_initializer=w_init, bias_initializer=b_init)
        lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)

        # output layer
        conv4 = tf.layers.conv2d(lrelu3, 1, [4, 4], strides=(1, 1), padding='valid', 
                                 kernel_initializer=w_init)
        o = tf.nn.sigmoid(conv4)

        return o, conv4

def constraints(x, dx,dy, filtertf):
    # inverse normalization
    x = x*(1.1*(nor_max_v-nor_min_v)/2)+(nor_max_v+nor_min_v)/2
    '''
    This function is the constraints of potentianl flow, 
    L Phi = 0, L is the laplace calculator
    Phi is potential function
    '''
    # x.shape [batch_size, n_mesh, n_mesh, 2]
    u = tf.slice(x, [0,0,0,0], [batch_size, n_mesh, n_mesh, 1])
    v = tf.slice(x, [0,0,0,1], [batch_size, n_mesh, n_mesh, 1])
    
    u = tf.reshape(u,[batch_size, n_mesh, n_mesh])
    v = tf.reshape(v,[batch_size, n_mesh, n_mesh])
    
    u_left = tf.slice(u, [0,0,0], [batch_size, n_mesh, n_mesh-1])
    u_right = tf.slice(u, [0,0,1], [batch_size, n_mesh, n_mesh-1])
    d_u = tf.divide(tf.subtract(u_right, u_left), dx)
    
    v_up = tf.slice(v, [0,0,0], [batch_size, n_mesh-1, n_mesh])
    v_down = tf.slice(v, [0,1,0], [batch_size, n_mesh-1, n_mesh])
    d_v = tf.divide(tf.subtract(v_down, v_up), dy)
    
    delta_u = tf.slice(d_u, [0,1,0],[batch_size, n_mesh-1, n_mesh-1])
    delta_v = tf.slice(d_v, [0,0,1],[batch_size, n_mesh-1, n_mesh-1])
    
    divergence_field = delta_u+delta_v
    #filter divergence
    divergence_filter = tf.multiply(divergence_field, filtertf)
    divergence_square = tf.square(divergence_filter)
    delta = tf.reduce_mean(divergence_square,2)
    divergence_mean = tf.reduce_mean(delta, 1)
    
    # soft constraints
    kesi = tf.ones(tf.shape(divergence_mean))*(cons_value)
    delta_lose_ = divergence_mean - kesi
    delta_lose_ = tf.nn.relu(delta_lose_)
    return delta_lose_, divergence_mean

global_step = tf.Variable(0, trainable=False)
lr = tf.train.exponential_decay(lr_setting, global_step, 500, 0.95, staircase=True)

# placeholder
x = tf.placeholder(tf.float32, shape=(None, n_mesh, n_mesh, n_label))
z = tf.placeholder(tf.float32, shape=(None, 1, 1, 100))
isTrain = tf.placeholder(dtype=tf.bool)

dx = tf.placeholder(tf.float32, shape=(None, n_mesh, n_mesh-1))
dy = tf.placeholder(tf.float32, shape=(None, n_mesh-1, n_mesh))
filtertf = tf.placeholder(tf.float32, shape=(None, n_mesh-1, n_mesh-1))

# networks : generator
G_z = generator(z, isTrain)

# networks : discriminator
D_real, D_real_logits = discriminator(x, isTrain)
D_fake, D_fake_logits = discriminator(G_z, isTrain, reuse=tf.AUTO_REUSE)
delta_lose, divergence_mean = constraints(G_z, dx, dy, filtertf)

# trainable variables for each network
T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]

lam_GP = 10

# WGAN-GP
eps = tf.random_uniform([batch_size, 1], minval=0., maxval=1.)
eps = tf.reshape(eps,[batch_size, 1, 1, 1])
eps = eps * np.ones([batch_size, n_mesh, n_mesh, 3])
X_inter = eps*x + (1. - eps)*G_z
grad = tf.gradients(discriminator(X_inter, isTrain, reuse=tf.AUTO_REUSE), [X_inter])[0]
grad_norm = tf.sqrt(tf.reduce_sum((grad)**2, axis=1))
grad_pen = lam_GP * tf.reduce_mean((grad_norm - 1)**2)

# loss for each network
D_loss_real = -tf.reduce_mean(D_real_logits)
D_loss_fake = tf.reduce_mean(D_fake_logits)
D_loss = D_loss_real + D_loss_fake + grad_pen
delta_loss = tf.reduce_mean(delta_lose)
G_loss_only = -tf.reduce_mean(D_fake_logits)
G_loss = G_loss_only + lam_cons*tf.log(delta_loss+1)


# record loss function for each network
root = './Potentialflow-results/'
if not os.path.isdir(root):
    os.mkdir(root)

# optimizer for each network 
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    optim = tf.train.AdamOptimizer(lr, beta1=0.5)
    D_optim = optim.minimize(D_loss, global_step=global_step, var_list=D_vars)
    # D_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(D_loss, var_list=D_vars)
    G_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(G_loss, var_list=G_vars)

filename_TFRecord = 'Potentialflow.tfrecord'
queue_train       = tf.data.TFRecordDataset(filename_TFRecord)
dataset_train     = queue_train.map(read_tfrecord).repeat().batch(batch_size)
iterator_train    = dataset_train.make_initializable_iterator()
next_element_train = iterator_train.get_next()

sess = tf.Session()
tf.global_variables_initializer().run()

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['delta_real'] = []
train_hist['delta_lose'] = []
train_hist['prediction'] = []
train_hist['prediction_fit'] = []
train_hist['ratio'] = []

# save model and all variables
saver = tf.train.Saver()

# training-loop
np.random.seed(int(time.time()))


print('training start!')
start_time = time.time()

for epoch in range(train_epoch+1):
    G_losses = []
    D_losses = []
    delta_real_record = []
    delta_lose_record = []
    epoch_start_time = time.time()
    for iter in range(20000 // batch_size):
        # training discriminator
        train_set,_ = sess.run(next_element_train)
        train_set[:,:,:,0:2] = (train_set[:,:,:,0:2]-(nor_max_v+nor_min_v)/2)/(1.1*(nor_max_v-nor_min_v)/2)
        train_set[:,:,:,2] = (train_set[:,:,:,2]-(nor_max_p+nor_min_p)/2)/(1.1*(nor_max_p-nor_min_p)/2)
        x_ = train_set
        z_ = np.random.normal(0, 1, (batch_size, 1, 1, 100))
        
        loss_d_, _ = sess.run([D_loss, D_optim], {x: x_, z: z_, isTrain: True})
        
        # training generator
        z_ = np.random.normal(0, 1, (batch_size, 1, 1, 100))
        loss_g_, _ = sess.run([G_loss, G_optim], {z:z_, x:x_, dx:d_x_, dy:d_y_, filtertf:filter_batch, isTrain: True})

        errD = D_loss.eval({z:z_, x:x_, filtertf:filter_batch, isTrain: False})
        errG = G_loss_only.eval({z: z_, dx:d_x_, dy:d_y_, filtertf:filter_batch, isTrain: False})
        errdelta_real = divergence_mean.eval({z:z_, dx:d_x_, dy:d_y_,filtertf:filter_batch, isTrain: False})
        errdelta_lose = delta_lose.eval({z: z_, dx:d_x_, dy:d_y_,filtertf:filter_batch, isTrain: False})
        
        D_losses.append(errD)
        G_losses.append(errG)
        delta_real_record.append(errdelta_real)
        delta_lose_record.append(errdelta_lose)

    epoch_end_time = time.time()
    if math.isnan(np.mean(G_losses)):
        break
    per_epoch_ptime = epoch_end_time - epoch_start_time
    print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f, delta: %.3f' % 
          ((epoch + 1), train_epoch, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses), np.mean(delta_real_record)))
    train_hist['D_losses'].append(np.mean(D_losses))
    train_hist['G_losses'].append(np.mean(G_losses))
    train_hist['delta_real'].append(np.mean(delta_real_record))
    train_hist['delta_lose'].append(np.mean(delta_lose_record))
    ### need change every time, PF: potential flow, 
    #root + 'PF-WGANGP-cons'+str(cons_value)+'-lam'+str(lam_cons)+'-lr'+str(lr_setting)+'-ep'+str(train_epoch)
    
    z_pred = np.random.normal(0, 1, (16, 1, 1, 100))
    prediction = G_z.eval({z:z_pred, isTrain: False})
    #prediction = prediction*np.max(U)+np.max(U)/2
    prediction[:,:,:,0:2] = prediction[:,:,:,0:2]*(1.1*(nor_max_v-nor_min_v)/2)+(nor_max_v+nor_min_v)/2
    prediction[:,:,:,2] = prediction[:,:,:,2]*(1.1*(nor_max_p-nor_min_p)/2)+(nor_max_p+nor_min_p)/2
    train_hist['prediction'].append(prediction)
    #plot_samples(X, Y, prediction)
    #plot_samples(X, Y, prediction, name)
    if epoch % 20 == 0:
        np.random.seed(1)
        z_pred = np.random.normal(0, 1, (2000, 1, 1, 100))
        prediction = G_z.eval({z:z_pred, isTrain: False})
        prediction[:,:,:,0:2] = prediction[:,:,:,0:2]*(1.1*(nor_max_v-nor_min_v)/2)+(nor_max_v+nor_min_v)/2
        prediction[:,:,:,2] = prediction[:,:,:,2]*(1.1*(nor_max_p-nor_min_p)/2)+(nor_max_p+nor_min_p)/2
        train_hist['prediction_fit'].append(prediction)

end_time = time.time()
total_ptime = end_time - start_time
name_data = root + 'PF-32-cons'+str(cons_value)+'-lam'+str(lam_cons)+'-lr'+str(lr_setting)+'-ep'+str(train_epoch)
np.savez_compressed(name_data, a=train_hist, b=per_epoch_ptime)
save_model = name_data+'.ckpt'
save_path = saver.save(sess, save_model)
