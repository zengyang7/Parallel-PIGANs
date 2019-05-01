#!/usr/bin/env python3
"""
Created on Wed May  1 09:27:43 2019

@author: zengyang
"""

# import package
import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd
import math
import os, time


# record loss function for each network
root = './Potentialflow-results/'
if not os.path.isdir(root):
    os.mkdir(root)
    
tf.reset_default_graph()

# parameter need to be changed
num_gpus = 2
cons_value = 0
lam_cons = 0.2
train_epoch = 2
lr_setting = 0.0005

factor = 10

# number of mesh
n_mesh = 32 # number of nodes on each mesh
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
d_x  = X[:,1:]-X[:,:-1]
d_y  = Y[1:,:]-Y[:-1,:]

# use to filter divergence
filter_ = np.ones((n_mesh-1, n_mesh-1))
filter_[int(n_mesh/2)-int(n_mesh/factor):int(n_mesh/2)+int(n_mesh/factor),
        int(n_mesh/2)-int(n_mesh/factor):int(n_mesh/2)+int(n_mesh/factor)] = 0

###############################################################################
#PI-GANs

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
        
        deconv1 = tf.layers.conv2d_transpose(z, 64, [4, 4], strides=(1, 1), padding='valid', 
                                             kernel_initializer=w_init, bias_initializer=b_init)
        lrelu1 = lrelu(tf.layers.batch_normalization(deconv1, training=isTrain), 0.2)
        # 2nd hidden layer
        deconv2 = tf.layers.conv2d_transpose(lrelu1, 64, [5, 5], strides=(2, 2), padding='same', 
                                             kernel_initializer=w_init, bias_initializer=b_init)
        lrelu2 = lrelu(tf.layers.batch_normalization(deconv2, training=isTrain), 0.2)
        
        # 3rd layer
        deconv3 = tf.layers.conv2d_transpose(lrelu2, 64, [5, 5], strides=(2, 2), padding='same', 
                                             kernel_initializer=w_init, bias_initializer=b_init)
        lrelu3 = lrelu(tf.layers.batch_normalization(deconv3, training=isTrain), 0.2)
        
        # More hidden layers for different problem sizes
        if n_mesh > 32:
            # 64*64 image
            deconv3 = tf.layers.conv2d_transpose(lrelu3, 64, [5, 5], strides=(2, 2), padding='same', 
                                                 kernel_initializer=w_init, bias_initializer=b_init)
            lrelu3 = lrelu(tf.layers.batch_normalization(deconv3, training=isTrain), 0.2)
        if n_mesh > 64:
            # 128*128
            deconv3 = tf.layers.conv2d_transpose(lrelu3, 64, [5, 5], strides=(2, 2), padding='same', 
                                                 kernel_initializer=w_init, bias_initializer=b_init)
            lrelu3 = lrelu(tf.layers.batch_normalization(deconv3, training=isTrain), 0.2)
        if n_mesh > 128:
            # 256*256
            deconv3 = tf.layers.conv2d_transpose(lrelu3, 64, [5, 5], strides=(2, 2), padding='same', 
                                                 kernel_initializer=w_init, bias_initializer=b_init)
            lrelu3 = lrelu(tf.layers.batch_normalization(deconv3, training=isTrain), 0.2)
        if n_mesh > 256:
            # 512*512
            deconv3 = tf.layers.conv2d_transpose(lrelu3, 64, [5, 5], strides=(2, 2), padding='same', 
                                                 kernel_initializer=w_init, bias_initializer=b_init)
            lrelu3 = lrelu(tf.layers.batch_normalization(deconv3, training=isTrain), 0.2)
        if n_mesh > 512:
            # 1024*1024
            deconv3 = tf.layers.conv2d_transpose(lrelu3, 64, [5, 5], strides=(2, 2), padding='same', 
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
        if n_mesh > 32:
            # for 64*64
            x = tf.layers.conv2d(x, 64, [5, 5], strides=(2, 2), padding='same', 
                                     kernel_initializer=w_init, bias_initializer=b_init)
            x = lrelu(tf.layers.batch_normalization(x, training=isTrain), 0.2)
        if n_mesh > 64:
            # for 128*128
            x = tf.layers.conv2d(x, 64, [5, 5], strides=(2, 2), padding='same', 
                                     kernel_initializer=w_init, bias_initializer=b_init)
            x = lrelu(tf.layers.batch_normalization(x, training=isTrain), 0.2)
        if n_mesh > 128:
            # for 256*256
            x = tf.layers.conv2d(x, 64, [5, 5], strides=(2, 2), padding='same', 
                                     kernel_initializer=w_init, bias_initializer=b_init)
            x = lrelu(tf.layers.batch_normalization(x, training=isTrain), 0.2)
        if n_mesh > 256:
            # for 512*512
            x = tf.layers.conv2d(x, 64, [5, 5], strides=(2, 2), padding='same', 
                                     kernel_initializer=w_init, bias_initializer=b_init)
            x = lrelu(tf.layers.batch_normalization(x, training=isTrain), 0.2)
        if n_mesh > 512:
            # for 1024*1024
            x = tf.layers.conv2d(x, 64, [5, 5], strides=(2, 2), padding='same', 
                                     kernel_initializer=w_init, bias_initializer=b_init)
            x = lrelu(tf.layers.batch_normalization(x, training=isTrain), 0.2)
        conv1 = tf.layers.conv2d(x, 64, [5, 5], strides=(2, 2), padding='same', 
                                 kernel_initializer=w_init, bias_initializer=b_init)
        lrelu1 = lrelu(tf.layers.batch_normalization(conv1, training=isTrain), 0.2)

        # 2nd hidden layer
        conv2 = tf.layers.conv2d(lrelu1, 64, [5, 5], strides=(2, 2), padding='same', 
                                 kernel_initializer=w_init, bias_initializer=b_init)
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)
        
        # 3rd hidden layer
        conv3 = tf.layers.conv2d(lrelu2, 64, [5, 5], strides=(2, 2), padding='same', 
                                 kernel_initializer=w_init, bias_initializer=b_init)
        lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)

        # output layer
        conv4 = tf.layers.conv2d(lrelu3, 1, [4, 4], strides=(1, 1), padding='valid', 
                                 kernel_initializer=w_init)
        o = tf.nn.sigmoid(conv4)

        return o, conv4
    

def constraints(x, dx, dy, filtertf):
    '''
    This function is the constraints of potentianl flow, 
    L Phi = 0, L is the laplace calculator
    Phi is potential function
    '''
    # x.shape [batch_size, n_mesh, n_mesh, 2]
    u = tf.slice(x, [0,0,0,0], [batch_size, n_mesh, n_mesh, 1])
    v = tf.slice(x, [0,0,0,1], [batch_size, n_mesh, n_mesh, 1])

    # inverse normalization
    u = u*(1.1*(nor_max_v-nor_min_v)/2)+(nor_max_v+nor_min_v)/2
    v = v*(1.1*(nor_max_v-nor_min_v)/2)+(nor_max_v+nor_min_v)/2
    
    u = tf.reshape(u,[batch_size, n_mesh, n_mesh])
    v = tf.reshape(v,[batch_size, n_mesh, n_mesh])
    
    u_left = tf.slice(u, [0,0,0], [batch_size, n_mesh, n_mesh-1])
    u_right = tf.slice(u, [0,0,1], [batch_size, n_mesh, n_mesh-1])
      
    v_up = tf.slice(v, [0,0,0], [batch_size, n_mesh-1, n_mesh])
    v_down = tf.slice(v, [0,1,0], [batch_size, n_mesh-1, n_mesh])
    
    du = tf.subtract(u_right, u_left)
    dv = tf.subtract(v_down, v_up)
    
    # partial 
    du_dx = []
    dv_dy = []
    for i in range(batch_size):
        du_dx_iter = tf.divide(du[i,:,:], dx)
        du_dx.append(du_dx_iter)
        
        dv_dy_iter = tf.divide(dv[i,:,:], dy)
        dv_dy.append(dv_dy_iter)
    du_dx = tf.stack(du_dx)
    dv_dy = tf.stack(dv_dy)
    
    delta_u = tf.slice(du_dx, [0,1,0], [batch_size, n_mesh-1, n_mesh-1])
    delta_v = tf.slice(dv_dy, [0,0,1], [batch_size, n_mesh-1, n_mesh-1])
    
    divergence_field = delta_u+delta_v
    #filter divergence
    divergence_filter = []
    for i in range(batch_size):
        divergence_filter.append(tf.multiply(divergence_field[i,:,:], filtertf))
    divergence_filter = tf.stack(divergence_filter)
    
    divergence_square = tf.square(divergence_filter)
    delta = tf.reduce_mean(divergence_square,2)
    divergence_mean = tf.reduce_mean(delta, 1)
    
    # soft constraints
    kesi = tf.ones(tf.shape(divergence_mean))*(cons_value)
    delta_lose_ = divergence_mean - kesi
    delta_lose_ = tf.nn.relu(delta_lose_)
    return delta_lose_, divergence_mean

# Horovod: initialize Horovod
hvd.init()

global_step = tf.Variable(0, trainable=False)
lr = tf.train.exponential_decay(lr_setting, global_step, 500, 0.95, staircase=True)
    
# placeholder
x = tf.placeholder(tf.float32, shape=(None, n_mesh, n_mesh, n_label))
z = tf.placeholder(tf.float32, shape=(None, 1, 1, 100))
isTrain = tf.placeholder(dtype=tf.bool)
    
dx = tf.placeholder(tf.float32, shape=(n_mesh, n_mesh-1))
dy = tf.placeholder(tf.float32, shape=(n_mesh-1, n_mesh))
filtertf = tf.placeholder(tf.float32, shape=(n_mesh-1, n_mesh-1))

# graph
with tf.variable_scope(tf.get_variable_scope()) as var_scope:

    # networks : generator
    G_z = generator(z, isTrain)

    # networks : discriminator
    D_real, D_real_logits = discriminator(x, isTrain, reuse=tf.AUTO_REUSE)
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
    
    
    # optimizer for each network 
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optim = tf.train.AdamOptimizer(lr*hvd.size(), beta1=0.5)
        optim = hvd.DistributedOptimizer(optim)
        global_step = tf.contrib.framework.get_or_create_global_step()
        D_optim = optim.minimize(D_loss, global_step=global_step, var_list=D_vars)
        G_optim = optim.minimize(G_loss, global_step=global_step, var_list=G_vars)


hooks = [hvd.BroadcastGlobalVariablesHook(0),
         #Hook that requests stop at a specified step.
         tf.train.StopAtStepHook(last_step=20000 // hvd.size()),
         ### Prints the given tensors every N local steps, every N seconds, or at end.
         tf.train.LoggingTensorHook(tensors={'step': global_step, 'loss': [D_loss,G_loss]},
                                             every_n_iter=10),
                                             ]

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())

checkpoint_dir = './checkpoints' if hvd.rank() == 0 else None

filename_TFRecord = 'Potentialflow'+str(n_mesh)+'.tfrecord'
# load tf.record
queue_train = tf.data.TFRecordDataset(filename_TFRecord)
dataset_train = queue_train.map(read_tfrecord).repeat().batch(batch_size)
iterator_train = dataset_train.make_initializable_iterator()
next_element_train = iterator_train.get_next()

np.random.seed(int(time.time()))
print('training start!')
start_time = time.time()

with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                       hooks=hooks,config=config) as mon_sess:
    while not mon_sess.should_stop():
        mon_sess.run(iterator_train.initializer)
        x_, _ = mon_sess.run(next_element_train)
        x_[:,:,:,0:2] = (x_[:,:,:,0:2]-(nor_max_v+nor_min_v)/2)/(1.1*(nor_max_v-nor_min_v)/2)
        x_[:,:,:,2] = (x_[:,:,:,2]-(nor_max_p+nor_min_p)/2)/(1.1*(nor_max_p-nor_min_p)/2)
                
        z_ = np.random.normal(0, 1, (num_gpus*batch_size, 1, 1, 100))
                
        loss_d_, _ = mon_sess.run([D_loss, D_optim], {x: x_, z: z_, isTrain: True})
                
        # training generator
        z_ = np.random.normal(0, 1, (num_gpus*batch_size, 1, 1, 100))
        loss_g_, _ = mon_sess.run([G_loss, G_optim], {z:z_, x:x_, dx:d_x, dy:d_y, filtertf:filter_, isTrain: True})
