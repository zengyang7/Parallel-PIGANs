#!/usr/bin/env python3
"""
Created on Fri Apr 26 10:18:33 2019

@author: zengyang
"""

## import package
import numpy as np
import tensorflow as tf

n_mesh = 32 # number of nodes on each mesh

# setting of training samples
n_sam = 20000
V_mu, V_sigma = 4, 0.8
alpha_mu, alpha_sigma = 0, np.pi/4
m_mu, m_sigma = 1, 0.2

samples = np.zeros([n_sam, 3])

V_sample     = np.random.normal(V_mu, V_sigma, n_sam)
alpha_sample = np.random.normal(alpha_mu, alpha_sigma, n_sam)
m_sample     = np.random.normal(m_mu, m_sigma, n_sam)

samples[:,0] = V_sample
samples[:,1] = alpha_sample
samples[:,2] = m_sample

# generate samples
def generate_sample(n, parameter):
    ''' 
    generate samples of potential flow
    two kinds of potential flows are used : Uniform and source
    Uniform: F1(z) = V*exp(-i*alpha)*z
    source:  F2(z) = m/(2*pi)*log(z)
    x: interval of x axis
    y: interval of y axis
    n: number size of mesh
    parameter: V, alpha, m
    output: u, v the velocity of x and y direction
    '''
    # mesh
    x = [-0.5, 0.5]
    y = [-0.5, 0.5]

    x_mesh = np.linspace(x[0], x[1], int(n))
    y_mesh = np.linspace(y[0], y[1], int(n))

    # For all samples, X and Y are the same (on a same mesh)
    X, Y = np.meshgrid(x_mesh, y_mesh)  
    U    = []
    
    #filter_x = np.ones((n_mesh, n_mesh, 3))
    #filter_x[14:17,14:17] = 0
    
    # What is the index i used for?
    for i, p in enumerate(parameter):
        V = p[0]
        alpha  = p[1]
        m = p[2]
        
        # velocity of uniform
        u1 = np.ones([n, n])*V*np.cos(alpha)
        v1 = np.ones([n, n])*V*np.sin(alpha)
        
        # velocity of source
        # u2 = m/2pi * x/(x^2+y^2)
        # v2 = m/2pi * y/(x^2+y^2)
        u2 = m/(2*np.pi)*X/(X**2+Y**2)
        v2 = m/(2*np.pi)*Y/(X**2+Y**2)
        
        u = u1+u2
        v = v1+v2
        
        # Bernoulli's principle
        # constant=0, rho = 1
        p = 0-1/2*(u**2+v**2)
        
        U_data = np.zeros([n, n, 3])
        
        U_data[:, :, 0] = u
        U_data[:, :, 1] = v
        U_data[:, :, 2] = p
        U.append(U_data)
    return X, Y, np.asarray(U)

# generate training samples
X, Y, U = generate_sample(n=n_mesh, parameter=samples)

# normalization
nor = []
nor.append(np.max(U[:,:,:,0:2]))
nor.append(np.min(U[:,:,:,0:2]))
nor.append(np.max(U[:,:,:,2]))
nor.append(np.min(U[:,:,:,2]))

def make_example(image, label):
    return tf.train.Example(features=tf.train.Features(feature={
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
            'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label]))
    }))

def write_tfrecord(datas, labels, filename_TFRecord):
    '''
    This function is to write the TFRecord file
    
    Inputs:
        datas - the training or testing data
        labels - labels
        filename_TFRecord - name of TFRecord
    '''
    writer = tf.python_io.TFRecordWriter(filename_TFRecord)
    num = len(datas)
    for i in range(num):
        data = datas[i]
        label = labels[i]
        ex = make_example(data.tobytes(), label.tobytes())
        
        # 需要写在循环里
        writer.write(ex.SerializeToString())
    writer.close()

filename_TFRecord = 'Potentialflow.tfrecord'
write_tfrecord(U, samples, filename_TFRecord)
print('Write the tfrecord successfully!')

##############################################################################
# This is to verify the tf.record

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

# load tf.record
queue_train = tf.data.TFRecordDataset(filename_TFRecord)
dataset_train = queue_train.map(read_tfrecord).repeat().batch(n_sam)
iterator_train = dataset_train.make_initializable_iterator()
next_element_train = iterator_train.get_next()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(iterator_train.initializer)
batch_x, batch_y = sess.run(next_element_train)
delta_verified = batch_x - U

if np.max(delta_verified)==0.0 and np.max(delta_verified)==0.0:
    print('The tfrecord is correct!')

np.savetxt('NormalizedParameter', nor)