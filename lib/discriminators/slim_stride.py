import tensorflow as tf
from lib.util.ops import *
from lib.util.globals import *
from lib.util.hc_tf import *

def discriminator(config, x, g, xs, gs):
    activation = config['discriminator.activation']
    batch_size = int(x.get_shape()[0])
    depth_increase = config['discriminator.pyramid.depth_increase']
    depth = config['discriminator.pyramid.layers']
    batch_norm = config['discriminator.regularizers.layer']
    net = x
    net = conv2d(net, 64, name='d_expand', k_w=3, k_h=3, d_h=2, d_w=2)

    xgs = []
    xgs_conv = []
    for i in range(depth):
      net = batch_norm(config['batch_size']*2, name='d_expand_bn_'+str(i))(net)
      net = activation(net)
      # APPEND xs[i] and gs[i]
      if(i*2+1 < len(xs) and i > 0):
        xg = tf.concat(0, [xs[i*2+1], gs[i*2+1]])
        xg += tf.random_normal(xg.get_shape(), mean=0, stddev=config['discriminator.noise_stddev']*i, dtype=config['dtype'])

        xgs.append(xg)
  
        s = [int(x) for x in xg.get_shape()]
        moments = tf.reshape(xg, [config['batch_size'], 2, s[1], s[2], s[3]])
        moments = tf.nn.moments(xg, [1], name="d_add_moments"+str(i))
        moments = tf.reshape(xg, s)

        net = tf.concat(3, [net, xg, moments])
      filter_size_w = 2
      filter_size_h = 2
      filter = [1,filter_size_w,filter_size_h,1]
      stride = [1,filter_size_w,filter_size_h,1]
      net = conv2d(net, int(int(net.get_shape()[3])*depth_increase), name='d_expand_layer'+str(i), k_w=3, k_h=3, d_h=2, d_w=2)
      net = tf.nn.avg_pool(net, ksize=filter, strides=stride, padding='SAME')

      print('Discriminator pyramid layer:', net)

    k=-1
    net = batch_norm(config['batch_size']*2, name='d_expand_bn_end_'+str(i))(net)
    net = activation(net)
    net = tf.reshape(net, [batch_size, -1])
 
    return net


