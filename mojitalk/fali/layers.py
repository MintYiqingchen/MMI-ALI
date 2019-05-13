import tensorflow as tf
from math import log2
layers = tf.layers
class ABCLayer(object):
    ''' Layer has trainable_variables after the first call '''
    def __init__(self, *args, **kwargs):
        ''' must define variables here ! '''
        self.trainable_variables = []
        self.has_build = False
    def __call__(self, *args, **kwargs):
        pass
    def _build_trainable_variables(self, *args, **kwargs):
        pass

class InstanceNorm(ABCLayer):
    count = 0
    def __init__(self, depth, name = "instance_norm"):
        with tf.variable_scope(name + str(InstanceNorm.count)):
            self.scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
            self.offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        self.trainable_variables = [self.scale, self.offset]
        InstanceNorm.count += 1
    def __call__(self, input, name = "inorm"):
        with tf.name_scope(name):
            mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
            epsilon = 1e-5
            inv = tf.rsqrt(variance + epsilon)
            normalized = (input-mean)*inv
            return self.scale*normalized + self.offset
# ----------------- Text ---------------------
class TextEncoder(ABCLayer):
    LAYER_CHANNEL = [1024, 512, 256, 128]
    def __init__(self, output_dim):
        super(TextEncoder, self).__init__()
        blocks = []
        for ch in TextEncoder.LAYER_CHANNEL:
            blocks.append((layers.Dense(ch, tf.nn.relu), layers.BatchNormalization()))

        self.linear = layers.Dense(output_dim)
        self.blocks = blocks
        self.output_shape = (output_dim, )
    def __call__(self, y, use_dropout, name = None):
        features = []
        with tf.name_scope(name, 'TextEncoder', [y, use_dropout]) as scope:
            for block in self.blocks:
                y = block[1](block[0](y))
                y = tf.nn.dropout(y, use_dropout)
                features.append(y)
            y = self.linear(y)
        self._build_trainable_variables()
        return y, features
    def _build_trainable_variables(self):
        for i in range(len(self.blocks)):
            self.trainable_variables.extend(self.blocks[i][0].trainable_variables)
            self.trainable_variables.extend(self.blocks[i][1].trainable_variables)
        self.trainable_variables.extend(self.linear.trainable_variables)
class TextDecoder(ABCLayer):
    LAYER_CHANNEL = [128, 256, 512, 1024]
    def __init__(self, output_dim):
        super(TextDecoder, self).__init__()
        blocks = []
        for ch in TextDecoder.LAYER_CHANNEL:
            blocks.append((layers.Dense(ch, tf.nn.relu), layers.BatchNormalization()))

        self.linear = layers.Dense(output_dim)
        self.blocks = blocks
    def __call__(self, y, features, use_dropout, name = None):
        with tf.name_scope(name, 'TextDecoder', [y, use_dropout, *features]) as scope:
            for fea,block in zip(features, self.blocks):
                y = block[1](block[0](y))
                y = tf.nn.dropout(y, use_dropout)
                y = tf.concat((fea, y), 1)
            y = self.linear(y)
        self._build_trainable_variables()
        return y
    def _build_trainable_variables(self):
        for i in range(len(self.blocks)):
            self.trainable_variables.extend(self.blocks[i][0].trainable_variables)
            self.trainable_variables.extend(self.blocks[i][1].trainable_variables)
        self.trainable_variables.extend(self.linear.trainable_variables)
class TextDiscriminator(ABCLayer):
    LAYER_CHANNEL = [1024, 512, 256, 128]
    def __init__(self):
        super(TextDiscriminator, self).__init__()
        blocks = []
        for ch in TextDiscriminator.LAYER_CHANNEL:
            blocks.append((layers.Dense(ch, tf.nn.relu), layers.BatchNormalization()))
        self.linear = layers.Dense(1)
        self.blocks = blocks
    def __call__(self, y, z, use_dropout,name = None):
        with tf.name_scope(name, 'TextDiscriminator', [y, z, use_dropout]) as scope:
            y = tf.concat((z, y), axis = 1)
            for block in self.blocks:
                y = block[1](block[0](y))
                y = tf.nn.dropout(y, use_dropout)
            y = self.linear(y)
        self._build_trainable_variables()
        return y
    def _build_trainable_variables(self):
        for i in range(len(self.blocks)):
            self.trainable_variables.extend(self.blocks[i][0].trainable_variables)
            self.trainable_variables.extend(self.blocks[i][1].trainable_variables)
        self.trainable_variables.extend(self.linear.trainable_variables)

if __name__ == '__main__':
    def train_collection():
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    encoder = TextEncoder(32)
    decoder = TextDecoder(32)
    dis = TextDiscriminator()
    dummy = tf.random_normal((1, 100))
    z, feature = encoder(dummy)
    x_fake = decoder(z, feature)
    pred = dis(dummy, z)
    print('trainable_variable: {} {} {}'.format(len(encoder.trainable_variables), len(decoder.trainable_variables), len(dis.trainable_variables)))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run([z, x_fake, pred])
    print('Text Adversarial Graph: z: {}, x_fake: {}, pred: {}'.format(res[0].shape, res[1].shape, res[2].shape))
    
