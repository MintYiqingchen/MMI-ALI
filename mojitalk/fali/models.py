import tensorflow as tf
from . import layers
import eval.classifier_metrics_impl as classifier_metrics
import eval.eval_utils as eval_utils
FLAGS = tf.app.flags.FLAGS
class ABCModel(object):
    ''' Model has trainable_variables after construction '''
    def __init__(self):
        self.placeholders_dict = None
        self.output_list = None # used for fetches
        self.outname_list = None
        self.loss_dict = None
        self.metric_dict = None
        self.log_dict = None
        self.trainable_variables = []
        # additional duty : define variables and submodel here
    
    def _build_input(self, *argv, **kargv):
        ''' build placeholders_dict if need'''
        pass
    def _build_body(self, *argv, **kargv):
        ''' build output_list and outname_list that contains model's evaluation output '''
        pass
    def _build_loss(self, *argv, **kargv):
        ''' build loss node and metric nodes that be used when train '''
        pass
    def _build_log(self, *argv, **kargv):
        ''' build log_dict that contains log information '''
        pass
    def _build_metrics(self, *argv, **kargv):
        ''' build metric_dict that contains log information '''
        return {}

def abs_criterion(in_, target, name = None):
    with tf.name_scope(name, 'abs_criterion', [in_, target]) as scope:
        return tf.reduce_mean(tf.abs(in_ - target))

def mae_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)

def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

def calc_cycle_loss(a_target, a, b_target, b, name = None):
    with tf.name_scope(name, 'cycle_loss', [a_target, a, b_target, b]) as scope: 
        res = tf.add(abs_criterion(a, a_target), abs_criterion(b ,b_target), name=scope)
        return res
def true_loss(pred):
    y = tf.sigmoid(pred)
    y = - tf.log(y + 1e-8)
    return tf.reduce_mean(y)
def fake_loss(pred):
    y = tf.sigmoid(pred)
    y = - tf.log(1 - y + 1e-8)
    return tf.reduce_mean(y)
def calc_gd_loss(real, fake, name = None):
    with tf.name_scope(name, 'gdloss', [real, fake]) as scope:
        a = tf.reduce_mean(tf.nn.relu(1.0 + fake))
        b = tf.reduce_mean(tf.nn.relu(1.0 - real))
        return tf.add(a, b, name = scope)

class BaseFaliModel(ABCModel):
    def __init__(self, cross_weight, cycle_weight):
        ''' subclass should define submodels in its __init__ '''
        super(BaseFaliModel, self).__init__()
        self.encoder_list = []
        self.decoder_list = []
        self.discriminator_list = []
        self.cross_weight = cross_weight
        self.cycle_weight = cycle_weight
        self.z_shape = None
        self.sups = None

    def _inner_ali(self, in_tensor, z_prior, i, use_dropout):
        with tf.name_scope('inner_ali'):
            z, features = self.encoder_list[i](in_tensor, use_dropout)
            x = self.decoder_list[i](z, features, use_dropout)
            x_fake = self.decoder_list[i](z_prior, features, use_dropout)
            z_recons, fea_recons = self.encoder_list[i](x_fake, use_dropout)        
            return z, x, z_recons, x_fake, features, fea_recons
    
    def _cycle(self, input_tensor_dict):
        fake_samples = [None] * self.domains
        dloss = tf.constant(0, dtype=tf.float32)
        gloss = tf.constant(0, dtype=tf.float32)
        cross_loss = tf.constant(0, dtype=tf.float32)
        cyc_loss = tf.constant(0, dtype=tf.float32)
        output_list = []
        outname_list = []
        ndomain = len(self.encoder_list)
        domain_inputs = input_tensor_dict['domain_inputs']
        use_dropout = input_tensor_dict['use_dropout']
        metrics = {}
        eA = self.encoder_list[0]
        eB = self.encoder_list[1]
        deA = self.decoder_list[0]
        deB = self.decoder_list[1]
        dA = self.discriminator_list[0]
        dB = self.discriminator_list[1]
        real_A = domain_inputs[0]
        real_B = domain_inputs[1]

        latent, features = eA(domain_inputs[0], use_dropout)
        fake_B = deB(latent, features, use_dropout)
        latent, features = eB(fake_B, use_dropout)
        fake_A_ = deA(latent, features, use_dropout)

        latent, features = eB(domain_inputs[1], use_dropout)
        fake_A = deA(latent, features, use_dropout)
        latent, features = eA(fake_A, use_dropout)
        fake_B_ = deB(latent, features, use_dropout)
        DB_fake = dB(fake_B, 0, use_dropout)
        DA_fake = dA(fake_A, 0, use_dropout)
        g_loss_a2b = mae_criterion(DB_fake, tf.ones_like(DB_fake))
        g_loss_b2a = mae_criterion(DA_fake, tf.ones_like(DA_fake))
        cyc_loss = self.cycle_weight * abs_criterion(real_A, fake_A_) \
            + self.cycle_weight * abs_criterion(real_B, fake_B_)

        gloss = mae_criterion(DA_fake, tf.ones_like(DA_fake)) \
            + mae_criterion(DB_fake, tf.ones_like(DB_fake)) \
            + self.cycle_weight * abs_criterion(real_A, fake_A_) \
            + self.cycle_weight * abs_criterion(real_B, fake_B_)
        output_list.append(fake_A); outname_list.append('fake_A')
        output_list.append(fake_B); outname_list.append('fake_B')

        DB_real = dB(real_B, 0, use_dropout)
        DA_real = dA(real_A, 0, use_dropout)
        DB_fake_sample = dB(fake_B, 0, use_dropout)
        DA_fake_sample = dA(fake_A, 0, use_dropout)

        db_loss_real = mae_criterion(DB_real, tf.ones_like(DB_real))
        db_loss_fake = mae_criterion(DB_fake_sample, tf.zeros_like(DB_fake_sample))
        db_loss = (db_loss_real + db_loss_fake) / 2
        da_loss_real = mae_criterion(DA_real, tf.ones_like(DA_real))
        da_loss_fake = mae_criterion(DA_fake_sample, tf.zeros_like(DA_fake_sample))
        da_loss = (da_loss_real + da_loss_fake) / 2
        dloss = da_loss + db_loss
        # 3. build trainer node
        self.log_dict = {'gloss': gloss, 'dloss': dloss, 'cyc_loss': cyc_loss}
        self.output_list = output_list
        self.outname_list = outname_list
        self.metric_dict = metrics

        self.loss_dict = {'d_train_loss': dloss,
            'g_train_loss': gloss}
    def _build_body(self, input_tensor_dict):
        ''' input_tensor_dict keys:
            domain_inputs : tensor lists
            z_prior: a random operator
        '''
        # 1. preparation
        dloss = tf.constant(0, dtype=tf.float32)
        gloss = tf.constant(0, dtype=tf.float32)
        cross_loss = tf.constant(0, dtype=tf.float32)
        cyc_loss = tf.constant(0, dtype=tf.float32)
        output_list = []
        outname_list = []
        ndomain = len(self.encoder_list)
        domain_inputs = input_tensor_dict['domain_inputs']
        # share a z_prior tensor across all domain ?
        z_prior = input_tensor_dict['z_prior']
        use_dropout = input_tensor_dict['use_dropout']
        # 2. domain transfer
        metrics = {}
        features_list = []
        zs = []
        fake_list = []
        for i in range(ndomain):
            input_i = domain_inputs[i]
            zi, _, z_recons, x_fake, features, fea_recons = self._inner_ali(input_i, z_prior, i, use_dropout)
            real_pred = self.discriminator_list[i](input_i,zi, use_dropout,name='real_pred')
            fake_pred = self.discriminator_list[i](x_fake, z_prior, use_dropout,name='fake_pred')
            gloss += mae_criterion(real_pred, tf.zeros_like(real_pred)) + mae_criterion(fake_pred, tf.ones_like(fake_pred))
            dloss += mae_criterion(real_pred, tf.ones_like(real_pred)) + mae_criterion(fake_pred, tf.zeros_like(fake_pred))
            cyc_loss += abs_criterion(z_prior, z_recons)
            # for f1, f2 in zip(features, fea_recons):
            #     cyc_loss += abs_criterion(f1, f2) / (len(features) * 1.0)
            features_list.append(features)
            zs.append(zi)
            fake_list.append(x_fake)
            output_list.append(input_i)
            outname_list.append('gt_' + str(i))

        for i in range(ndomain):
            for j in range(ndomain):
                if i == j:
                    continue
                suffix = str(i) + str(j)
                zi = zs[i]
                xj_fake = self.decoder_list[j](zi, features_list[i], use_dropout, name = 'xj_fake_' + suffix)
                if self.sups is not None and j in self.sups[i]:
                    cyc_loss += abs_criterion(xj_fake, domain_inputs[j])
                ret_j, feat_j = self.encoder_list[j](xj_fake, use_dropout)
                x_recons = self.decoder_list[i](ret_j, feat_j, use_dropout)
                cyc_loss += abs_criterion(x_recons, domain_inputs[i])
                fake_pred = self.discriminator_list[j](xj_fake, zi, use_dropout, 'cross_fake_pred')
                gloss += mae_criterion(fake_pred, tf.ones_like(fake_pred))
                dloss += mae_criterion(fake_pred, tf.zeros_like(fake_pred))

                xj_fake_in = self.decoder_list[j](z_prior, features_list[j], use_dropout)
                zj_hat, feat_j_hat = self.encoder_list[j](xj_fake_in, use_dropout)
                x_recons_hat = self.decoder_list[i](zj_hat, feat_j_hat, use_dropout)
                cyc_loss += abs_criterion(x_recons_hat, fake_list[i])

                output_list.append(xj_fake)
                outname_list.append('cross_' + suffix)
                metrics.update(self._build_metrics(domain_inputs[j], xj_fake, prefix = 'cross_'+suffix))
            
        
        # 3. build trainer node
        self.log_dict = {'gloss': gloss, 'dloss': dloss, 'cyc_loss': cyc_loss}
        self.output_list = output_list
        self.outname_list = outname_list
        self.metric_dict = metrics

        self.loss_dict = {'d_train_loss': dloss,
            'g_train_loss': gloss + cyc_loss * self.cycle_weight}


class TextFaliModel(BaseFaliModel):
    def __init__(self, input_dim, latent_dim, domains = 3,
            cross_weight = 0, cycle_weight = 0):
        super(TextFaliModel, self).__init__(cross_weight, cycle_weight)
        self.domains = domains
        self.latent_size = latent_dim
        self.input_size = input_dim
        # build layer and trainable_variables
        for i in range(domains):
            self.encoder_list.append(layers.TextEncoder(latent_dim))
            self.decoder_list.append(layers.TextDecoder(input_dim))
            self.discriminator_list.append(layers.TextDiscriminator())
        self.z_shape = self.encoder_list[0].output_shape
        # build model components
        self._build_input()
        self._build_body(self.placeholders_dict)
        # self._cycle(self.placeholders_dict)
        
        # build layer and trainable_variables
        self.gtrainable_variables = []
        self.dtrainable_variables = []
        for i in range(domains):
            self.gtrainable_variables.extend(self.encoder_list[i].trainable_variables)
            self.gtrainable_variables.extend(self.decoder_list[i].trainable_variables)
            self.dtrainable_variables.extend(self.discriminator_list[i].trainable_variables)
        self.trainable_variables = self.gtrainable_variables + self.dtrainable_variables

    def _build_input(self):
        a = []
        for _ in range(self.domains):
            a.append(tf.placeholder(tf.float32, shape=(None, self.input_size)))
        res = {
            'z_prior': tf.placeholder(tf.float32, shape= (None, *self.z_shape)),
            'domain_inputs': a,
            'use_dropout': tf.placeholder(tf.float32, shape=())
            }
        self.placeholders_dict = res
    def _build_metrics(self, real, fake, prefix = 'gan_metrics'):
        with tf.name_scope(prefix, 'gan_metrics', [real, fake]) as scope:
            distance = tf.norm(real - fake)
            res = {prefix+' distance': distance}
        return res

if __name__ == '__main__':
    textFali = TextFaliModel(32, 32, cycle_weight=0.001, cross_weight=0.001)
    print('dvars: {}, gvars:{}'.format(len(imgFali.dtrainable_variables), len(imgFali.gtrainable_variables)))

    z = tf.random_normal((1, *textFali.z_shape))
    dummy = tf.random_normal((1, 32))

    placeholders = textFali.placeholders_dict
    output_list = textFali.output_list
    loss_list = [textFali.loss_dict['d_train_loss'], textFali.loss_dict['g_train_loss']]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        numpy_in = sess.run([z, dummy, dummy, dummy])
        feeds = {placeholders['z_prior']: numpy_in[0]}
        for i, plc in enumerate(placeholders['domain_inputs']):
            feeds[plc] = numpy_in[i+1]
        
        losses = sess.run(loss_list, feed_dict = feeds)

    print('TextFali gloss: {} dloss: {}'.format(losses[1], losses[0]))
        
