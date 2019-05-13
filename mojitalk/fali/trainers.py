import tensorflow as tf
import os, time, sys, shutil
import skimage
import numpy as np
FLAGS = tf.app.flags.FLAGS
class ABCTrainer(object):
    ''' no arguments, use tf.app.flags.FLAGS as arguments'''
    def __init__(self):
        self.model = None
        self.trainset = None
        self.testset = None
    def train(self):
        pass
    def evaluate(self, step = 0):
        pass

def clip_train_op(loss, vars, opt, name = None):
    with tf.name_scope(name, 'clip_train_op', [loss, vars]) as scope:
        grads = tf.gradients(loss, vars)
        clipped_gradients, _= tf.clip_by_global_norm(grads, FLAGS.max_gradient_norm)
        train_op = opt.apply_gradients(zip(clipped_gradients, vars), name = scope)
        return train_op
def makeExists(path):
    if os.path.exists(path):
        return path
    os.makedirs(path)
    return path
def remove_last_when_more(num, dirname):
    items = [os.path.join(dirname, e) for e in os.listdir(dirname)]
    if len(items) <= num:
        return
    items = sorted(items, key = lambda x: os.path.getmtime(x))
    for name in items[:len(items) - num]:
        os.remove(name)
def save_tanh_image(value, fname):
    value = value * 127.5 + 127.5
    value = value.astype(np.uint8)
    img = skimage.io.imsave(fname, value)
class FaliTrainer(object):
    def __init__(self, model, gopt = None, dopt = None, trainset = None, testset = None, config = None, data_kind = 'image'):
        super(FaliTrainer, self).__init__()
        self.model = model
        self.trainset = trainset
        self.testset = testset
        self.data_kind = data_kind
        self.config = config
        self.evaluator = self._build_evaluator(data_kind)
        # -- optimizer
        self.gopt = gopt
        self.dopt = dopt
        # -- path
        self._build_path()
    
    def train(self):
        # -- data op
        iterator = self.trainset.make_one_shot_iterator()
        batch_op = iterator.get_next()
        eval_iter = self.testset.make_one_shot_iterator()
        eval_batch_op = eval_iter.get_next()
        z_prior = tf.random_uniform((FLAGS.batch_size, *self.model.z_shape)) * 2.0 - 1.0
        # -- optimize op
        if FLAGS.max_gradient_norm:
            d_train_op = clip_train_op(self.model.loss_dict['d_train_loss'], self.model.dtrainable_variables, self.dopt)
            g_train_op = clip_train_op(self.model.loss_dict['g_train_loss'], self.model.gtrainable_variables, self.gopt)
        else:
            d_train_op = self.dopt.minimize(self.model.loss_dict['d_train_loss'], var_list = self.model.dtrainable_variables)
            g_train_op = self.gopt.minimize(self.model.loss_dict['g_train_loss'], var_list = self.model.gtrainable_variables)
        # -- restore (not compatible with 1.4)
        # saveable = tf.contrib.data.make_saveable_from_iterator(iterator)
        # tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saveable)
        # saveable = tf.contrib.data.make_saveable_from_iterator(eval_iter)
        # tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saveable)
        sess = self.sess = tf.Session(config=self.config)
        sess.run(tf.global_variables_initializer())

        saver = self._build_saver(sess)
        # -- train iteration
           
        for step in range(FLAGS.start_step, FLAGS.max_steps + 1):
            start_time = time.time()
            domain_inputs = sess.run([batch_op, z_prior])
            feeddict = self._build_feeddict(domain_inputs)
            
            # -- extra fetches
            is_output = step % FLAGS.output_step == 0
            is_log = step % FLAGS.log_step == 0
            fetch_list, fetch_idx_table = self._build_fetchs(is_output, is_log, False, False)
            # -- train D or G
            if step % FLAGS.g_train_step == 0:
                res = sess.run(fetch_list + [g_train_op], feeddict)
                self._write_or_save(step, res[:-1], fetch_idx_table)
            if step % FLAGS.d_train_step == 0:
                sess.run(d_train_op, feeddict)
            # -- incremental evaluation
            domain_inputs = sess.run([eval_batch_op, z_prior])
            feeddict = self._build_feeddict(domain_inputs)
            fetch_list, fetch_idx_table = self._build_fetchs(is_output, is_log)
            self.incremental_evaluate(step, feeddict, fetch_list, fetch_idx_table)
            if is_log:
                print('Iter{} Training time: {:.2f}s'.format(step, time.time() - start_time))

        saver.save(sess, self.ckpt_path, global_step=FLAGS.max_steps)
        print('Training finish !')
        
        self.sess.close()
    def incremental_evaluate(self, step, feeddict, fetch_list, fetch_idx_table):
        self.evaluator.incremental_evaluate(step, feeddict, fetch_list, fetch_idx_table)
    def _write_or_save(self, step, res_list, idx_table):
        if step % FLAGS.ckpt_step == 0:
            self.saver.save(self.sess, self.ckpt_path, global_step = step)
        # -- output_list (images)
        if self.data_kind == "image":
            for name in self.model.outname_list:
                if name not in idx_table:
                    break
                remove_last_when_more(4 * len(self.model.outname_list), self.train_visual_path)
                fname = os.path.join(self.train_visual_path, name + '_' + str(step) + '.jpg')
                save_tanh_image(res_list[idx_table[name]][0], fname)
        # -- log_dict (values)
        s = ''
        for name in self.model.log_dict:
            if name not in idx_table:
                break
            s += ' {}: {:.3f}'.format(name, res_list[idx_table[name]])
        for name in self.model.metric_dict:
            if name not in idx_table:
                break
            s += ' {}: {:.3f}'.format(name, res_list[idx_table[name]])
        for name in self.model.loss_dict:
            if name not in idx_table:
                break
            s += ' {}: {:.3f}'.format(name, res_list[idx_table[name]])
        if(len(s) > 0):
            print('Train Step {}: '.format(step), s)
    def _build_feeddict(self, py_values):
        placeholders = self.model.placeholders_dict
        feeds = {placeholders['z_prior']: py_values[1], placeholders['use_dropout']: FLAGS.dropout}
        py_values = py_values[0]
        for ph, v in zip(placeholders['domain_inputs'], py_values):
            feeds[ph] = v 
        return feeds
    def _build_fetchs(self, outputs = False, logs = False, metrics = False, loss = False):
        table = {}
        feeds = []
        id = 0
        if(outputs):
            for name, op in zip(self.model.outname_list, self.model.output_list):
                table[name] = id
                feeds.append(op)
                id += 1
        if(metrics or (self.data_kind == 'text' and logs)):
            for name, op in self.model.metric_dict.items():
                feeds.append(op)
                table[name] = id
                id += 1
        if(logs):
            for k, v in self.model.log_dict.items():
                feeds.append(v)
                table[k] = id
                id += 1
        if(loss):
            feeds.append(self.model.loss_dict['g_train_loss'])
            table['g_train_loss'] = id; id += 1
            feeds.append(self.model.loss_dict['d_train_loss'])
            table['d_train_loss'] = id; id += 1
        return feeds, table
    def _build_saver(self, sess):
        saver = tf.train.Saver()
        path = FLAGS.restore_path
        if path != '':
            if os.path.isdir(path):
                saver.restore(sess, tf.train.latest_checkpoint(path))
            else:
                saver.restore(sess, path)
        self.saver = saver
        return saver
    def _build_path(self):
        base = FLAGS.base_dir
        self.ckpt_path = makeExists(os.path.join(base, 'ckpts', 'fali'))
        if self.data_kind == "image":
            self.train_visual_path = makeExists(os.path.join(base, 'visual', 'train'))
        if FLAGS.use_tensorboard:
            self.log_path = makeExists(os.path.join(base, 'logs'))
    def _build_evaluator(self, data_kind):
        return FaliEvaluator(self.model,self.data_kind, self)
class FaliEvaluator(ABCTrainer):
    def __init__(self, model, data_kind, trainer = None, evalset = None, is_test = False):
        self.model = model
        self.trainer = trainer
        self.evalset = evalset
        self.data_kind = data_kind
        self.is_test = is_test
        self._build_path()
    def incremental_evaluate(self, step, feeddict, fetch_list, fetch_idx_table):
        ''' cooperate with trainer '''
        if len(fetch_list) == 0:
            return
        res = self.trainer.sess.run(fetch_list, feeddict)
        self._write_and_log(step, res, fetch_idx_table)
    def evaluate(self, domain_idx, sess, bs=1):
        ''' test model from start '''
        # -- data op
        batch_op = self.evalset[domain_idx].make_one_shot_iterator().get_next() 
        z_prior = tf.random_uniform((bs, *self.model.z_shape)) * 2.0 - 1.0
        if self.data_kind == "image":
            dummy_img = np.zeros((bs, FLAGS.image_size, FLAGS.image_size, FLAGS.image_ch), dtype=np.float32)
        elif self.data_kind == "text":
            dummy_img = np.zeros((bs, FLAGS.image_size), dtype=np.float32)
        start_idx = 0
        start_time = time.time()
        # setting fetch list
        orig_name_list = self.model.outname_list
        orig_out_list = self.model.output_list
        fetch_list = []
        fetchname_list = []
        for i, name in enumerate(self.model.outname_list):
            # cross_{domain_idx}
            # gt_{domain_idx}
            print(name)
            if ('gt' in name and name[3] == str(domain_idx)) or ('cross' in name and name[6] == str(domain_idx)):
                fetch_list.append(self.model.output_list[i])
                fetchname_list.append(name)
        self.model.outname_list = fetchname_list
        self.model.output_list = fetch_list

        while 1:
            try:
                input, z = sess.run([batch_op, z_prior])
            except tf.errors.OutOfRangeError:
                break
            domain_inputs = [dummy_img] * FLAGS.domains
            domain_inputs[domain_idx] = input
            feeddict = self._build_feeddict((domain_inputs, z))
            # -- test G
            res = sess.run(self.model.output_list, feeddict)
            start_idx = self._write_output(start_idx, res)
        tf.logging.info('Evaluate time: {:.2f}s'.format(time.time() - start_time))
        self.model.outname_list = orig_name_list
        self.model.output_list = orig_out_list

    def _build_feeddict(self, py_values):
        placeholders = self.model.placeholders_dict
        feeds = {placeholders['z_prior']: py_values[1], placeholders['use_dropout']: 1.0}
        py_values = py_values[0]
        for ph, v in zip(placeholders['domain_inputs'], py_values):
            feeds[ph] = v 
        return feeds
    def _write_output(self, start_idx, output_list):
        if self.data_kind == "text" and self.is_test:
            for name, value in zip(self.model.outname_list, output_list):
                fname = os.path.join(self.eval_visual_path, name + '_' + str(start_idx) + '.npy')
                # print(fname, value[:5])
                # input()
                np.save(fname, value)
            return start_idx + 1
        for name, value in zip(self.model.outname_list, output_list):
            for img in value:
                fname = os.path.join(self.eval_visual_path, name + '_' + str(start_idx) + '.jpg')
                save_tanh_image(img, fname)
        start_idx += len(output_list[0])
        return start_idx
    def _write_and_log(self, step, res, idx_table):
        # -- output_list (images)
        if self.data_kind == "image":
            for name in self.model.outname_list:
                if name not in idx_table:
                    break
                remove_last_when_more(4 * len(self.model.outname_list), self.inc_visual_path)
                fname = os.path.join(self.inc_visual_path, name + '_' + str(step) + '.jpg')
                save_tanh_image(res[idx_table[name]][0], fname)
        # -- log_dict (values)
        s = ''
        for name in self.model.log_dict:
            if name not in idx_table:
                return
            s += ' {}: {:.3f}'.format(name, res[idx_table[name]])
        print('Train Step (eval) {}: '.format(step), s)
    def _build_path(self):
        if self.data_kind == "image":
            self.inc_visual_path = makeExists(os.path.join(FLAGS.base_dir, 'visual', 'inc'))
        self.eval_visual_path = makeExists(os.path.join(FLAGS.base_dir, 'visual', 'eval'))
