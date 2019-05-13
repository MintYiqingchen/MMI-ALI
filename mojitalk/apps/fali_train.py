from fali import dataset, models, trainers
import tensorflow as tf
tf.set_random_seed(19)
# --- global value
global_step = None 
flags = tf.app.flags
FLAGS = flags.FLAGS
# ----- model setting
flags.DEFINE_integer('image_size', 256, 'model-used image size')
flags.DEFINE_integer('image_ch', 3, 'model-used image channel')
flags.DEFINE_integer('latent_dim', 128, 'model-used latent space dimension')
flags.DEFINE_integer('domains', 3, 'how many domains will transfer')
flags.DEFINE_float('cross_weight', 1, 'cross loss weight')
flags.DEFINE_float('cycle_weight', 10, 'cycle loss weight')
# ----- training setting
flags.DEFINE_string('dataname', 'cityscape', 'datasets name, list in data archive')
flags.DEFINE_integer('batch_size', 1, '')
flags.DEFINE_integer('start_step', 1, '')
flags.DEFINE_integer('max_steps', 10000, '')
flags.DEFINE_integer('d_train_step', 1, '')
flags.DEFINE_integer('g_train_step', 1, '')
flags.DEFINE_integer('output_step', 100, '')
flags.DEFINE_integer('log_step', 5, 'show log every {log_step}')
flags.DEFINE_integer('ckpt_step', 10000, 'save a checkpoint every {ckpt_step}')
flags.DEFINE_integer('load_size', 286, '')
flags.DEFINE_float('glearning_rate', 0.0002, '')
flags.DEFINE_float('dlearning_rate', 0.0002,'')
flags.DEFINE_float('beta1', 0.5, 'adam 1st order momentum')
flags.DEFINE_float('beta2', 0.999, 'adam 2st order momentum')
flags.DEFINE_float('dropout', 1.0, 'keep_prob')
flags.DEFINE_float('max_gradient_norm', 0, 'set None to chancel gradient clip')
flags.DEFINE_string('restore_path', '', 'used when restore ckpt saver')
flags.DEFINE_string('base_dir', './Outputs/default', 'where to create directorys')
flags.DEFINE_boolean('use_tensorboard', False, 'not support yet')
flags.DEFINE_boolean('stand', False, 'if true, use image standardize rather than unify')
flags.DEFINE_boolean('sup', False, 'if true, must define archive.py')

# --- functions
def get_all():
    if FLAGS.dataname in dataset.archive['text']:
        # TODO(mintyi): add text pair info?
        fali = models.TextFaliModel(FLAGS.image_size, FLAGS.latent_dim, FLAGS.domains,
            FLAGS.cross_weight, FLAGS.cycle_weight)
        trainsets = []
        embedf = dataset.archive['text'][FLAGS.dataname]['embed']
        textf = dataset.archive['text'][FLAGS.dataname]['text']
        for i in range(FLAGS.domains):
            a = dataset.TextDataset(textf['train'][i], embedf['train'][i], i).repeat().batch(FLAGS.batch_size)
            trainsets.append(a)
        trainset = tf.data.Dataset.zip(tuple(trainsets)).shuffle(50)

        testsets = []
        for i in range(FLAGS.domains):
            a = dataset.TextDataset(textf['val'][i], embedf['val'][i], i).repeat().batch(FLAGS.batch_size)
            testsets.append(a)
        testset = tf.data.Dataset.zip(tuple(testsets))
        data_kind = "text"
    return fali, trainset, testset, data_kind

def main(argv):
    fali, trainset, testset, data_kind = get_all()
    global global_step
    global_step = tf.Variable(FLAGS.start_step, trainable = False)
    # -- set opt
    learning_rate = tf.train.exponential_decay(
        FLAGS.glearning_rate, global_step, 50000, 0.96, staircase=True)
    gopt = tf.train.AdamOptimizer(learning_rate, beta1 = FLAGS.beta1, beta2 = FLAGS.beta2)
    learning_rate = tf.train.exponential_decay(
        FLAGS.dlearning_rate, global_step, 50000, 0.96, staircase=True)
    dopt = tf.train.AdamOptimizer(learning_rate, beta1 = FLAGS.beta1, beta2 = FLAGS.beta2)

    session_config = tf.ConfigProto(
                  log_device_placement=False,
                  allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True

    trainer = trainers.FaliTrainer(fali, gopt, dopt, trainset, testset, config = session_config, data_kind=data_kind)
    trainer.train()
tf.app.run()
