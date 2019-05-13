from fali import dataset, models, trainers
import tensorflow as tf
import os
flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS
# ----- model setting
flags.DEFINE_integer('image_size', 256, 'model-used image size')
flags.DEFINE_integer('load_size', 256, 'model-used image size')
flags.DEFINE_integer('image_ch', 3, 'model-used image channel')
flags.DEFINE_integer('latent_dim', 128, 'model-used latent space dimension')
flags.DEFINE_integer('domains', 3, 'how many domains will transfer')
flags.DEFINE_float('cross_weight', 1, 'cross loss weight')
flags.DEFINE_float('cycle_weight', 10, 'cycle loss weight')
# ----- save setting
flags.DEFINE_string('restore_path', '', 'used when restore ckpt saver')
flags.DEFINE_string('base_dir', './Outputs/default', 'where to create directorys')
flags.DEFINE_string('dataname', 'cityscape', 'datasets name, list in data archive')

flags.DEFINE_integer('step', 0, 'model-used image size')

def image_unify(x):
    x = (x / 127.5 ) - 1.0
    return x

def get_all():
    norm_mapf = image_unify
    if FLAGS.dataname in dataset.archive['text']:
        fali = models.TextFaliModel(FLAGS.image_size, FLAGS.latent_dim, FLAGS.domains,
            FLAGS.cross_weight, FLAGS.cycle_weight)
        testsets = []
        embedf = dataset.archive['text'][FLAGS.dataname]['embed']
        textf = dataset.archive['text'][FLAGS.dataname]['text']

        for i in range(FLAGS.domains):
            a = dataset.TextDataset(textf['test'][i], embedf['test'][i], i).batch(128)
            testsets.append(a)
        testset = tuple(testsets)
        data_kind = "text"
    return fali, testset, data_kind

if __name__ == '__main__':
    fali, testset, datakind = get_all()
    session_config = tf.ConfigProto(
                  log_device_placement=False,
                  allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True
    sess = tf.Session(config = session_config)
    sess.run(tf.global_variables_initializer())
    
    saver = tf.train.Saver()
    if os.path.isdir(FLAGS.restore_path):
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.path))
    else:
        saver.restore(sess, FLAGS.restore_path)

    evaluator = trainers.FaliEvaluator(fali, datakind, evalset = testset, is_test = True)
    for i in range(FLAGS.domains):
        evaluator.evaluate(i, sess)
