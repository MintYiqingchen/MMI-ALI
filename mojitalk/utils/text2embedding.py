from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os.path
import scipy.spatial.distance as sd
from skip_thoughts import configuration
from skip_thoughts import encoder_manager
import argparse

# Set paths to the model.
VOCAB_FILE = "exp_vocab/vocab.txt"
EMBEDDING_MATRIX_FILE = "exp_vocab/embeddings.npy"
CHECKPOINT_PATH = "pretrained/skip_thoughts_bi_2017_02_16/model.ckpt-500008"
# The following directory should contain files rt-polarity.neg and
# rt-polarity.pos.
MR_DATA_DIR = "/mnt/lustre/yanxiaopeng/datasets/fali_mojitalk/"

suffix = ['HAPPY', 'ANGRY','PENSIVE','ABASH'] #,'SLEEP', 'UNHAPPY']
prefix = ['train', 'test']
for suf in suffix:
    vocab_file = os.path.join('mojitalk_skip', suf, 'vocab.txt')
    embedding_matrix_file = os.path.join('mojitalk_skip', suf, 'embeddings.npy')
    encoder = encoder_manager.EncoderManager()
    encoder.load_model(configuration.model_config(bidirectional_encoder=True),
            vocabulary_file=vocab_file,
            embedding_matrix_file=embedding_matrix_file,
            checkpoint_path=CHECKPOINT_PATH)
    for pre in prefix:
        fname = pre + '_' + suf + '.txt'
        fname = os.path.join(MR_DATA_DIR, fname)
        data = []
        with open(fname) as f:
            data.extend([l.strip() for l in f])
        print('data[0]: ', data[0])
        encoding = encoder.encode(data)
        print('embed[0]: ', encoding[0])
        np.save(fname[:-4] + '.npy', encoding)
