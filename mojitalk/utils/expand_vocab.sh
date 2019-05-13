INPUT_FILES="${HOME}/datasets/fali_mojitalk/skip_ver_${CLASS}.txt"
DATA_DIR="${HOME}/skip_thoughts/mojitalk_skip/${CLASS}"
bazel-bin/skip_thoughts/data/preprocess_dataset \
	  --input_files=${INPUT_FILES} \
	  --output_dir=${DATA_DIR}


CHECKPOINT_PATH="${HOME}/skip_thoughts/pretrained/skip_thoughts_bi_2017_02_16/model.ckpt-500008"
SKIP_THOUGHTS_VOCAB="${HOME}/skip_thoughts/mojitalk_skip/${CLASS}/vocab.txt"
WORD2VEC_MODEL="${HOME}/datasets/GoogleNews-vectors-negative300.bin"
EXP_VOCAB_DIR="${HOME}/skip_thoughts/mojitalk_skip/${CLASS}"
bazel-bin/skip_thoughts/vocabulary_expansion \
	--skip_thoughts_model=${CHECKPOINT_PATH} \
	--skip_thoughts_vocab=${SKIP_THOUGHTS_VOCAB} \
	--word2vec_model=${WORD2VEC_MODEL} \
	--output_dir=${EXP_VOCAB_DIR}

