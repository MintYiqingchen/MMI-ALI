## Get Stared
### Requirements
* python==3.6
* pytorch>=0.4.1
* numpy
* python-opencv
* visdom

### Prepare Data
1. Download 2-domain cityscapes
```
bash ./datasets/download_cyclegan_dataset.sh cityscapes
```
2. Generate scratch domain
```
python ./datasets/convert_color_to_edge.py ./datasets/cityscapes/trainB ./datasets/cityscapes/trainC
python ./datasets/convert_color_to_edge.py ./datasets/cityscapes/testB ./datasets/cityscapes/testC
```

### Usage Example
```
# train unsupervised mmiali
bash cityscapes_train.sh \
    --model mmiali \
    --ndomains 3 \
    --lambda_cyc 10 \
    --lambda_semi 10 \
    --lambda_in 5 \
    --lambda_ali 0.2

# train supervised mmiali
bash cityscapes_train.sh \
    --model mmiali \
    --ndomains 3 \
    --lambda_cyc 10 \
    --lambda_semi 10 \
    --lambda_in 5 \
    --lambda_ali 0.2 \
    --supervised \
    --serial_batches

# test mmiali
sh cityscape_test.sh --model mmiali --ndomains 3
```