FILE=$1

if [[ $FILE != "ae_photos" ]]; then
    echo "Available datasets are: summer2winter_yosemite, horse2zebra, monet2photo, cezanne2photo, ukiyoe2photo, vangogh2photo, cityscapes, ae_photos"
    exit 1
fi
if [[ ! -d './glossary']]; then
    mkdir 'glossary'
fi

echo "Specified [$FILE]"
URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/$FILE.zip
ZIP_FILE=./glossary/$FILE.zip
TARGET_DIR=./glossary/$FILE/
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_DIR
unzip $ZIP_FILE -d ./glossary/
rm $ZIP_FILE