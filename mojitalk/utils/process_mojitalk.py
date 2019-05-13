import unicodedata
import sys
import os, re
import argparse
HAPPY_NAME = set(['SMILING FACE WITH SMILING EYES', 'SMILING FACE WITH OPEN MOUTH AND SMILING EYES', 'WHITE SMILING FACE'])
ANGRY_NAME = set(['CONFUSED FACE', 'FACE WITH LOOK OF TRIUMPH', 'POUTING FACE', 'ANGRY FACE'])
PENSIVE_NAME = set(['WEARY FACE', 'PENSIVE FACE', 'TIRED FACE', 'PERSEVERING FACE', 'CONFOUNDED FACE'])
UNHAPPY_NAME = ['UNAMUSED FACE']
ABASH_NAME = ['FLUSHED FACE']
SLEEP_NAME = set(['SLEEPING FACE', 'SLEEPY FACE'])

def write_to_file(dirname, prefix, write_dict):
    for key, value in write_dict.items():
        fname = os.path.join(dirname, prefix+'_' + key + '.txt')
        with open(fname, 'w', encoding='utf8') as f:
            f.write("\n".join(value))

def filter_emoji(desstr, restr=''):
    try:
        co = re.compile(u'[\U00010000-\U0010ffff]')
    except re.error:
        co = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')
    return co.sub(restr, desstr)
parser = argparse.ArgumentParser()
parser.add_argument('--src_dir', type=str)
parser.add_argument('--dst_dir', type=str)

if __name__ == '__main__':
    args = parser.parse_args()
    if not os.path.exists(args.dst_dir):
        os.makedirs(args.dst_dir)

    root_dir = args.src_dir
    dest_dir = args.dst_dir

    prefix = ['train', 'dev', 'test']
    all_write_dict = {'HAPPY':[], 'ANGRY': [], 'PENSIVE':[], 'UNHAPPY': [], 'ABASH': [], 'SLEEP': []}
    write_dict = {'HAPPY':[], 'ANGRY': [], 'PENSIVE':[], 'UNHAPPY': [], 'ABASH': [], 'SLEEP': []}
    for pre in prefix:
        with open(os.path.join(root_dir, pre + '.ori'), encoding='utf8') as orif:
            ori = orif.readlines()
        with open(os.path.join(root_dir, pre + '.rep'), encoding='utf8') as repf:
            rep = repf.readlines()
        for up, down in zip(ori, rep):
            name = unicodedata.name(up[0])
            if name in HAPPY_NAME:
                line = up.strip() + '\t' + down.strip()
                line = filter_emoji(line).lower()
                line = re.sub(r'\d', '#', line)
                write_dict['HAPPY'].append(line.strip())
            elif name in ANGRY_NAME:
                line = up.strip() + '\t' + down.strip()
                line = filter_emoji(line).lower()
                line = re.sub(r'\d', '#', line)
                write_dict['ANGRY'].append(line.strip())
            elif name in PENSIVE_NAME:
                line = up.strip() + '\t' + down.strip()
                line = filter_emoji(line).lower()
                line = re.sub(r'\d', '#', line)
                write_dict['PENSIVE'].append(line.strip())
            elif name in UNHAPPY_NAME:
                line = up.strip() + '\t' + down.strip()
                line = filter_emoji(line).lower()
                line = re.sub(r'\d', '#', line)
                write_dict['UNHAPPY'].append(line.strip())
            elif name in ABASH_NAME:
                line = up.strip() + '\t' + down.strip()
                line = filter_emoji(line).lower()
                line = re.sub(r'\d', '#', line)
                write_dict['ABASH'].append(line.strip())
            elif name in SLEEP_NAME:
                line = up.strip() + '\t' + down.strip()
                line = filter_emoji(line).lower()
                line = re.sub(r'\d', '#', line)
                write_dict['SLEEP'].append(line.strip())

        
        if pre == "train":
            min_length = min(map(len, write_dict.values()))
            for k, v in write_dict.items():
                write_dict[k] = write_dict[k][:min_length]
                all_write_dict[k].extend(write_dict[k])

            write_to_file(dest_dir, pre, write_dict)
            write_dict = {'HAPPY':[], 'ANGRY': [], 'PENSIVE':[], 'UNHAPPY': [], 'ABASH': [], 'SLEEP': []}
        elif pre == "test":
            min_length = min(map(len, write_dict.values()))
            for k, v in write_dict.items():
                write_dict[k] = write_dict[k][:min_length]
                all_write_dict[k].extend(write_dict[k])
            write_to_file(dest_dir, pre, write_dict)

    write_to_file(dest_dir, 'skip_ver', all_write_dict)
