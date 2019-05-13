import unicodedata
import sys
import os
HAPPY_NAME = set(['SMILING FACE WITH SMILING EYES', 'SMILING FACE WITH OPEN MOUTH AND SMILING EYES', 'WHITE SMILING FACE'])
ANGRY_NAME = set(['CONFUSED FACE', 'FACE WITH LOOK OF TRIUMPH', 'POUTING FACE', 'ANGRY FACE'])
PENSIVE_NAME = set(['WEARY FACE', 'PENSIVE FACE', 'TIRED FACE', 'PERSEVERING FACE', 'CONFOUNDED FACE'])

def write_to_file(dirname, prefix, write_dict):
    for key, value in write_dict.items():
        fname = os.path.join(dirname, prefix+'_' + key + '.txt')
        with open(fname, 'w', encoding='utf8') as f:
            f.write("\n".join(value))

if __name__ == '__main__':
    root_dir = sys.argv[1]
    new_dir = os.path.join(root_dir, 'fali_ver')
    prefix = ['train', 'dev', 'test']
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    for pre in prefix:
        with open(os.path.join(root_dir, pre + '.ori'), encoding='utf8') as orif:
            ori = orif.readlines()
        with open(os.path.join(root_dir, pre + '.rep'), encoding='utf8') as repf:
            rep = repf.readlines()
        write_dict = {'HAPPY':[], 'ANGRY': [], 'PENSIVE':[]}
        for up, down in zip(ori, rep):
            name = unicodedata.name(up[0])
            if name in HAPPY_NAME:
                write_dict['HAPPY'].append(up.strip() + '\t' + down.strip())
            elif name in ANGRY_NAME:
                write_dict['ANGRY'].append(up.strip() + '\t' + down.strip())
            elif name in PENSIVE_NAME:
                write_dict['PENSIVE'].append(up.strip() + '\t' + down.strip())
        write_to_file(new_dir, pre, write_dict)
