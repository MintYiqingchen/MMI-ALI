import scipy.spatial.distance as sd
import numpy as np
import sys, os
import glob
import argparse

class Dataset(object):
    def __init__(self, embedpath, text_path, label):
        self.embedding = np.load(embedpath)
        self.sentences = []
        with open(text_path) as f:
            self.sentences.extend([l for l in f])
        self.label = label

def find_which(place, n):
    for i, v in enumerate(place):
        if n < v:
            return i
    return -1

labels = ['HAPPY', 'ANGRY', 'ABASH', 'PENSIVE']
parser = argparse.ArgumentParser()
parser.add_argument('--gt_dir', help="dir contains ground truth numpy embedding")
parser.add_argument('--phase', choices=['train', 'dev', 'test'], default='test')
parser.add_argument('--target', choices=labels, help="which domain to compare", default="ANGRY")
parser.add_argument('--gen_npy', help="generated numpy array path")
parser.add_argument('--only_vis', action="store_true")
parser.add_argument('--calc_gt', action="store_true", help="indicate if gen_npy is a ground truth array")

if __name__ == '__main__':
    args = parser.parse_args()

    # stack generated numpy array
    pattern = args.gen_npy
    fnames = glob.glob(pattern)
    fnames = sorted(fnames)
    array = np.load(fnames[0])
    for name in fnames[1:]:
        a = np.load(name)
        array = np.vstack((array, a))
    print('pattern numpy size: {}'.format(array.shape))

    ### stack ground truth ###
    numpy_format = os.path.join(args.gt_dir, args.phase+'_{}.npy')
    text_format = os.path.join(args.gt_dir, args.phase+'_{}.txt')
    datasets = [Dataset(numpy_format.format(label), text_format.format(label), label) for label in labels]
    place = [len(datasets[0].sentences)]
    for dataset in datasets[1:]:
        print('dataset size: {}'.format(len(dataset.sentences)))
        place.append(place[-1] + len(dataset.sentences))
    gt_embedding = datasets[0].embedding
    for dataset in datasets[1:]:
        # print(gt_embedding.shape, dataset.embedding.shape)
        gt_embedding = np.concatenate((gt_embedding, dataset.embedding), axis=0)
    print('gt_embedding size: {}'.format(gt_embedding.shape))

    ### calculate mmr ###
    score = 0
    gt_dataset_idx = labels.index(args.target)
    for embed, text in zip(array, datasets[gt_dataset_idx].sentences):
        # print(embed)
        scores = sd.cdist([embed], gt_embedding, "cosine")[0]
        sorted_ids = np.argsort(scores)
        if args.only_vis:
            print("Sentence:")
            print("", text)
            print("\nNearest neighbors:")

            j = 1
            for id_ in sorted_ids[:5]:
                data_idx = find_which(place, id_)
                sub_id = id_ - place[data_idx]
                print(" %d. %s (%.3f) domain: %d" %
                    (j, datasets[data_idx].sentences[sub_id], scores[sorted_ids[j-1]], data_idx))

                j += 1
            input()
        else:
            j = 1
            rank = 0
            if args.calc_gt:
                sorted_ids = sorted_ids[1:]
            for id_ in sorted_ids:
                data_idx = find_which(place, id_)
                sub_id = id_ - place[data_idx]
                if data_idx == gt_dataset_idx:
                    rank = j
                    break
                j += 1
            if rank != 0:
                score += 1 / j

    if not args.only_vis:
        print('average mrr: {}'.format(score / len(array)))
