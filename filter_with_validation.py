import os
import csv
import json
import jsonlines
from sklearn.decomposition import PCA

import numpy as np
from tqdm import tqdm
import random
import argparse
random.seed(1)


def fuzzy_c(X, n_clusters, centers, max_iter=50):

    def _dist(A, B):
        """Compute the euclidean distance two matrices"""
        return np.sqrt(np.einsum("ijk->ij", (A[:, None, :] - B) ** 2))

    def soft_predict(X, _centers):
        m = 2.0
        temp = _dist(X, _centers) ** float(2 / (m - 1))
        denominator_ = temp.reshape(
            (X.shape[0], 1, -1)).repeat(temp.shape[-1], axis=1)
        denominator_ = temp[:, :, np.newaxis] / denominator_
        return 1 / denominator_.sum(2)
    _centers = centers
    u = soft_predict(X, _centers)
    return u


parser = argparse.ArgumentParser(
    description="Filter coarse labeled training set with validation images")
parser.add_argument("--val_feature_path", type=str,
                    default="/path/to/feature/of/validation", help="path to validation feature")
parser.add_argument("--train_feature_path", type=str,
                    default="/path/to/feature/of/training", help="path to training feature")
parser.add_argument("--dump_path", type=str, default=".",
                    help="experiment dump path for checkpoints and log")
args = parser.parse_args()

if not os.path.exists(os.path.join(os.path.abspath('.'), args.dump_path)):
    print('log path is {}'.format(os.path.join(
        os.path.abspath('.'), args.dump_path)))
    os.makedirs(os.path.join(os.path.abspath('.'), args.dump_path))

# Get feature of validation images
val_dataset = []
val_feature = []
val_label = []

# with open("/mnt/beatle/yshen/datajsonl/WATER20211020_SHIZHENPIYAN/12_15_17_39_prediction_model_2.tsv") as fd:
with open(args.val_feature_path) as fd:
    rd = csv.reader(fd, delimiter="\t", quotechar='"')
    for row in tqdm(rd):
        unit = {}
        for token in row:
            line = json.loads(token)
            if 'label' in line:
                unit['label'] = line['label']
                val_label.append(line['label'])
            if 'predictions' in line:
                unit['predictions'] = line['predictions']
                val_feature.append(line['predictions'])
            if 'image_path' in line:
                unit['image_path'] = line['image_path']
        val_dataset.append(unit)
val_feature = np.array(val_feature)
val_label = np.array(val_label)
label_set = np.unique(val_label)
val_dataset = np.array(val_dataset)
val_number = val_feature.shape[0]

# Get feature of train images
train_dataset = []
train_feature = []
train_label = []
# with open('/mnt/beatle/yshen/code/ANY20211223_CONTRA_LEARNING/swav-master-2-6-1/data/features_add_tieba/prediction_model_2.tsv') as fd:
with open(args.train_feature_path) as fd:
    rd = csv.reader(fd, delimiter="\t", quotechar='"')
    for row in tqdm(rd):
        unit = {}
        for token in row:
            line = json.loads(token)
            if 'label' in line:
                unit['label'] = line['label']
                train_label.append(line['label'])
            if 'predictions' in line:
                unit['predictions'] = line['predictions']
                train_feature.append(line['predictions'])
            if 'image_path' in line:
                unit['image_path'] = line['image_path']
        train_dataset.append(unit)
train_feature = np.array(train_feature)
train_label = np.array(train_label)
train_dataset = np.array(train_dataset)
train_number = train_feature.shape[0]
TRAIN_LABEL_DICT = {}


SELECT = []
count = 0
pca = PCA(n_components=100, random_state=0).fit(val_feature)
val_pca = pca.transform(val_feature)

VAL_CENTER = {}
for label in label_set:
    center = np.mean(val_pca[val_label == label, :], axis=0)
    VAL_CENTER[label] = center

print('pca transform')
train_pca = pca.transform(train_feature)
print('pca transform finish')


center = []
trainlabel = {}
index = 0
for label in label_set:
    print(label)
    trainlabel[label] = index
    index += 1
    center.append(VAL_CENTER[label])
center = np.array(center)
u = fuzzy_c(train_pca, label_set.shape[0], center)
print(label_set,u.shape)

DISCARD = []
REMAINED = []

for i in range(train_dataset.shape[0]):
    u_i = u[i, :]
    maxindex = np.argmax(u_i)
    max_value = np.max(u_i)

    if trainlabel[train_dataset[i]['label']] == maxindex:
        REMAINED.append(
            {'image_path': train_dataset[i]['image_path'], 'label': train_dataset[i]['label']})
    else:
        DISCARD.append(
            {'image_path': train_dataset[i]['image_path'], 'label': train_dataset[i]['label']})


with jsonlines.open(os.path.join(args.dump_path, 'discarded.jsonl'), mode='w') as writer:
    for item in tqdm(DISCARD):
        writer.write(item)
with jsonlines.open(os.path.join(args.dump_path, 'filtered.jsonl'), mode='w') as writer:
    for item in tqdm(REMAINED):
        writer.write(item)
