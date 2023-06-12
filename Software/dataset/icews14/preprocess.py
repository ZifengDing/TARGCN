import pickle
from collections import defaultdict as ddict

data_noinv, data_noinv_rest, train_data, val_data, test_data = [], [], [], [], []
entities, entities_train, entities_val, entities_test, times = set(), set(), set(), set(), set()
relations = set()
o2srt_train = ddict(list)
o2srt_train_val = ddict(list)
sr2o = ddict(list)
srt2o = ddict(list)

# object_appeared = set()
def load_data_all():
    for name in ['train.txt', 'valid.txt', 'test.txt']:
        with open(name, 'r') as f:
            for line in f:
                line_split = line.strip().split()
                sub, rel, obj, ts = line_split

                entities.add(sub)
                entities.add(obj)
                relations.add(rel)
                relations.add(str(int(rel)+230))
                times.add(ts)

                if name == 'train.txt':
                    train_data.append([int(sub), int(rel), int(obj), 24 * int(ts)])
                    train_data.append([int(obj), int(rel) + 230, int(sub), 24 * int(ts)])
                    entities_train.add(sub)
                    entities_train.add(obj)
                elif name == 'valid.txt':
                    val_data.append([int(sub), int(rel), int(obj), 24 * int(ts)])
                    val_data.append([int(obj), int(rel) + 230, int(sub), 24 * int(ts)])
                    entities_val.add(sub)
                    entities_val.add(obj)
                else:
                    test_data.append([int(sub), int(rel), int(obj), 24 * int(ts)])
                    test_data.append([int(obj), int(rel) + 230, int(sub), 24 * int(ts)])
                    entities_test.add(sub)
                    entities_test.add(obj)

def construct_o2srt():
    for q in train_data:
        o2srt_train[q[2]].append([int(q[0]), int(q[1]), int(q[3])])

def construct_o2srt_all():
    for q in train_data:
        o2srt_train_val[q[2]].append([int(q[0]), int(q[1]), int(q[3])])
    for q in val_data:
        o2srt_train_val[q[2]].append([int(q[0]), int(q[1]), int(q[3])])

def construct_sr2o():
    for q in train_data:
        sr2o[(q[0],q[1])].append(int(q[2]))
    for q in val_data:
        sr2o[(q[0],q[1])].append(int(q[2]))
    for q in test_data:
        sr2o[(q[0],q[1])].append(int(q[2]))

def construct_srt2o():
    for q in train_data:
        srt2o[(q[0],q[1],q[3])].append(int(q[2]))
    for q in val_data:
        srt2o[(q[0],q[1],q[3])].append(int(q[2]))
    for q in test_data:
        srt2o[(q[0],q[1],q[3])].append(int(q[2]))

load_data_all()

print(len(entities), len(entities_train), len(entities_val), len(entities_test))
print(len(relations))
print(len(times))
print(entities_train-entities)

construct_o2srt()
construct_o2srt_all()
construct_sr2o()
construct_srt2o()
with open('train_data.pkl', 'wb') as trd:
    pickle.dump(train_data, trd)

with open('valid_data.pkl', 'wb') as vald:
    pickle.dump(val_data, vald)

with open('test_data.pkl', 'wb') as ted:
    pickle.dump(test_data, ted)

with open('o2srt_train.pkl', 'wb') as osrtd:
    pickle.dump(o2srt_train, osrtd)

with open('o2srt_train_val.pkl', 'wb') as osrtd:
    pickle.dump(o2srt_train_val, osrtd)

with open('sr2o.pkl', 'wb') as sro:
    pickle.dump(sr2o, sro)

with open('srt2o.pkl', 'wb') as srto:
    pickle.dump(srt2o, srto)