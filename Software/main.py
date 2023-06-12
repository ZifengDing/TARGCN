import numpy as np
import time
import os
import torch
import argparse
from torch.utils.data import TensorDataset, DataLoader
from model.TARGCN_multi import TARGCN
from utils import *
from eval import *
from tqdm import tqdm


# help Module for custom Dataloader
class SimpleCustomBatch:
    def __init__(self, data):
        # print(data)
        transposed_data = list(zip(*data))
        self.src_idx = np.array(transposed_data[0], dtype=np.int32)
        self.rel_idx = np.array(transposed_data[1], dtype=np.int32)
        self.target_idx = np.array(transposed_data[2], dtype=np.int32)
        self.ts = np.array(transposed_data[3], dtype=np.int32)
        self.event_idx = np.array(transposed_data[-1], dtype=np.int32)

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.src_idx = self.src_idx.pin_memory()
        self.rel_idx = self.rel_idx.pin_memory()
        self.target_idx = self.target_idx.pin_memory()
        self.ts = self.ts.pin_memory()
        self.event_idx = self.event_idx.pin_memory()

        return self

    def __str__(self):
        return "Batch Information:\nsrc_idx: {}\nrel_idx: {}\ntarget_idx: {}\nts: {}".format(self.src_idx, self.rel_idx,
                                                                                             self.target_idx, self.ts)

def collate_wrapper(batch):
    return SimpleCustomBatch(batch)

def save_model(model, args, best_val, best_epoch, optimizer, save_path):
    state = {
        'state_dict': model.state_dict(),
        'best_val': best_val,
        'best_epoch': best_epoch,
        'optimizer': optimizer.state_dict(),
        'args': vars(args)
    }
    torch.save(state, save_path)


def load_model(load_path, optimizer, model):
    state = torch.load(load_path, map_location={'cuda:3': 'cuda:1'})
    state_dict = state['state_dict']
    best_val = state['best_val']
    best_val_mrr = best_val['mrr']

    model.load_state_dict(state_dict)
    optimizer.load_state_dict(state['optimizer'])

    return best_val_mrr

def adjust_learning_rate(optimizer, lr, gamma):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr_ = lr * gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_
    return lr_

if __name__ == '__main__':
    modelpth = './checkpoints/'
    parser = argparse.ArgumentParser(description='TARGCN')
    parser.add_argument('--score_func', type=str, default='distmult', help='score function')
    parser.add_argument('--num_epoch', type=int, default=100, help='number of maximum epoch')
    parser.add_argument('--batch_size', type=int, default=128, help='number of examples in a batch')
    parser.add_argument('--embsize', type=int, default=300, help='size of output embeddings')
    parser.add_argument('--test_step', type=int, default=1, help='test every test_step epoch')
    parser.add_argument('--test', action='store_true', help='whether testing or not')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay_step', type=int, default=30,
                        help='number of epochs after which learning rate decays if performance does not improve')
    parser.add_argument('--gamma', type=float, default=0.8, help='learning rate decay')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight of regularizer')
    parser.add_argument('--device', type=str, default='cuda:0', help='device name')
    parser.add_argument('--dataset', type=str, default='ICEWS14', help='dataset name')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout')
    parser.add_argument('--shuffle', action='store_false', help='shuffle in dataloader')
    parser.add_argument('--resume', action='store_true', help='resume one model')
    parser.add_argument('--name', type=str, default='TARGCN', help='name of the run')
    parser.add_argument('--activation', type=str, default='relu', help='activation function')
    parser.add_argument('--weight_factor', type=float, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_neighbors', type=int, default=100)
    parser.add_argument('--step', type=int, default=1)

    args = parser.parse_args()
    if not args.resume: args.name = args.name + '_' + time.strftime('%Y_%m_%d') + '_' + time.strftime('%H:%M:%S')

    logger = setup_logger(args.name)

    if not os.path.exists(modelpth):
        os.mkdir(modelpth)

    loadpth = modelpth + args.name + args.dataset
    device = args.device if torch.cuda.is_available() else 'cpu'
    args.device = device

    print("Using device: ", device)
    logger.info(vars(args))

    # seed for repeatability
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.manual_seed(0)
    np.random.seed(0)

    # load data
    num_entities, num_relations, num_ts = get_dataset_stat(args.dataset.lower())

    if args.dataset.lower()[:5] == 'icews':
        time_granularity = 24
    else:
        time_granularity = 1

    for file in ['train_data.pkl', 'valid_data.pkl', 'test_data.pkl']:
        with open('dataset/' + args.dataset.lower() + '/' + file, 'rb') as f:
            if file == 'train_data.pkl':
                train_data = pickle.load(f)
                max_time_train = max(np.array(train_data).transpose()[3])
            elif file == 'valid_data.pkl':
                val_data = pickle.load(f)
                max_time_val = max(np.array(val_data).transpose()[3])
                max_time_ = max(max_time_train, max_time_val)
            else:
                test_data = pickle.load(f)
                max_time_test = max(np.array(test_data).transpose()[3])
                max_time = max(max_time_, max_time_test)

    with open('dataset/' + args.dataset.lower() + '/o2srt_train.pkl', 'rb') as f:
        adj = pickle.load(f)  # here adj equals o2srt_train, its is used for sampling temporal neighborhood

    with open('dataset/' + args.dataset.lower() + '/o2srt_train_val.pkl', 'rb') as f:
        adj_test = pickle.load(
            f)  # here adj_test equals o2srt_train_val, its is used for sampling temporal neighborhood

    # sr2o and srt2o are for filtering in evaluation
    with open('dataset/' + args.dataset.lower() + '/sr2o.pkl', 'rb') as f:
        sr2o = pickle.load(f)

    with open('dataset/' + args.dataset.lower() + '/srt2o.pkl', 'rb') as f:
        srt2o = pickle.load(f)

    # initialize dataloader
    train_data_loader = DataLoader(train_data, batch_size=args.batch_size, collate_fn=collate_wrapper,
                                   pin_memory=False, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, collate_fn=collate_wrapper,
                             pin_memory=False, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, collate_fn=collate_wrapper,
                             pin_memory=False, shuffle=True)

    # initialize neighborfinder
    nf = NeighborFinder(adj, sampling=3, max_time=max_time, num_entities=num_entities,
                        weight_factor=args.weight_factor, time_granularity=time_granularity)

    # initialize model
    model = TARGCN(nf, args.embsize, num_entities, num_relations, logger, decoder=args.score_func, steps=args.step,
                   device=device)
    model.to(device)

    # inspect model parameters
    for name, param in model.named_parameters():
        print(name, '     ', param.size())

    # optimizer
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr = args.lr

    # whether to load model
    best_val_mrr = 0
    if args.resume:
        best_val_mrr = load_model(loadpth, optim, model)
        logger.info('Successfully Loaded previous model')

    if args.test == 0: # training
        # training loop
        for epoch in range(args.num_epoch):
            running_loss = 0
            batch_num = 0

            t_train_start = time.time()
            model.nf.set_adj(adj)
            for batch in tqdm(train_data_loader):
                optim.zero_grad()
                model.train()
                score = model(batch, num_neighbors=args.num_neighbors)
                loss = model.loss(score, torch.tensor(batch.target_idx, dtype=torch.long, device=device))

                loss.backward()
                optim.step()

                running_loss += loss.item()
                batch_num += 1

            running_loss /= batch_num
            t_train_end = time.time()

            # report loss information
            print("Epoch " + str(epoch + 1) + ": " + str(running_loss) + " Time: " + str(t_train_end - t_train_start))
            logger.info("Epoch " + str(epoch + 1) + ": " + str(running_loss) + " Time: " + str(t_train_end - t_train_start))

            # validation
            if (epoch + 1) % args.test_step == 0:
                # model.nf.set_adj(adj_test)
                results = predict(val_loader, model, args, num_entities, sr2o, srt2o, logger)

                print("===========RAW===========")
                print("Epoch {}, HITS10 {}".format(epoch + 1, results['hits@10_raw']))
                print("Epoch {}, HITS3 {}".format(epoch + 1, results['hits@3_raw']))
                print("Epoch {}, HITS1 {}".format(epoch + 1, results['hits@1_raw']))
                print("Epoch {}, MRR {}".format(epoch + 1, results['mrr_raw']))
                print("Epoch {}, MAR {}".format(epoch + 1, results['mar_raw']))

                print("=====TIME DEP FILTER=====")
                print("Epoch {}, HITS10 {}".format(epoch + 1, results['hits@10']))
                print("Epoch {}, HITS3 {}".format(epoch + 1, results['hits@3']))
                print("Epoch {}, HITS1 {}".format(epoch + 1, results['hits@1']))
                print("Epoch {}, MRR {}".format(epoch + 1, results['mrr']))
                print("Epoch {}, MAR {}".format(epoch + 1, results['mar']))

                print("====TIME INDEP FILTER====")
                print("Epoch {}, HITS10 {}".format(epoch + 1, results['hits@10_ind']))
                print("Epoch {}, HITS3 {}".format(epoch + 1, results['hits@3_ind']))
                print("Epoch {}, HITS1 {}".format(epoch + 1, results['hits@1_ind']))
                print("Epoch {}, MRR {}".format(epoch + 1, results['mrr_ind']))
                print("Epoch {}, MAR {}".format(epoch + 1, results['mar_ind']))

                logger.info("===========RAW===========")
                logger.info("Epoch {}, HITS10 {}".format(epoch + 1, results['hits@10_raw']))
                logger.info("Epoch {}, HITS3 {}".format(epoch + 1, results['hits@3_raw']))
                logger.info("Epoch {}, HITS1 {}".format(epoch + 1, results['hits@1_raw']))
                logger.info("Epoch {}, MRR {}".format(epoch + 1, results['mrr_raw']))
                logger.info("Epoch {}, MAR {}".format(epoch + 1, results['mar_raw']))

                logger.info("=====TIME DEP FILTER=====")
                logger.info("Epoch {}, HITS10 {}".format(epoch + 1, results['hits@10']))
                logger.info("Epoch {}, HITS3 {}".format(epoch + 1, results['hits@3']))
                logger.info("Epoch {}, HITS1 {}".format(epoch + 1, results['hits@1']))
                logger.info("Epoch {}, MRR {}".format(epoch + 1, results['mrr']))
                logger.info("Epoch {}, MAR {}".format(epoch + 1, results['mar']))

                logger.info("====TIME INDEP FILTER====")
                logger.info("Epoch {}, HITS10 {}".format(epoch + 1, results['hits@10_ind']))
                logger.info("Epoch {}, HITS3 {}".format(epoch + 1, results['hits@3_ind']))
                logger.info("Epoch {}, HITS1 {}".format(epoch + 1, results['hits@1_ind']))
                logger.info("Epoch {}, MRR {}".format(epoch + 1, results['mrr_ind']))
                logger.info("Epoch {}, MAR {}".format(epoch + 1, results['mar_ind']))

                if results['mrr_ind'] > best_val_mrr:
                    best_val = results
                    best_val_mrr = results['mrr_ind']
                    best_epoch = epoch
                    save_model(model, args, best_val, best_epoch, optim, loadpth)

                print("========BEST MRR=========")
                print("Epoch {}, MRR {}".format(epoch + 1, best_val_mrr))
                logger.info("========BEST MRR=========")
                logger.info("Epoch {}, MRR {}".format(epoch + 1, best_val_mrr))
    else: # test
        epoch = 0
        model.nf.set_adj(adj_test)
        results = predict(test_loader, model, args, num_entities, sr2o, srt2o, logger)

        print("===========RAW===========")
        print("Epoch {}, HITS10 {}".format(epoch + 1, results['hits@10_raw']))
        print("Epoch {}, HITS3 {}".format(epoch + 1, results['hits@3_raw']))
        print("Epoch {}, HITS1 {}".format(epoch + 1, results['hits@1_raw']))
        print("Epoch {}, MRR {}".format(epoch + 1, results['mrr_raw']))
        print("Epoch {}, MAR {}".format(epoch + 1, results['mar_raw']))

        print("=====TIME DEP FILTER=====")
        print("Epoch {}, HITS10 {}".format(epoch + 1, results['hits@10']))
        print("Epoch {}, HITS3 {}".format(epoch + 1, results['hits@3']))
        print("Epoch {}, HITS1 {}".format(epoch + 1, results['hits@1']))
        print("Epoch {}, MRR {}".format(epoch + 1, results['mrr']))
        print("Epoch {}, MAR {}".format(epoch + 1, results['mar']))

        print("====TIME INDEP FILTER====")
        print("Epoch {}, HITS10 {}".format(epoch + 1, results['hits@10_ind']))
        print("Epoch {}, HITS3 {}".format(epoch + 1, results['hits@3_ind']))
        print("Epoch {}, HITS1 {}".format(epoch + 1, results['hits@1_ind']))
        print("Epoch {}, MRR {}".format(epoch + 1, results['mrr_ind']))
        print("Epoch {}, MAR {}".format(epoch + 1, results['mar_ind']))

        logger.info("===========RAW===========")
        logger.info("Epoch {}, HITS10 {}".format(epoch + 1, results['hits@10_raw']))
        logger.info("Epoch {}, HITS3 {}".format(epoch + 1, results['hits@3_raw']))
        logger.info("Epoch {}, HITS1 {}".format(epoch + 1, results['hits@1_raw']))
        logger.info("Epoch {}, MRR {}".format(epoch + 1, results['mrr_raw']))
        logger.info("Epoch {}, MAR {}".format(epoch + 1, results['mar_raw']))

        logger.info("=====TIME DEP FILTER=====")
        logger.info("Epoch {}, HITS10 {}".format(epoch + 1, results['hits@10']))
        logger.info("Epoch {}, HITS3 {}".format(epoch + 1, results['hits@3']))
        logger.info("Epoch {}, HITS1 {}".format(epoch + 1, results['hits@1']))
        logger.info("Epoch {}, MRR {}".format(epoch + 1, results['mrr']))
        logger.info("Epoch {}, MAR {}".format(epoch + 1, results['mar']))

        logger.info("====TIME INDEP FILTER====")
        logger.info("Epoch {}, HITS10 {}".format(epoch + 1, results['hits@10_ind']))
        logger.info("Epoch {}, HITS3 {}".format(epoch + 1, results['hits@3_ind']))
        logger.info("Epoch {}, HITS1 {}".format(epoch + 1, results['hits@1_ind']))
        logger.info("Epoch {}, MRR {}".format(epoch + 1, results['mrr_ind']))
        logger.info("Epoch {}, MAR {}".format(epoch + 1, results['mar_ind']))