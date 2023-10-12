import numpy as np
import setproctitle

import torch

import create_graphs
from GG_train import *
import random
import networkx as nx
from dataReader import DataReader_snapshot
from dataLodaer import GraphData
from torch.utils.data import DataLoader
from model import GCN
from model import GAT
import torch.nn.functional as F
import processing16
from processing16 import DataReader
from multiprocessing.dummy import Pool as Pool
import heapq

global F_score_global
global exe_time

def F_score_computation(pred, labels_mul_hot, source_num):  
    B = pred.shape[0]
    F_score_total = 0
    pred = torch.transpose(pred, 1, 2)

    for i in range(B):
        F_score_one_B = 0
        
        pred_one_B = pred[i, :, 1]  
        labels_mul_hot_one_B = labels_mul_hot[i, :] 
        tmp = zip(range(len(pred_one_B.tolist())), pred_one_B.tolist())
        largeN = heapq.nlargest(source_num, tmp, key=lambda x: x[1])
        for YZ in largeN:
            if labels_mul_hot_one_B[YZ[0]] == 1:
                F_score_one_B = F_score_one_B + 1 / (source_num)
        F_score_total += F_score_one_B
    return F_score_total/B



def collate_batch(batch):

    
    B = len(batch)  

    Chanels = batch[0][-1].shape[1]  
    N_nodes_max = batch[0][-1].shape[0]

    A = torch.zeros(1, N_nodes_max, N_nodes_max)
    A[0, :, :] = batch[0][0]

    x = torch.zeros(B, 5, N_nodes_max, Chanels)
    P = torch.zeros(B, 5, N_nodes_max)
    labels = torch.zeros(B, batch[0][1].shape[0])
    for b in range(B):
        x[b, 0, :] = batch[b][3]
        x[b, 1, :] = batch[b][4]
        x[b, 2, :] = batch[b][5]
        x[b, 3, :] = batch[b][6]
        x[b, 4, :] = batch[b][7]
        P[b, :, :] = batch[b][2]
        labels[b, :] = batch[b][1]

    N_nodes = torch.from_numpy(np.array(N_nodes_max)).long()
    
    return [A, labels, P, x, N_nodes]


def one_batch_thread(batch_index, T_index, A_hat, data, out_):  
    out_[batch_index, T_index, :, ] = A_hat @ data[batch_index, T_index, :, ]

def train(train_loader, args):
    global F_score_global, exe_time
    args.device = 'cuda'

    start = time.time()
    train_loss, n_samples, F_score_total = 0, 0, 0
    for batch_idx, data in enumerate(train_loader):
        opt.zero_grad()
        B = data[3].shape[0]
        network_verNum = data[4]
        for i in range(len(data)):
            data[i] = data[i].to(args.device)

        labels = data[1]  
        labels_mul_hot = labels.to(args.device)

        pred = torch.zeros(B, 5, network_verNum, 4).to(args.device)

        matrixA = data[0][0].to(args.device)
        D = torch.diag(matrixA.sum(1)).to(args.device)
        D_hat = D ** (-0.5)
        D_hat[torch.isinf(D_hat)] = 0
        I = torch.eye(matrixA.shape[0]).to(args.device)
        A_hat = (D_hat @ (matrixA + I) @ D_hat).to(args.device).float()

        out_= torch.zeros(data[3].shape[0], data[3].shape[1], data[3].shape[2], data[3].shape[3]).to(args.device)
        
        for bi in range(0, data[3].shape[0]):
            for ti in range(0, data[3].shape[1]):
                out_[bi, ti, :, ] = A_hat @ data[3][bi, ti, :, ]

        for v_index in range(network_verNum):    
            pred[:, :, v_index, :], _ = GRU_model(out_[:, :, v_index, :])
        

        forward_out = pred[:, :, :, :2]
        backward_out = pred[:, :, :, 2:]
        average_out = (forward_out + backward_out) / 2.0   

        average_over_sequence = F.sigmoid(torch.mean(average_out, dim=1))  

        def weight_loss(pred_re, sourceNum, label_hot):
            B = pred_re.shape[0]
            loss_total = torch.tensor([0.], ).to(args.device)
            
            

            weight_I = 0.9
            weight_S = 0.1

            for i in range(B):
                pred_S_one_B = pred_re[i, 0, :]  
                pred_I_one_B = pred_re[i, 1, :]  
                loss_total = loss_total - (weight_I * sum(label_hot[i] * torch.log(pred_I_one_B)) +
                                           weight_S * sum((1 - label_hot[i]) * torch.log(pred_S_one_B))) / (pred_re.shape[2])
            loss_total = loss_total / B
            return loss_total

        average_over_sequence = torch.transpose(average_over_sequence, 1, 2)  
        
        loss = weight_loss(pred_re=average_over_sequence, sourceNum=1, label_hot=labels_mul_hot.long())
        
        loss.backward()
        opt.step()

        time_iter = time.time() - start
        train_loss += loss.item() * len(out_)
        n_samples += len(out_)
        
        
        
        F_score_global += len(out_) * F_score_computation(pred, labels_mul_hot,
                                    source_num=1)  
        exe_time += 1
        if batch_idx % 10 == 0 or batch_idx == len(train_loader) - 1 or 0 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} (avg: {:.6f})\tF-score:{:.4f} \tsecond/iteration: {:.4f}'.format(
                epoch + 1, n_samples, len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader), loss.item(), train_loss / n_samples, F_score_global / exe_time,
                time_iter / (batch_idx + 1)))


if __name__ == '__main__':
    F_score_global = 0.0
    exe_time = 0.0
    
    args = Args()
    rumor_cascade, union_graph_inf = create_graphs.create_graph1(args)

    print(len(rumor_cascade))
    print("union_graph_inf 初始化完毕!")

    
    rumor_cascade = sorted(rumor_cascade, key=lambda g: len(g.nodes))
    ten_percent = int(0.9 * len(rumor_cascade))

    subset = rumor_cascade[:ten_percent]

    right_num = 0
    all_num = 0
    all_extend_subG = []
    for index, test_cascade in enumerate(tqdm(subset)):
        try:
            
            one_hop_neighbors = set()
            for node in test_cascade.nodes():
                one_hop_neighbors.update(union_graph_inf.neighbors(node))

            
            all_nodes = set(test_cascade.nodes()) | one_hop_neighbors


            
            subgraph = union_graph_inf.subgraph(all_nodes).copy()
        except nx.NetworkXError as e:
            print(f"Error: {e}. Skipping this graph.")
            continue

        
        for node in test_cascade.nodes():
            subgraph.nodes[node]['time'] = test_cascade.nodes[node]['time']

        
        for node in one_hop_neighbors:
            if 'time' not in subgraph.nodes[node]:
                subgraph.nodes[node]['time'] = 999999

        
        all_extend_subG.append(subgraph)


    rnd_state = np.random.RandomState(1111)
    datareader = DataReader_snapshot(all_extend_subG,
                                    rnd_state=rnd_state,
                                    folds=10,
                                    union_graph_inf = union_graph_inf)

    print('datareader构建完成')

    n_folds = 10
    for fold_id in range(n_folds):
        loaders = []
        for split in ['train', 'test']:
            gdata = GraphData(fold_id=fold_id,
                              datareader=datareader,  
                              split=split)

            loader = DataLoader(gdata,  
                                batch_size=1,  
                                shuffle=True,  
                                num_workers=4,
                                collate_fn=collate_batch)  
            loaders.append(loader)  
            

        print('\nFOLD {}/{}, train {}, test {}'.format(fold_id + 1, n_folds, len(loaders[0].dataset),
                                                       len(loaders[1].dataset)))
        GCN_models_CU = []

        optimizers = []
        schedulers = []
        GRU_model = nn.LSTM(12, hidden_size=2, num_layers=3, batch_first=True,
                           bidirectional=True).to('cuda')
        print('-------------------------------------------------------------------------------------------')
        print(GRU_model)
        train_params3 = list(filter(lambda p: p.requires_grad, GRU_model.parameters()))
        opt = torch.optim.Adam([
            {'params': GRU_model.parameters(), 'lr': 0.003},
        ])

        for epoch in range(40):
            GRU_model.train()
            train(loaders[0], args)






