import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold, KFold
from torch_geometric.data import DenseDataLoader as DenseLoader
from tqdm import tqdm
import pdb

from dataloader import DataLoader  # replace with custom dataloader to handle subgraphs


def cross_validation_with_val_set(args,
                                  dataset,
                                  model,
                                  folds,
                                  epochs,
                                  batch_size,
                                  lr,
                                  lr_decay_factor,
                                  lr_decay_step_size,
                                  weight_decay,
                                  device, 
                                  logger=None):

    final_train_losses, val_losses, accs, durations = [], [], [], []
    # for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(dataset, folds))):
    for fold in [0]:
        # train_dataset = dataset[train_idx]
        # test_dataset = dataset[test_idx]
        # val_dataset = dataset[val_idx]
        if (len(dataset) == 130):
            train_dataset = dataset[:65]
            test_dataset = dataset[65:]
            val_dataset = dataset[65:]
        elif (len(dataset) == 10):
            train_dataset = dataset
            test_dataset = dataset
            val_dataset = dataset
        elif (len(dataset) == 30):
            train_dataset = dataset[:10]
            test_dataset = dataset[10:20]
            val_dataset = dataset[20:30]
        else:
            train_dataset = dataset[:65]
            val_dataset = dataset[65:130]
            test_dataset = dataset[130:]

        if 'adj' in train_dataset[0]:
            train_loader = DenseLoader(train_dataset, batch_size, shuffle=True)
            val_loader = DenseLoader(val_dataset, batch_size, shuffle=False)
            test_loader = DenseLoader(test_dataset, batch_size, shuffle=False)
        else:
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        pbar = tqdm(range(1, epochs + 1), ncols=70)
        cur_val_losses = []
        cur_accs = []

        # lzy re
        best_val_loss = 0
        best_val_acc = 0
        time_list = []
        best_model_path = "best_model_" + f"{args.task}_{args.layers}"

        for epoch in pbar:
            t1 = time.time()
            train_loss = train(args, model, optimizer, train_loader, device)
            t2 = time.time()
            val_loss, val_acc, acc_0, acc_1, recall, F1_score = eval_loss(args, model, val_loader, device)
            cur_val_losses.append(val_loss)
            cur_accs.append(val_acc)
            t3 = time.time()
            # val_acc = eval_acc(model, test_loader, device) 

            if val_loss < best_val_loss or epoch == 1:
                best_val_loss = val_loss
                model.save_model(best_model_path)
            if val_acc > best_val_acc or epoch == 1:
                best_val_acc = val_acc

            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': cur_val_losses[-1],
                'test_acc': cur_accs[-1],
            }
            # #DeleteLog
            # log = 'Fold: %d, train_loss: %0.4f, val_loss: %0.4f, test_acc: %0.4f' % (
            #     fold, eval_info["train_loss"], eval_info["val_loss"], eval_info["test_acc"]
            # )
            # pbar.set_description(log)

            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']

            # print(
            #     # f"epoch {epoch} trn time {t2-t1:.2f} val time {t3-t2:.2f} memory {torch.cuda.max_memory_allocated()/1024**3:.2f} GB  l1loss {losss:.4f} val MAE {val_score:.4f} Best MAE\patience {best_val:.4f}\\{patience}"
            #     f"epoch {epoch} | trn time {t2-t1:.2f} val time {t3-t2:.2f} | trn_loss {losss:.4f} train MAPE {train_mape_loss:.4f} | val MAE {val_score:.4f} val MAPE {val_mape_loss:.4f} | Best MAE\MAPR\patience {best_val:.4f}\\{best_val_MAPE:.4f}\\{patience}"
            # )
            time_list.append(t2-t1)
            if args.task == "FA":
                print(
                    # f"epoch {epoch} trn time {t2-t1:.2f} val time {t3-t2:.2f} memory {torch.cuda.max_memory_allocated()/1024**3:.2f} GB  l1loss {losss:.4f} val MAE {val_score:.4f} Best MAE\patience {best_val:.4f}\\{patience}"
                    f"epoch {epoch} | trn time {t2-t1:.2f} | trn_loss {train_loss:.4f}| val_losss {val_loss:.4f} | best_val_loss {best_val_loss:.4f}"
                )
            else:
                print(
                    # f"epoch {epoch} trn time {t2-t1:.2f} val time {t3-t2:.2f} memory {torch.cuda.max_memory_allocated()/1024**3:.2f} GB  l1loss {losss:.4f} val MAE {val_score:.4f} Best MAE\patience {best_val:.4f}\\{patience}"
                    f"epoch {epoch} | trn_loss {train_loss:.4f}| val_acc {val_acc:.4f} | best_val_acc {best_val_acc:.4f} | acc_0 {acc_0:.4f}, acc_1 {acc_1:.4f}, recall {recall:.4f}, F1_score {F1_score:.4f}"
                )


        # 自行增加最终测试部分
        print("loading best model...")
        model.load_model(best_model_path)
        val_loss, val_acc, acc_0, acc_1, recall, F1_score= eval_loss(args, model, test_loader, device, eval_type = "test") 
        print("---------------------------\nFinal Info:")
        if args.task == "FA":
                print(
                    # f"epoch {epoch} trn time {t2-t1:.2f} val time {t3-t2:.2f} memory {torch.cuda.max_memory_allocated()/1024**3:.2f} GB  l1loss {losss:.4f} val MAE {val_score:.4f} Best MAE\patience {best_val:.4f}\\{patience}"
                    f"test | trn time {t2-t1:.2f} | val_losss {val_loss:.4f} | best_val_loss {best_val_loss:.4f}"
                )
        else:
            print(
                # f"epoch {epoch} trn time {t2-t1:.2f} val time {t3-t2:.2f} memory {torch.cuda.max_memory_allocated()/1024**3:.2f} GB  l1loss {losss:.4f} val MAE {val_score:.4f} Best MAE\patience {best_val:.4f}\\{patience}"
                f"test | val_acc {val_acc:.4f} | acc_0 {acc_0:.4f}, acc_1 {acc_1:.4f}, recall {recall:.4f}, F1_score {F1_score:.4f}"
            )
        print(f"agv time: {sum(time_list)/len(time_list):.4f}")
        print("----------------------------")

        val_losses += cur_val_losses
        accs += cur_accs

        loss, argmin = tensor(cur_val_losses).min(dim=0)
        acc = cur_accs[argmin.item()]
        final_train_losses.append(eval_info["train_loss"])
        # #DeleteLog
        # log = 'Fold: %d, final train_loss: %0.4f, best val_loss: %0.4f, test_acc: %0.4f' % (
        #     fold, eval_info["train_loss"], loss, acc
        # )
        # print(log)
        # if logger is not None:
        #     logger(log)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)
    # loss, acc = loss.view(folds, epochs), acc.view(folds, epochs)
    loss, acc = loss.view(1, epochs), acc.view(1, epochs)
    loss, argmin = loss.min(dim=1)
    # acc = acc[torch.arange(folds, dtype=torch.long), argmin]
    acc = acc[torch.arange(1, dtype=torch.long), argmin]
    #average_train_loss = float(np.mean(final_train_losses))
    #std_train_loss = float(np.std(final_train_losses))

    # #DeleteLog
    # log = 'Val Loss: {:.4f}, Test Accuracy: {:.3f} ± {:.3f}, Duration: {:.3f}'.format(
    #     loss.mean().item(),
    #     acc.mean().item(),
    #     acc.std().item(),
    #     duration.mean().item()
    # ) #+ ', Avg Train Loss: {:.4f}'.format(average_train_loss)
    # print(log)
    # if logger is not None:
    #     logger(log)

    return loss.mean().item(), acc.mean().item(), acc.std().item()


def cross_validation_without_val_set( dataset,
                                      model,
                                      folds,
                                      epochs,
                                      batch_size,
                                      lr,
                                      lr_decay_factor,
                                      lr_decay_step_size,
                                      weight_decay,
                                      device, 
                                      logger=None):

    test_losses, accs, durations = [], [], []
    count = 1
    # for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(dataset, folds))):
    for fold in [0]:
        print("CV fold " + str(count))
        count += 1

        train_idx = 30
        # train_idx = torch.cat([train_idx, val_idx], 0)  # combine train and val
        train_dataset = dataset[:train_idx]
        test_dataset = dataset[train_idx:]

        if 'adj' in train_dataset[0]:
            train_loader = DenseLoader(train_dataset, batch_size, shuffle=True)
            test_loader = DenseLoader(test_dataset, batch_size, shuffle=False)
        else:
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        pbar = tqdm(range(1, epochs + 1), ncols=70)

        for epoch in pbar:
            train_loss = train(model, optimizer, train_loader, device)
            test_losses.append(eval_loss(model, test_loader, device))
            accs.append(eval_acc(model, test_loader, device))
            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'test_loss': test_losses[-1],
                'test_acc': accs[-1],
            }
            log = 'Fold: %d, train_loss: %0.4f, test_loss: %0.4f, test_acc: %0.4f' % (
                fold, eval_info["train_loss"], eval_info["test_loss"], eval_info["test_acc"]
            )
            pbar.set_description(log)

            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']

        if logger is not None:
            logger(log)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    loss, acc, duration = tensor(test_losses), tensor(accs), tensor(durations)
    loss, acc = loss.view(folds, epochs), acc.view(folds, epochs)
    acc_mean = acc.mean(0)
    acc_max, argmax = acc_mean.max(dim=0)
    acc_final = acc_mean[-1]

    log = ('Test Loss: {:.4f}, Test Max Accuracy: {:.3f} ± {:.3f}, ' + 
          'Test Final Accuracy: {:.3f} ± {:.3f}, Duration: {:.3f}').format(
        loss.mean().item(),
        acc_max.item(),
        acc[:, argmax].std().item(),
        acc_final.item(),
        acc[:, -1].std().item(),
        duration.mean().item()
    )
    print(log)
    if logger is not None:
        logger(log)

    #return loss.mean().item(), acc_final.item(), acc[:, -1].std().item()
    return loss.mean().item(), acc_max.item(), acc[:, argmax].std().item()


def k_fold(dataset, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y[dataset.indices()]):
        test_indices.append(torch.from_numpy(idx))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero().view(-1))

    return train_indices, test_indices, val_indices


def k_fold2(dataset, folds):
    kf = KFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, test_idx in kf.split(dataset):
        test_indices.append(torch.from_numpy(test_idx))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero().view(-1))

    return train_indices, test_indices, val_indices


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train(args, model, optimizer, loader, device):
    model.train()

    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)

        # 适应原图
        data.subgraph_to_graph = data.batch
        data.x_ori = data.x
        data.original_edge_index = data.edge_index
        if hasattr(data, "edge_attr"):
            data.edge_type_ori = data.edge_attr
        if hasattr(data, "edge_tyep"):
            data.edge_type_ori = data.edge_type


        out = model(data)

        # compute loss
        if args.task in ["HA", "FA"]:
            label = data[args.task].unsqueeze(-1)
            loss = F.l1_loss(out, label, reduction="mean")
            # mape = mape_loss(graph_pred, label)
        elif args.task == "type":
            label = data[args.task]
            loss = F.nll_loss(out, label)
            acc = (torch.argmax(out, dim=1)==label).sum()/label.shape[0]
            mape = acc    # 暂时用mape 表示acc
        elif args.task in ["0", "1", "2", "3", "4", "5", "6"]:
            label = data["node_label"][:, int(args.task)]

            # weight = torch.tensor([0.1, 1.0]).to(label.device) 

            loss = F.nll_loss(out, label)
            label = label[data.x_ori[:, 0]==1]
            acc = (torch.argmax(out, dim=1)[data.x_ori[:, 0]==1]==label).sum()/label.shape[0]
            mape = acc    # 暂时用mape 表示acc
        else:
            print(f"task {args.task} dosen't support\n. please choise one from:HA, FA, type, 0 ~6")
            exit()

        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        total_loss += loss.item()
        optimizer.step()
    return total_loss / len(loader.dataset)


def eval_acc(model, loader, device):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


def eval_loss(args, model, loader, device, eval_type = "val"):
    model.eval()

    losses = 0
    correct = 0
    total_num = 0

    correct_0 = 0
    correct_1 = 0
    total_num_0 = 0
    total_num_1 = 0

    fn, fp, tn, tp = 0, 0, 0, 0,

    for data in loader if eval_type == "val" else tqdm(loader):
        data = data.to(device)

        # 适应原图
        data.subgraph_to_graph = data.batch
        data.x_ori = data.x
        data.original_edge_index = data.edge_index
        if hasattr(data, "edge_tyep"):
            data.edge_type_ori = data.edge_type
        

        with torch.no_grad():
            out = model(data)

            # compute loss
            if args.task in ["HA", "FA"]:
                label = data[args.task].unsqueeze(-1)
                loss = F.l1_loss(out, label, reduction="mean")
                correct += 1
                # mape = mape_loss(graph_pred, label)
            elif args.task == "type":
                label = data[args.task]
                loss = F.nll_loss(out, label)
    
                pred = out.max(1)[1]
                acc = (torch.argmax(out, dim=1)==label).sum()/label.shape[0]
                correct += pred.eq(data[args.task].view(-1)).sum().item()
                # mape = acc    # 暂时用mape 表示acc
            elif args.task in ["0", "1", "2", "3", "4", "5", "6"]:
                label = data["node_label"][:, int(args.task)]
                loss = F.nll_loss(out, label)
                # acc = (torch.argmax(out, dim=1)==label).sum()/label.shape[0]
                label = label[data.x_ori[:, 2]!=1]
                pred = out.max(1)[1][data.x_ori[:, 2]!=1]
                # acc = (torch.argmax(out, dim=1)==label).sum()/label.shape[0]
                correct += pred.eq(label.view(-1)).sum().item()
                total_num += pred.shape[0]

                correct_0 += (pred.eq(label) & (label == 0)).sum()
                correct_1 += (pred.eq(label) & (label == 1)).sum()

                total_num_0 += (label==0).sum()
                total_num_1 += (label==1).sum()
                # mape = acc    # 暂时用mape 表示acc

                # 计算recall and F1_score
                fn += ((pred == 0) & (label != 0)).sum().item()
                tp += ((pred != 0) & (label != 0)).sum().item()
                tn += ((pred == 0) & (label == 0)).sum().item()
                fp += ((pred != 0) & (label == 0)).sum().item()

            else:
                print(f"task {args.task} dosen't support\n. please choise one from:HA, FA, type, 0 ~6")
                exit()

        # calculate recall, precision and F1-score
        recall = 0
        precision = 0
        if tp != 0:
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
        F1_score = 0
        if precision != 0 or recall != 0:
            F1_score = 2 * recall * precision / (recall + precision)

        losses += loss * num_graphs(data)
        total = len(loader.dataset)
        if args.task in ["0", "1", "2", "3", "4", "5", "6"]:
            total = total_num
    return losses / len(loader.dataset), correct / total, correct_0 / total_num_0, correct_1 / total_num_1, recall, F1_score
