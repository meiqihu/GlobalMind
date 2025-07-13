import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import math
import torch.nn as nn
import torch
import random
import argparse
from sklearn import metrics, preprocessing
from torch.utils.data import DataLoader
import torch.utils.data as Data


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

def get_sample(dataset):
    layernum = []
    C,H, W = 0,0,0
    # feature_dim = 128
    data_split_num = 1 # for LARGE DATASET like Bay, Barbara, GF5B_BI, data is firstly split into several pieces
    # for dataset,
    # 'GRS-GRS' is recommended if the height >> width, such as Hermiston(307, 241), Farmland(450, 140), River(463, 241),
    # 'GCS-GRS' is recommended if the height ~= width, such as Bay(600, 500), Barbara(984, 740),GF5B_BI(463, 241)
    GAS_mode = 'GRS-GRS'
    if dataset == 'Hermiston':
        C, H, W = 154, 307, 241
        layernum = [C, 128, 128, 128, 128,128, 128]   # February, MM1
        data_split_num = 1
        GAS_mode = 'GRS-GRS'
    elif dataset == 'Farmland':
        C, H, W = 155, 450, 140
        layernum = [C, 128, 128, 128, 128, 128, 128]  # February, MM1
        data_split_num = 1
        GAS_mode = 'GRS-GRS'
    elif dataset == 'River':
        C, H, W = 198, 463, 241
        layernum = [C, 128, 128, 128, 128, 128, 128]  # February, MM1
        data_split_num = 1
        GAS_mode = 'GRS-GRS'
    elif dataset == 'Bay':
        C, H, W = 224, 600, 500
        layernum = [C, 128, 128, 128, 128, 128, 128]  # February, MM1;[224, 180, 180, 180, 180, 180, 180]
        data_split_num = 2
        GAS_mode = 'GCS-GRS'
    elif dataset == 'Barbara':
        C, H, W = 224, 984, 740
        layernum = [C, 128, 128, 128, 128, 128, 128]  # February, MM1
        data_split_num = 4
        GAS_mode = 'GCS-GRS'
    elif dataset == 'GF5B_BI':
        C, H, W = 150, 512, 512
        layernum = [C, 128, 128, 128, 128, 128, 128]  # February, MM1
        data_split_num = 4
        GAS_mode = 'GCS-GRS'
    else:
        raise ValueError("Dataset should be one of the {Hermiston, Farmland, River, Bay, Barbara, GF5B_BI}.")
    return data_split_num,GAS_mode, layernum, C, H, W


def get_args(seed, Dataset, Dataset_path):
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--Dataset', default=Dataset,
                        type=str, help='path filename of training data')
    parser.add_argument('--Dataset_path', default=Dataset_path,
                        type=str, help='path filename of training data')
    parser.add_argument('--model_name', default='GlobalMind',
                        type=str, help='path filename of training data')
    data_split_num,GAS_mode, layernum,C,H, W = get_sample(Dataset)
    parser.add_argument('--layernum', default=layernum, type=list, nargs='+', help='layernum')
    parser.add_argument('--seed', default=seed, type=int, metavar='seed', help='seed for randn seed')
    parser.add_argument('--C', default=C, type=int, metavar='C', help='C')
    parser.add_argument('--H', default=H, type=int, metavar='H', help='H')
    parser.add_argument('--W', default=W, type=int, metavar='W', help='W')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='defalut: number of total epochs to run')
    parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
    parser.add_argument('--data_split_num', default=data_split_num, type=int, metavar='N', help='')
    parser.add_argument('--GAS_mode', default=GAS_mode, type=str, metavar='N', help='')
    pixel_num = int(H * W / data_split_num)
    parser.add_argument('--pixel_num', default=pixel_num, type=int, metavar='N', help='')
    if Dataset =='GF5B_BI':
        parser.add_argument('--ChangeSamle_num', default=1000, type=int, metavar='N', help='')
        parser.add_argument('--UncangeSample_num', default=1000, type=int, metavar='N', help='')
    else:
        parser.add_argument('--ChangeSamle_num', default=500, type=int, metavar='N', help='')
        parser.add_argument('--UncangeSample_num', default=500, type=int, metavar='N', help='')

    parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float, metavar='LR',
                        help='initial (base) learning rate', dest='lr')
    args = parser.parse_args()
    print('Dataset:', args.Dataset)
    print('epochs:', args.epochs)
    print('lr:  ', args.lr)
    return args

def load_dataset(data_path, dataset):
    TT1, TT2, gt_hsi, TOTAL_SIZE, TRAIN_SIZE, VALIDATION_SPLIT=0,0,0,0,0,0
    if dataset=='Bay':
        data = sio.loadmat(data_path +'BayArea.mat')
        TT1 = data['T1']
        TT2 = data['T2']
        gt_hsi = data['GT']  # 变化像元的数值为1，未变化像元的数值为2，不确定像元的数值为0
    elif dataset == 'Barbara':
        data = sio.loadmat(data_path +'Barbara.mat')
        TT1 = data['T1']
        TT2 = data['T2']
        gt_hsi = data['GT']  # 变化像元的数值为1，未变化像元的数值为2，不确定像元的数值为0
    elif dataset == 'GF5B_BI':
        data = sio.loadmat(data_path + 'GF5B_BI2.mat')
        TT1 = data['T1']
        TT2 = data['T2']
        gt_hsi = data['GT']# 变化像元的数值为1，未变化像元的数值为0
    elif dataset=='River':
        data = sio.loadmat(data_path + 'River.mat')
        TT1 = data['T1']
        TT2 = data['T2']
        gt_hsi = data['GT']  # 变化像元的数值为1，未变化像元的数值为0
    elif dataset=='Farmland':
        data = sio.loadmat(data_path + 'Farmland.mat')
        TT1 = data['T1']
        TT2 = data['T2']
        gt_hsi = data['GT']  # 变化像元的数值为1，未变化像元的数值为0

    elif dataset == 'Hermiston':
        data = sio.loadmat(data_path + 'Hermiston.mat')
        TT1 = data['T1']
        TT2 = data['T2']
        gt_hsi = data['GT']  # 变化像元的数值为1，未变化像元的数值为0


    return TT1, TT2, gt_hsi

def pre_process(args, T11,T22,gt,ChangeSamle_num=500, UncangeSample_num=500):
    H, W, C = T11.shape
    if args.Dataset in ['River']:
        data1 = np.zeros(T11.shape)
        data2 = np.zeros(T11.shape)
        for i in range(T11.shape[2]):
            input_max = max(np.max(T11[:, :, i]), np.max(T22[:, :, i]))
            input_min = min(np.min(T11[:, :, i]), np.min(T22[:, :, i]))
            data1[:, :, i] = (T11[:, :, i] - input_min) / (input_max - input_min)
            data2[:, :, i] = (T22[:, :, i] - input_min) / (input_max - input_min)
    else:
        T11 = np.reshape(T11, [H * W, C])
        T22 = np.reshape(T22, [H * W, C])

        data1 = preprocessing.scale(T11, axis=0)  # each channel
        data2 = preprocessing.scale(T22, axis=0)
        data1 = data1.reshape(H, W, C)
        data2 = data2.reshape(H, W, C)

    if args.data_split_num >1:
        data1 = splitImg(data1, args.data_split_num)  # [num_split, h, w, C]
        data2 = splitImg(data2, args.data_split_num)
        data1 = torch.tensor(data1, dtype=torch.float32).cuda()  # [num_split, C, H,W]
        data2 = torch.tensor(data2, dtype=torch.float32).cuda()
        data1 = data1.permute(0, 3, 1, 2)
        data2 = data2.permute(0, 3, 1, 2)

        idx, binary_label = select_sample_frmGT(args, gt, ChangeSamle_num, UncangeSample_num)
        idx_c_u_split, label_c_u_split = split_idx_label(idx, binary_label, args.data_split_num, H, W)
        idx_new = torch.tensor(idx_c_u_split, dtype=torch.long).cuda()
        binary_label = torch.from_numpy(binary_label).squeeze().cuda()  # [num_split, n]
        binary_label_new = torch.tensor(label_c_u_split.squeeze(), dtype=torch.long).cuda()
        torch_dataset = Data.TensorDataset(data1, data2, idx_new, binary_label_new)
        train_Loader = DataLoader(dataset=torch_dataset, batch_size=1, shuffle=True)

    else:
        data1 = torch.tensor(np.transpose(data1, [2, 0, 1]), dtype=torch.float32).unsqueeze(0).cuda()
        data2 = torch.tensor(np.transpose(data2, [2, 0, 1]), dtype=torch.float32).unsqueeze(0).cuda()

        idx, binary_label = select_sample_frmGT(args, gt, ChangeSamle_num, UncangeSample_num)
        idx = torch.tensor(idx).unsqueeze(0).cuda()
        binary_label = torch.tensor(binary_label, dtype=torch.long).unsqueeze(0).cuda()

        torch_dataset = Data.TensorDataset(data1, data2, idx, binary_label)
        train_Loader = DataLoader(dataset=torch_dataset, batch_size=1, shuffle=True)
    # return data1,data2, train_Loader
    return data1, data2, idx, binary_label, train_Loader
def select_sample_frmGT(args, GT, change_num, Uchange_num):
    setup_seed(0) # defalut

    print('GT.shape:', GT.shape)
    GT = np.reshape(GT, [-1, 1]).squeeze()
    if args.Dataset in ['Bay' ,'Barbara']:
        value_c, value_uc = 1, 2
    elif args.Dataset in ['Farmland','Hermiston','River','GF5B_BI']: # change,unchange=2,1
        value_c, value_uc = 1, 0
    else:
        raise ValueError("Unknow dataset")

    uc_position = np.array(np.where(GT == value_uc)).transpose(1, 0)
    c_position = np.array(np.where(GT == value_c)).transpose(1, 0)
    selected_uc = np.random.choice(uc_position.shape[0], int(Uchange_num), replace = False)
    selected_c = np.random.choice(c_position.shape[0], int(change_num), replace = False)
    selected_uc_position = uc_position[selected_uc]  # (500,),adarray
    selected_c_position = c_position[selected_c]
    idx = np.concatenate((selected_c_position.squeeze(), selected_uc_position.squeeze()), axis=0)
    label = np.concatenate((np.ones([1, change_num], dtype=int), np.zeros([1, Uchange_num])), axis=1)
    label = label.squeeze()
    # print('selected change num ', str(change_num))
    # print('selected Unchange num ', str(Uchange_num))
    return idx,label

def splitImg(img, num):
    # idx: index according to python , 1D[list]
    # H, W, C = img.shape
    H , W = img.shape[0], img.shape[1]
    img_split = []
    if num ==2:
        img_split = np.split(img,2,axis=0) #img_split1:[H/2, W, C]
        img_split = np.asarray(img_split)
    elif num==4:
        img_split1, img_split2 = np.split(img, 2, axis=0)  # img_split1:[H/2, W, C]
        img_split1 = np.split(img_split1, 2, axis=1)  # img_split11:[H/2, W/2, C]
        img_split2 = np.split(img_split2, 2, axis=1)  # img_split11:[H/2, W/2, C]
        img_split = np.concatenate([img_split1, img_split2], axis=0)
    else:
        print(' the num of split parts should be even number and equal to 2 or 4')
    return img_split
def recover_split_img(img_split):
    # img_split,the binary change result,[2, num_split, h,w]
    img_recover =[]
    num_class,num_split, h,w = img_split.shape
    if num_split==2:  # [num_class,num_split, H/2,W]
        img_recover = img_split.reshape([num_class, num_split*h,w]) # [num_class,H,W]
    elif num_split==4: # [num_class,num_split, H/2,W/2]
        img_recover1 = torch.cat((img_split[:,0, :,:], img_split[:,1, :,:]), dim=-1) # [num_class,H/2,W]
        img_recover2 = torch.cat((img_split[:, 2, :, :], img_split[:, 3, :, :]), dim=-1)  # [num_class,H/2,W]
        img_recover = torch.cat((img_recover1,img_recover2), dim=-2)  # [num_class,H,W]
    else:
        print(' the num of split parts should be even number and equal to 2 or 4')
    return img_recover  # [num_class,H,W]
# for bay and barbara dataset
def split_idx_label(idx, binary_label, num, H,W):
    # idx:and the corresponding training idx is split
    # num: split the img and corresponding idx into num parts
    idx_c_u_show_split = []
    label_c_u_split = []
    idx_c = np.where(binary_label == 1)[0]
    idx_c = idx[idx_c]
    idx_u = np.where(binary_label == 0)[0]
    idx_u = idx[idx_u]

    # for changed training sample
    idx_show = np.zeros([H * W, 1])
    idx_show[idx_c] = 1
    idx_show[idx_u] = -1
    idx_show = np.reshape(idx_show, [H,W])
    if num ==2 or 4:
        tmp_idx = []
        tmp_label = []
        # idx_show_split = np.split(idx_show, 4, axis=0)  # img_split1:[H/2, W/2]
        idx_show_split = splitImg(idx_show, num) # img_split1:[H/2, W/2]
        idx_show_split = idx_show_split.reshape([num, -1, 1])
        num_size = []
        for i in range(num):
            idx_show_split_i = idx_show_split[i]
            idx_c_show_split_i = np.where(idx_show_split_i == 1)[0]
            idx_u_show_split_i = np.where(idx_show_split_i == -1)[0]
            # idx_c_u_show_split_i = np.stack([idx_c_show_split_i, idx_u_show_split_i], axis=0)
            idx_c_u_show_split_i = np.concatenate([idx_c_show_split_i, idx_u_show_split_i], axis=0)
            tmp_idx.append(idx_c_u_show_split_i)

            label_c_i = np.ones([idx_c_show_split_i.size,])  # label of change is 1
            label_u_i = np.zeros([idx_u_show_split_i.size, ]) # label of unchange is 0
            label_c_u_i = np.concatenate([label_c_i, label_u_i], axis=0)
            tmp_label.append(label_c_u_i)

            num_i = idx_c_u_show_split_i.size
            num_size.append(num_i)
        num_size_np = np.asarray(num_size)
        num_max = np.max(num_size_np)
        # new_idx_c_u_show_split = []
        for i in range(num):
            if num_size[i]<num_max:
                diff = num_max - num_size[i]
                diff = -1 * np.ones([diff, ])
                idx_c_u_show_split_i = tmp_idx[i]
                idx_c_u_show_split_i = np.concatenate([idx_c_u_show_split_i, diff], axis=0)
                idx_c_u_show_split.append(idx_c_u_show_split_i)

                label_c_u_i = tmp_label[i]
                label_c_u_i = np.concatenate([label_c_u_i, diff], axis=0)
                label_c_u_split.append(label_c_u_i)
            else:
                idx_c_u_show_split_i = tmp_idx[i]
                idx_c_u_show_split.append(idx_c_u_show_split_i)

                label_c_u_i = tmp_label[i]
                label_c_u_split.append(label_c_u_i)
                # idx_c_u_show_split[i] = np.concatenate([idx_c_u_show_split, diff], axis=0)
    else:
        print(' the num of split parts should be even number and equal to 2 or 4')
    idx_c_u_show_split = np.asarray(idx_c_u_show_split)
    label_c_u_split = np.asarray(label_c_u_split)
    return idx_c_u_show_split,label_c_u_split


def initNetParams(net):
    '''Init net parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight.data)
            # init.xavier_uniform(m.weight)
            if m.bias:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)      # m.weight.data.normal_(0, 0.001)
            # init.xavier_uniform(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
def adjust_learning_rate(optimizer, init_lr, epoch, epochs):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr

def access(args,gt,bmap):
    if args.Dataset in ['Bay','Barbara']:
        oa_kappa = two_cls_access_for_Bay_Barbara(gt, bmap)
    elif args.Dataset in ['Farmland','Hermiston','River','GF5B_BI']: # change,unchange=2,1
        oa_kappa = two_cls_access(gt, bmap)
    else:
        raise ValueError("Unknow dataset")
    return oa_kappa
def two_cls_access(reference,result):
    # 对二类变化检测的结果进行精度评价，指标为kappad系数和OA值
    # 输入：
    #      reference：二元变化reference(二值图，H*W)
    #      resultz:算法检测得到的二类变化结果图(二值图，H*W)]
    oa_kappa = []
    m,n = reference.shape
    if reference.shape != result.shape:
        print('the size of reference shoulf be equal to that of result')
        return oa_kappa
    reference = np.reshape(reference, -1)
    result = np.reshape(result, -1)
    label_0 = np.where(reference == 0)
    label_1 = np.where(reference == 1)
    predict_0 = np.where(result == 0)
    predict_1 = np.where(result == 1)
    label_0 = label_0[0]
    label_1 = label_1[0]
    predict_0 = predict_0[0]
    predict_1 = predict_1[0]
    tp = set(label_1).intersection(set(predict_1))  # True Positive
    tn = set(label_0).intersection(set(predict_0))  # False Positive
    fp = set(label_0).intersection(set(predict_1))  # False Positive
    fn = set(label_1).intersection(set(predict_0))  # False Negative

    precision = len(tp) / (len(tp) + len(fp))
    recall = len(tp) / (len(tp) + len(fn))

    precision = round(precision, 4)
    recall = round(recall, 4)
    F1 = 2 * (precision * recall) / (precision + recall)
    F1 = round(F1, 4)
    print('F1=   ' + str(F1))
    print('recall=   ' + str(recall))
    print('precision=   ' + str(precision))

    oa = (len(tp)+len(tn))/m/n      # Overall precision
    pe = (len(label_1)*len(predict_1)+len(label_0)*len(predict_0))/m/n/m/n
    kappa = (oa-pe)/(1-pe)
    oa = round(oa, 4)
    kappa = round(kappa, 4)
    oa_kappa.append('OA')
    oa_kappa.append(oa)
    oa_kappa.append('kappa')
    oa_kappa.append(kappa)
    oa_kappa.append('F1')
    oa_kappa.append(F1)
    oa_kappa.append('recall')
    oa_kappa.append(recall)
    oa_kappa.append('precision')
    oa_kappa.append(precision)

    print('OA:  ' + str(oa) + '    ' + 'kappa:  ' + str(kappa))
    return oa_kappa
def two_cls_access_for_Bay_Barbara(reference,result):
    # 对二类变化检测的结果进行精度评价，指标为kappad系数和OA值
    # 输入：
    #      reference：二元变化reference(二值图，H*W), change=1; unchanged=2;uncertain=0
    #      resultz:算法检测得到的二类变化结果图(二值图，H*W)]
    oa_kappa = []
    # m,n = reference.shape
    if reference.shape != result.shape:
        print('the size of reference shoulf be equal to that of result')
        return oa_kappa
    reference = np.reshape(reference, -1)
    result = np.reshape(result, -1)

    label_0 = np.where(reference == 2)  # Unchanged
    label_1 = np.where(reference == 1)  # Changed
    predict_0 = np.where(result == 0)  # Unchanged
    predict_1 = np.where(result == 1)  # Changed
    label_0 = label_0[0]
    label_1 = label_1[0]
    predict_0 = predict_0[0]
    predict_1 = predict_1[0]
    tp = set(label_1).intersection(set(predict_1))  # True Positive
    tn = set(label_0).intersection(set(predict_0))  # True Negative
    fp = set(label_0).intersection(set(predict_1))  # False Positive
    fn = set(label_1).intersection(set(predict_0))  # False Negative

    precision = len(tp) / (len(tp) + len(fp))  # (预测为1且正确预测的样本数) / (所有真实情况为1的样本数)
    recall = len(tp) / (len(tp) + len(fn))  # (预测为1且正确预测的样本数) / (所有真实情况为1的样本数)

    precision = round(precision, 4)
    recall = round(recall, 4)
    F1 = 2 * (precision * recall) / (precision + recall)
    F1 = round(F1, 4)
    print('F1=   ' + str(F1))
    print('recall=   ' + str(recall))
    print('precision=   ' + str(precision))
    total_num = len(label_0) +len(label_1)
    oa = (len(tp) + len(tn)) / total_num  # Overall precision
    pe = ((len(tp)+len(fn))*(len(tp)+len(fp)) +(len(fp)+len(tn))*(len(fn)+len(tn)))/ total_num / total_num

    kappa = (oa-pe)/(1-pe)
    oa = round(oa, 4)
    kappa = round(kappa, 4)
    oa_kappa.append('OA')
    oa_kappa.append(oa)
    oa_kappa.append('kappa')
    oa_kappa.append(kappa)
    oa_kappa.append('F1')
    oa_kappa.append(F1)
    oa_kappa.append('recall')
    oa_kappa.append(recall)
    oa_kappa.append('precision')
    oa_kappa.append(precision)

    print('OA:  ' + str(oa) + '    ' + 'kappa:  ' + str(kappa))
    # print('whole OA is' + str(oa))
    # print('whole kappa is' + str(kappa))
    return oa_kappa
def get_avg_oa_kappa(oa, kappa, f1, recall, precision, seed):
    num_repeat = len(seed)
    oa = oa/len(seed)
    kappa = kappa / len(seed)
    f1 = f1 / len(seed)
    recall = recall / len(seed)
    precision = precision / len(seed)

    oa = round(oa, 4)
    kappa = round(kappa, 4)
    precision = round(precision, 4)
    recall = round(recall, 4)
    f1 = round(f1, 4)
    oa_kappa = []
    oa_kappa.append('OA')
    oa_kappa.append(oa)
    oa_kappa.append('kappa')
    oa_kappa.append(kappa)
    oa_kappa.append('F1')
    oa_kappa.append(f1)
    oa_kappa.append('recall')
    oa_kappa.append(recall)
    oa_kappa.append('precision')
    oa_kappa.append(precision)
    return oa_kappa

def write2txt(filename, args, repeat_OA_KAPPA,avg_oa_kappa):
    print('save filename:', '\n',filename)
    file3 = open(filename, 'w', encoding='UTF-8')
    file3.write(f'model_name:{args.model_name}\n')
    file3.write(f'dataset:{args.Dataset}\n')

    # avgerage oa_kappa
    file3.write('1. Average OA, Kappa, F1, Recall, Precision: ' + '\n')
    file3.write(str(avg_oa_kappa) + '\n')

    #  repeated  oa_kappa
    num_runs = repeat_OA_KAPPA.shape[0]
    file3.write('\n')
    file3.write('2. Repeated Experiments, OA, Kappa, F1, Recall, Precision:' + '\n')
    file3.write('num_runs:' +str(num_runs)+ '\n')
    file3.write(str(repeat_OA_KAPPA) + '\n')

    file3.close()
