
import scipy.io as sio
import numpy as np
import time
import os
import torch
from torch import optim
import torch.backends.cudnn as cudnn
from UTILS import setup_seed,get_args,load_dataset,write2txt
from UTILS import pre_process, adjust_learning_rate
from UTILS import initNetParams,recover_split_img,access,get_avg_oa_kappa
from model import GlobalMind

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
torch.set_num_threads(2)


def train(args):
    setup_seed(args.seed)
    T11, T22, gt = load_dataset(args.Dataset_path, args.Dataset)
    data1, data2, idx, binary_label, train_Loader = pre_process(args, T11, T22,gt,
                                                                args.ChangeSamle_num,args.UncangeSample_num)
    model = GlobalMind(args)
    model.apply(initNetParams)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)
    cudnn.benchmark = True
    model.cuda()
    print('trainging begins----------------------------')
    loss_fuc = torch.nn.CrossEntropyLoss()
    torch.cuda.empty_cache()
    train_loss_list = []
    t0=time.time()
    for epoch in range(args.epochs):  # args.epochs
        model.train()
        adjust_learning_rate(optimizer, args.lr, epoch, args.epochs)
        loss_epoch = 0
        for step, (data1_, data2_, idx_, binary_label_) in enumerate(train_Loader):
            idx_true = torch.nonzero(idx_.squeeze()>-1, as_tuple=False).squeeze()
            idx_ = idx_.squeeze()[idx_true]
            binary_label_ = binary_label_.squeeze()[idx_true]
            l = model(data1_.contiguous(), data2_.contiguous(), idx_, binary_label_, loss_fuc)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            loss_epoch += l.item()
        l = loss_epoch/args.data_split_num
        train_loss_list.append(l)
        if epoch % 30 == 0:
            print('epoch %d, Loss: %.6f' % (epoch + 1, l))
        del l
    del train_Loader, data1_, data2_, idx_, binary_label_
    t1 = time.time()
    time_epoch = (t1 - t0) / args.epochs
    print(args.Dataset)
    print('----1.time of train', time_epoch)


    t0 = time.time()
    with torch.no_grad():
        model.eval()
        result1, result2 = model(data1, data2, [], [], [])

        if args.data_split_num>1:
            result1 = torch.transpose(result1, 1, 0)  # num_class, num_split, H,W
            result1 = recover_split_img(result1)  # [num_class,H,W]
        result1 = result1.reshape([2, -1])  # [2, H*W]
        result1 = np.array(result1.cpu().argmax(axis=0).long())
        bmap1 = result1.reshape([args.H, args.W])
        oa_kappa1 = access(args, gt, bmap1)

        if args.data_split_num > 1:
            result2 = torch.transpose(result2, 1, 0)  # num_class, num_split, H,W
            result2 = recover_split_img(result2)  # [num_class,H,W]
        result2 = result2.reshape([2, -1])  # [2, H*W]
        result2 = np.array(result2.cpu().argmax(axis=0).long())
        bmap2 = result2.reshape([args.H, args.W])
        oa_kappa2 = access(args, gt, bmap2)

    t1 = time.time()
    time_epoch = t1 - t0
    print(args.Dataset)
    print('----2. time of test: ', time_epoch)
    return bmap1, oa_kappa1

def repeat_runs(num_run, Dataset_name, Dataset_path, save_path):

    repeat_BMP1, repeat_OA_KAPPA1= [], []
    oa, kappa, f1,recall,precision =0,0,0,0,0
    seed = np.arange(num_run)
    for i in np.arange(num_run):
        seed_i = seed[i]
        setup_seed(seed_i)
        args = get_args(seed_i, Dataset_name,Dataset_path)
        bmap1, oa_kappa1 = train(args)
        repeat_BMP1.append(bmap1)
        repeat_OA_KAPPA1.append(oa_kappa1)

        oa = oa + repeat_OA_KAPPA1[i][1]
        kappa = kappa + repeat_OA_KAPPA1[i][3]
        f1 = f1 + repeat_OA_KAPPA1[i][5]
        recall = recall + repeat_OA_KAPPA1[i][7]
        precision = precision + repeat_OA_KAPPA1[i][9]
    avg_oa_kappa1 = get_avg_oa_kappa(oa, kappa, f1, recall, precision, seed)
    repeat_BMP1= np.array(repeat_BMP1)
    repeat_OA_KAPPA1 = np.array(repeat_OA_KAPPA1)

    result_file = save_path + '/' + Dataset_name + args.model_name +'_runs'+ str(num_run)+'_result.mat'
    print(result_file)
    sio.savemat(result_file, {'repeat_bmap':repeat_BMP1, 'repeat_oa_kappa': repeat_OA_KAPPA1})


    filename = save_path + '/' + Dataset_name + args.model_name +'_runs'+ str(num_run) + '_oa_kappa.txt'
    print(filename)
    write2txt(filename, args, repeat_OA_KAPPA1, avg_oa_kappa1)
    print('Dataset_name: ', Dataset_name)
    print('num_runs: ', num_run)
    print('------------------Average results-----------------')
    print('OA:  ' + str(oa) + '    ' + 'kappa:  ' + str(kappa))
    print('F1=   ' + str(f1))
    print('recall=   ' + str(recall))
    print('precision=   ' + str(precision))



if __name__=='__main__':
    # please download the hyperspectral change detection datasets, and set the "Dataset_path"

    Dataset_path = '' # the path you put the dataset
    num_run = 10
    save_path = os.path.dirname(os.path.abspath(__file__))+'/result/'
    os.makedirs(save_path, exist_ok=True)
    # Dataset_list = ['Farmland','Hermiston','River','Bay','Barbara','GF5B_BI']
    Dataset_list = ['Hermiston']
    for Dataset_name in Dataset_list:
        repeat_runs(num_run, Dataset_name, save_path)

