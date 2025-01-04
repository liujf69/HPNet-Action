import torch
import pickle
import argparse
import numpy as np
import pandas as pd

def get_parser():
    parser = argparse.ArgumentParser(description = 'multi-stream ensemble') 
    parser.add_argument(
        '--gpu1_Score', 
        type = str,
        # default = '/data/liujinfu/TPAMI-24/TPAMI-24/HPNet/output/NTU120/XSub_1207/0_best_score.npy') # NTU120XSub
        # default = '/data/liujinfu/TPAMI-24/TPAMI-24/HPNet/output/NTU120/XSet_1220_best/0_best_score.npy') # NTU120XSet
        # default = '/data/liujinfu/TPAMI-24/TPAMI-24/HPNet/output/NTU60/XSub_1222/0_best_score.npy') # NTU60XSub
        default = '/data/liujinfu/TPAMI-24/TPAMI-24/HPNet/output/NTU60/XView_1223/0_best_score.npy') # NTU60XView
    parser.add_argument(
        '--gpu2_Score', 
        type = str,
        # default = '/data/liujinfu/TPAMI-24/TPAMI-24/HPNet/output/NTU120/XSub_1207/1_best_score.npy') # NTU120XSub
        # default = '/data/liujinfu/TPAMI-24/TPAMI-24/HPNet/output/NTU120/XSet_1220_best/1_best_score.npy') # NTU120XSet
        # default = '/data/liujinfu/TPAMI-24/TPAMI-24/HPNet/output/NTU60/XSub_1222/1_best_score.npy') # NTU60XSub
        default = '/data/liujinfu/TPAMI-24/TPAMI-24/HPNet/output/NTU60/XView_1223/1_best_score.npy') # NTU60XView
    parser.add_argument(
        '--gpu3_Score', 
        type = str,
        # default = '/data/liujinfu/TPAMI-24/TPAMI-24/HPNet/output/NTU120/XSub_1207/2_best_score.npy') # NTU120XSub
        # default = '/data/liujinfu/TPAMI-24/TPAMI-24/HPNet/output/NTU120/XSet_1220_best/2_best_score.npy') # NTU120XSet
        # default = '/data/liujinfu/TPAMI-24/TPAMI-24/HPNet/output/NTU60/XSub_1222/2_best_score.npy') # NTU60XSub
        default = '/data/liujinfu/TPAMI-24/TPAMI-24/HPNet/output/NTU60/XView_1223/2_best_score.npy') # NTU60XView
    parser.add_argument(
        '--gpu4_Score', 
        type = str,
        # default = '/data/liujinfu/TPAMI-24/TPAMI-24/HPNet/output/NTU120/XSub_1207/3_best_score.npy') # NTU120XSub
        # default = '/data/liujinfu/TPAMI-24/TPAMI-24/HPNet/output/NTU120/XSet_1220_best/3_best_score.npy') # NTU120XSet
        # default = '/data/liujinfu/TPAMI-24/TPAMI-24/HPNet/output/NTU60/XSub_1222/3_best_score.npy') # NTU60XSub
        default = '/data/liujinfu/TPAMI-24/TPAMI-24/HPNet/output/NTU60/XView_1223/3_best_score.npy') # NTU60XView
    parser.add_argument(
        '--gpu1_Name', 
        type = str,
        # default = '/data/liujinfu/TPAMI-24/TPAMI-24/HPNet/output/NTU120/XSub_1207/0_best_name.txt') # NTU120XSub
        # default = '/data/liujinfu/TPAMI-24/TPAMI-24/HPNet/output/NTU120/XSet_1220_best/0_best_name.txt') # NTU120XSet
        # default = '/data/liujinfu/TPAMI-24/TPAMI-24/HPNet/output/NTU60/XSub_1222/0_best_name.txt') # NTU60XSub
        default = '/data/liujinfu/TPAMI-24/TPAMI-24/HPNet/output/NTU60/XView_1223/0_best_name.txt') # NTU60XView
    parser.add_argument(
        '--gpu2_Name', 
        type = str,
        # default = '/data/liujinfu/TPAMI-24/TPAMI-24/HPNet/output/NTU120/XSub_1207/1_best_name.txt') # NTU120XSub
        # default = '/data/liujinfu/TPAMI-24/TPAMI-24/HPNet/output/NTU120/XSet_1220_best/1_best_name.txt') # NTU120XSet
        # default = '/data/liujinfu/TPAMI-24/TPAMI-24/HPNet/output/NTU60/XSub_1222/1_best_name.txt') # NTU60XSub
        default = '/data/liujinfu/TPAMI-24/TPAMI-24/HPNet/output/NTU60/XView_1223/1_best_name.txt') # NTU60XView
    parser.add_argument(
        '--gpu3_Name', 
        type = str,
        # default = '/data/liujinfu/TPAMI-24/TPAMI-24/HPNet/output/NTU120/XSub_1207/2_best_name.txt') # NTU120XSub
        # default = '/data/liujinfu/TPAMI-24/TPAMI-24/HPNet/output/NTU120/XSet_1220_best/2_best_name.txt') # NTU120XSet
        # default = '/data/liujinfu/TPAMI-24/TPAMI-24/HPNet/output/NTU60/XSub_1222/2_best_name.txt') # NTU60XSub
        default = '/data/liujinfu/TPAMI-24/TPAMI-24/HPNet/output/NTU60/XView_1223/2_best_name.txt') # NTU60XView
    parser.add_argument(
        '--gpu4_Name', 
        type = str,
        # default = '/data/liujinfu/TPAMI-24/TPAMI-24/HPNet/output/NTU120/XSub_1207/3_best_name.txt') # NTU120XSub
        # default = '/data/liujinfu/TPAMI-24/TPAMI-24/HPNet/output/NTU120/XSet_1220_best/3_best_name.txt') # NTU120XSet
        # default = '/data/liujinfu/TPAMI-24/TPAMI-24/HPNet/output/NTU60/XSub_1222/3_best_name.txt') # NTU60XSub
        default = '/data/liujinfu/TPAMI-24/TPAMI-24/HPNet/output/NTU60/XView_1223/3_best_name.txt') # NTU60XView
    parser.add_argument(
        '--J_Score', 
        type = str,
        # default = '/data/liujinfu/TPAMI-24/TPAMI-24/GCN/work_dir/ntu120/XSub/joint_pooling_0424/epoch1_test_score.pkl') # NTU120XSub H
        # default = '/data/liujinfu/TPAMI-24/TPAMI-24/GCN/work_dir/ntu120/XSet/joint_pooling_0505/epoch1_test_score.pkl') # NTU120XSet H
        # default = '/data/liujinfu/TPAMI-24/TPAMI-24/GCN/work_dir/ntu60/NTU60_XSub/joint_pooling_0508/epoch1_test_score.pkl') # NTU60XSub H
        default = '/data/liujinfu/TPAMI-24/TPAMI-24/GCN/work_dir/ntu60/NTU60_XView/joint_pooling_0509/epoch1_test_score.pkl') # NTU60XView H
    parser.add_argument(
        '--B_Score', 
        type = str,
        # default = '/data/liujinfu/TPAMI-24/TPAMI-24/GCN/work_dir/ntu120/XSub/bone_pooling_0424/epoch1_test_score.pkl') # NTU120XSub H
        # default = '/data/liujinfu/TPAMI-24/TPAMI-24/GCN/work_dir/ntu120/XSet/bone_pooling_0505/epoch1_test_score.pkl') # NTU120XSet H
        # default = '/data/liujinfu/TPAMI-24/TPAMI-24/GCN/work_dir/ntu60/NTU60_XSub/bone_pooling_0508/epoch1_test_score.pkl') # NTU60XSub H
        default = '/data/liujinfu/TPAMI-24/TPAMI-24/GCN/work_dir/ntu60/NTU60_XView/bone_pooling_0509/epoch1_test_score.pkl') # NTU60XView H
    parser.add_argument(
        '--JM_Score', 
        type = str,
        # default = '/data/liujinfu/TPAMI-24/TPAMI-24/GCN/work_dir/ntu120/XSub/jointmotion_pooling_0425/epoch1_test_score.pkl') # NTU120XSub H
        # default = '/data/liujinfu/TPAMI-24/TPAMI-24/GCN/work_dir/ntu120/XSet/jointmotion_pooling_0506/epoch1_test_score.pkl') # NTU120XSet H
        # default = '/data/liujinfu/TPAMI-24/TPAMI-24/GCN/work_dir/ntu60/NTU60_XSub/jointmotion_pooling_0508/epoch1_test_score.pkl') # NTU60XSub H
        default = '/data/liujinfu/TPAMI-24/TPAMI-24/GCN/work_dir/ntu60/NTU60_XView/jointmotion_pooling_0509/epoch1_test_score.pkl') # NTU60XView H
    parser.add_argument(
        '--BM_Score', 
        type = str,
        # default = '/data/liujinfu/TPAMI-24/TPAMI-24/GCN/work_dir/ntu120/XSub/bonemotion_pooling_0425/epoch1_test_score.pkl') # NTU120XSub H
        # default = '/data/liujinfu/TPAMI-24/TPAMI-24/GCN/work_dir/ntu120/XSet/bonemotion_pooling_0506/epoch1_test_score.pkl') # NTU120XSet H
        # default = '/data/liujinfu/TPAMI-24/TPAMI-24/GCN/work_dir/ntu60/NTU60_XSub/bonemotion_pooling_0508/epoch1_test_score.pkl') # NTU60XSub H
        default = '/data/liujinfu/TPAMI-24/TPAMI-24/GCN/work_dir/ntu60/NTU60_XView/bonemotion_pooling_0509/epoch1_test_score.pkl') # NTU60XView H
    parser.add_argument(
        '--val_sample', 
        type = str,
        # default = '/data/liujinfu/TPAMI-24/TPAMI-24/GCN/dataset/NTU120_XSub_val.txt') # NTU120XSub
        # default = '/data/liujinfu/TPAMI-24/TPAMI-24/GCN/dataset/NTU120_XSet_val.txt') # NTU120XSet
        # default = '/data/liujinfu/TPAMI-24/TPAMI-24/GCN/dataset/NTU60_XSub_val.txt') # NTU60XSub
        default = '/data/liujinfu/TPAMI-24/TPAMI-24/GCN/dataset/NTU60_XView_val.txt') # NTU60XView
    parser.add_argument(
        '--benchmark', 
        type = str,
        # default = 'NTU120XSub')
        # default = 'NTU120XSet')
        # default = 'NTU60XSub')
        default = 'NTU60XView')
    return parser

def Cal_Score(File, RGB_Score, Rate, ntu60XS_num, Numclass):
    final_score = torch.zeros(ntu60XS_num, Numclass)
    for idx in range(4):
        fr = open(File[idx],'rb') 
        inf = pickle.load(fr)
        df = pd.DataFrame(inf)
        df = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
        score = torch.tensor(data = df.values)
        final_score += Rate[idx] * score
    final_score += Rate[-1] * RGB_Score
    return final_score

def Cal_Acc(final_score, true_label):
    wrong_index = []
    _, predict_label = torch.max(final_score, 1)
    for index, p_label in enumerate(predict_label):
        if p_label != true_label[index]:
            wrong_index.append(index)
            
    wrong_num = np.array(wrong_index).shape[0]
    print('wrong_num: ', wrong_num)

    total_num = true_label.shape[0]
    print('total_num: ', total_num)
    Acc = (total_num - wrong_num) / total_num
    return Acc

def gen_label_ntu(val_txt_path):
    true_label = []
    val_txt = np.loadtxt(val_txt_path, dtype = str)
    for idx, name in enumerate(val_txt):
        label = int(name[-3:]) - 1
        true_label.append(label)

    true_label = torch.from_numpy(np.array(true_label))
    return true_label

def match_score(gpu1_Name, RGB_Score, val_txt_file):
    val_txt = np.loadtxt(val_txt_file, dtype = str)
    Score = torch.zeros_like(RGB_Score)
    for idx, name in enumerate(gpu1_Name):
        match_idx = np.where(val_txt == name)[0].item()
        Score[match_idx] = RGB_Score[idx]
    
    return Score

# Smarthome
def mean_class_accuracies(preds, labels, num_classes):
    """
    Computes the mean per-class accuracy over all classes

    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        num_classes (int): number of classes that can be predicted

    Returns:
        conf_matrix (torch.tensor): confidence matrix of predictions (rows are ground-truth class, columns are predicted class)
        class_accuracies (list): list of individual class accuracies
        mean_class_accuracy (float): the mean class accuracy
    """
    conf_matrix = torch.zeros((num_classes, num_classes))
    pred_classes = torch.argmax(preds, dim=1)

    for (gt_class, pred_class) in zip(labels, pred_classes):
      conf_matrix[int(gt_class.item()), int(pred_class.item())] += 1

    class_accuracies = conf_matrix.diag() / conf_matrix.sum(dim=1) # num_classes
    mean_perclass_accuracy = class_accuracies.mean()

    return conf_matrix, class_accuracies, mean_perclass_accuracy
    
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    gpu1_Score_file = args.gpu1_Score
    gpu2_Score_file = args.gpu2_Score
    gpu3_Score_file = args.gpu3_Score
    gpu4_Score_file = args.gpu4_Score
    gpu1_Name_file = args.gpu1_Name
    gpu2_Name_file = args.gpu2_Name
    gpu3_Name_file = args.gpu3_Name
    gpu4_Name_file = args.gpu4_Name

    gpu1_score = torch.from_numpy(np.load(gpu1_Score_file, allow_pickle = True))
    gpu2_score = torch.from_numpy(np.load(gpu2_Score_file, allow_pickle = True))
    gpu3_score = torch.from_numpy(np.load(gpu3_Score_file, allow_pickle = True))
    gpu4_score = torch.from_numpy(np.load(gpu4_Score_file, allow_pickle = True))
    gpu1_Name = np.loadtxt(gpu1_Name_file, dtype = str).tolist()
    gpu2_Name = np.loadtxt(gpu2_Name_file, dtype = str).tolist()
    gpu3_Name = np.loadtxt(gpu3_Name_file, dtype = str).tolist()
    gpu4_Name = np.loadtxt(gpu4_Name_file, dtype = str).tolist()
    
    j_file = args.J_Score
    b_file = args.B_Score
    jm_file = args.JM_Score
    bm_file = args.BM_Score
    val_txt_file = args.val_sample
    
    File = [j_file, b_file, jm_file, bm_file] 
    Rate = [0.1, 0.1, 0.1, 0.1, 3.0]
    
    RGB_Score = torch.concat((gpu1_score, gpu2_score, gpu3_score, gpu4_score), dim = 0) # Sample_Num, Numclass
    gpu1_Name.extend(gpu2_Name)
    gpu1_Name.extend(gpu3_Name)
    gpu1_Name.extend(gpu4_Name)
    RGB_Score = match_score(gpu1_Name, RGB_Score, val_txt_file)
    
    if args.benchmark == 'NTU60XSub': # [0.1, 0.1, 0.1, 0.1, 4.0]
        Numclass = 60
        Sample_Num = 16487
        Score = Cal_Score(File, RGB_Score, Rate, Sample_Num, Numclass)
        true_label = gen_label_ntu(val_txt_file)
    
    elif args.benchmark == 'NTU60XView': # [0.1, 0.1, 0.1, 0.1, 3.0]
        Numclass = 60
        Sample_Num = 18932
        Score = Cal_Score(File, RGB_Score, Rate, Sample_Num, Numclass)
        true_label = gen_label_ntu(val_txt_file)
    
    elif args.benchmark == 'NTU120XSub': # [0.1, 0.1, 0.1, 0.1, 6.0]
        Numclass = 120
        Sample_Num = 50919
        Score = Cal_Score(File, RGB_Score, Rate, Sample_Num, Numclass)
        true_label = gen_label_ntu(val_txt_file)
    
    elif args.benchmark == 'NTU120XSet': # [0.1, 0.1, 0.1, 0.1, 6.0]
        Numclass = 120
        Sample_Num = 59477
        Score = Cal_Score(File, RGB_Score, Rate, Sample_Num, Numclass)
        true_label = gen_label_ntu(val_txt_file)

    Acc = Cal_Acc(Score, true_label)
    conf_matrix, class_accuracies, mean_perclass_accuracy = mean_class_accuracies(Score, true_label, Numclass)

    print('acc:', Acc)
    print('mca:', mean_perclass_accuracy)
    print("All Done!")