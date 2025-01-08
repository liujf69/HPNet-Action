import torch
import pickle
import argparse
import numpy as np
import pandas as pd

# Smarthome
Smarthome_action_label_dict = {'Cook.Cleandishes': 0, 'Cook.Cleanup': 1, 'Cook.Cut': 2, 'Cook.Stir': 3, 'Cook.Usestove': 4, 'Cutbread': 5, 'Drink.Frombottle': 6,
                               'Drink.Fromcan': 7, 'Drink.Fromcup': 8, 'Drink.Fromglass': 9, 'Eat.Attable': 10, 'Eat.Snack': 11, 'Enter': 12, 'Getup': 13, 
                               'Laydown': 14, 'Leave': 15, 'Makecoffee.Pourgrains': 16, 'Makecoffee.Pourwater': 17, 'Maketea.Boilwater': 18, 'Maketea.Insertteabag': 19,
                               'Pour.Frombottle': 20, 'Pour.Fromcan': 21, 'Pour.Fromkettle': 22, 'Readbook': 23, 'Sitdown': 24, 'Takepills': 25, 'Uselaptop': 26,
                               'Usetablet': 27, 'Usetelephone': 28, 'Walk': 29, 'WatchTV': 30}

Smarthome_action_label_dict_CV = {'Cutbread': 0, 'Drink.Frombottle': 1, 'Drink.Fromcan': 2, 'Drink.Fromcup': 3, 'Drink.Fromglass': 4, 'Eat.Attable': 5, 'Eat.Snack': 6,
                               'Enter': 7, 'Getup': 8, 'Leave': 9, 'Pour.Frombottle': 10, 'Pour.Fromcan': 11, 'Readbook': 12, 'Sitdown': 13, 
                               'Takepills': 14, 'Uselaptop': 15, 'Usetablet': 16, 'Usetelephone': 17, 'Walk': 18}

def get_parser():
    parser = argparse.ArgumentParser(description = 'multi-stream ensemble') 
    parser.add_argument(
        '--gpu1_Score', 
        type = str,
        default = './Smarthome_CS/0_best_score.npy') # Smarthome_CS
        # default = './Smarthome_CV1/0_best_score.npy') # Smarthome_CV1
        # default = './Smarthome_CV2/0_best_score.npy') # Smarthome_CV2
    parser.add_argument(
        '--gpu2_Score', 
        type = str,
        default = './Smarthome_CS/1_best_score.npy') # Smarthome_CS
        # default = './Smarthome_CV1/1_best_score.npy') # Smarthome_CV1
        # default = './Smarthome_CV2/1_best_score.npy') # Smarthome_CV2
    parser.add_argument(
        '--gpu3_Score', 
        type = str,
        default = './Smarthome_CS/2_best_score.npy') # Smarthome_CS
        # default = './Smarthome_CV1/2_best_score.npy') # Smarthome_CV1
        # default = './Smarthome_CV2/2_best_score.npy') # Smarthome_CV2
    parser.add_argument(
        '--gpu4_Score', 
        type = str,
        default = './Smarthome_CS/3_best_score.npy') # Smarthome_CS
        # default = './Smarthome_CV1/3_best_score.npy') # Smarthome_CV1
        # default = './Smarthome_CV2/3_best_score.npy') # Smarthome_CV2
    parser.add_argument(
        '--gpu1_Name', 
        type = str,
        default = './Smarthome_CS/0_best_name.txt') # Smarthome CS
        # default = './Smarthome_CV1/0_best_name.txt') # Smarthome CV1
        # default = './Smarthome_CV2/0_best_name.txt') # Smarthome_CV2
    parser.add_argument(
        '--gpu2_Name', 
        type = str,
        default = './Smarthome_CS/1_best_name.txt') # Smarthome CS
        # default = './Smarthome_CV1/1_best_name.txt') # Smarthome CV1
        # default = './Smarthome_CV2/1_best_name.txt') # Smarthome CV2
    parser.add_argument(
        '--gpu3_Name', 
        type = str,
        default = './Smarthome_CS/2_best_name.txt') # Smarthome CS
        # default = './Smarthome_CV1/2_best_name.txt') # Smarthome CV1
        # default = './Smarthome_CV2/2_best_name.txt') # Smarthome CV2
    parser.add_argument(
        '--gpu4_Name', 
        type = str,
        default = './Smarthome_CS/3_best_name.txt') # Smarthome CS
        # default = './Smarthome_CV1/3_best_name.txt') # Smarthome CV1
        # default = './Smarthome_CV2/3_best_name.txt') # Smarthome CV2
    parser.add_argument(
        '--J_Score', 
        type = str,
        default = './Smarthome_CS/J_epoch1_test_score.pkl') # Smarthome CS H
        # default = './Smarthome_CV1/J_epoch1_test_score.pkl') # Smarthome CV1 H
        # default = './Smarthome_CV2/J_epoch1_test_score.pkl') # Smarthome CV2 H
    parser.add_argument(
        '--B_Score', 
        type = str,
        default = './Smarthome_CS/B_epoch1_test_score.pkl') # Smarthome CS H
        # default = './Smarthome_CV1/B_epoch1_test_score.pkl') # Smarthome CV1 H
        # default = './Smarthome_CV2/B_epoch1_test_score.pkl') # Smarthome CV2 H
    parser.add_argument(
        '--JM_Score', 
        type = str,
        default = './Smarthome_CS/JM_epoch1_test_score.pkl') # Smarthome CS H
        # default = './Smarthome_CV1/JM_epoch1_test_score.pkl') # Smarthome CV1 H
        # default = './Smarthome_CV2/JM_epoch1_test_score.pkl') # Smarthome CV2 H
    parser.add_argument(
        '--BM_Score', 
        type = str,
        default = './Smarthome_CS/BM_epoch1_test_score.pkl') # Smarthome CS H
        # default = './Smarthome_CV1/BM_epoch1_test_score.pkl') # Smarthome CV1 H
        # default = './Smarthome_CV2/BM_epoch1_test_score.pkl') # Smarthome CV2 H
    parser.add_argument(
        '--val_sample', 
        type = str,
        default = './Smarthome_CS/test_CS.txt') # Smarthome CS
        # default = './Smarthome_CV1/test_CV1.txt') # Smarthome CV1
        # default = './Smarthome_CV2/test_CV2.txt') # Smarthome CV2
    parser.add_argument(
        '--benchmark', 
        type = str,
        default = 'Smarthome_CS')
        # default = 'Smarthome_CV1')
        # default = 'Smarthome_CV2')
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
    conf_matrix = torch.zeros((num_classes, num_classes))
    pred_classes = torch.argmax(preds, dim=1)

    for (gt_class, pred_class) in zip(labels, pred_classes):
      conf_matrix[int(gt_class.item()), int(pred_class.item())] += 1

    class_accuracies = conf_matrix.diag() / conf_matrix.sum(dim=1) # num_classes
    mean_perclass_accuracy = class_accuracies.mean()

    return conf_matrix, class_accuracies, mean_perclass_accuracy

def gen_label_sh(val_txt_file, benchmark):
    true_label = []
    val_txt = np.loadtxt(val_txt_file, dtype = str)
    for idx, name in enumerate(val_txt):
        action = name.split("_")[0]
        if benchmark == 'Smarthome_CS':
            label = int(Smarthome_action_label_dict[action])
        else:
            label = int(Smarthome_action_label_dict_CV[action])
        true_label.append(label)
    true_label = torch.from_numpy(np.array(true_label))
    return true_label
    
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
    Rate = [0.1, 0.1, 0.1, 0.1, 4.0]
    
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
    
    elif args.benchmark == 'Smarthome_CS': # [0.1, 0.1, 0.1, 0.1, 11.0]
        Numclass = 31
        Sample_Num = 5433
        Score = Cal_Score(File, RGB_Score, Rate, Sample_Num, Numclass)
        true_label = gen_label_sh(val_txt_file, args.benchmark) 
    
    elif args.benchmark == 'Smarthome_CV1': # [0.1, 0.1, 0.1, 0.1, 4.0]
        Numclass = 19
        Sample_Num = 1901
        Score = Cal_Score(File, RGB_Score, Rate, Sample_Num, Numclass)
        true_label = gen_label_sh(val_txt_file, args.benchmark) 
    
    elif args.benchmark == 'Smarthome_CV2': # [0.1, 0.1, 0.1, 0.1, 8.0]
        Numclass = 19
        Sample_Num = 1901
        Score = Cal_Score(File, RGB_Score, Rate, Sample_Num, Numclass)
        true_label = gen_label_sh(val_txt_file, args.benchmark) 
        
    Acc = Cal_Acc(Score, true_label)
    conf_matrix, class_accuracies, mean_perclass_accuracy = mean_class_accuracies(Score, true_label, Numclass)

    print('acc:', Acc)
    print('mca:', mean_perclass_accuracy.item())
    print("All Done!")