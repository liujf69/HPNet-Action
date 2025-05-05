import numpy as np

if __name__ == "__main__":
    train_txt_path = '/data/liujinfu/GCN_OBJ_HAR/dataset/NTU60_XView_train.txt'
    train_txt = np.loadtxt(train_txt_path, dtype = str)
    label = []
    for idx, name in enumerate(train_txt):
        label.append(name[-3:])
    np.savetxt("/data/liujinfu/GCN_OBJ_HAR/dataset/NTU60_XView_train_label.txt", label, fmt = "%s")

    test_txt_path = '/data/liujinfu/GCN_OBJ_HAR/dataset/NTU60_XView_val.txt'
    test_txt = np.loadtxt(test_txt_path, dtype = str)
    label = []
    for idx, name in enumerate(test_txt):
        label.append(name[-3:])
    np.savetxt("/data/liujinfu/GCN_OBJ_HAR/dataset/NTU60_XView_val_label.txt", label, fmt = "%s")
