import os
import numpy as np

if __name__ == "__main__":
    # data_path = "/data/liujinfu/dataset/Toyota-Smarthome/person_224"
    data_path = "/data/liujinfu/dataset/NW-UCLA/person_224"
    name_list = os.listdir(data_path)
    tgt_txt = []
    zero_txt = []
    for idx, name in enumerate(name_list):
        name_path = os.path.join(data_path, name)
        imgs_nums = len(os.listdir(name_path))
        if imgs_nums == 0: # zero frames
            zero_txt.append(name)
        elif imgs_nums < 8: # less 8 frames
            tgt_txt.append(name)
            print(name)
    np.savetxt('./zero.txt', zero_txt, fmt = "%s")
    np.savetxt('./less_8_frames.txt', tgt_txt, fmt = "%s")
    print("debug pause")