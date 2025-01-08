# HPNet-Action
Implementation of the paper “Heatmap Pooling for Action Recognition from Videos”.

# Download Dataset
1. **NTU-RGB+D 60** dataset from [https://rose1.ntu.edu.sg/dataset/actionRecognition/](https://rose1.ntu.edu.sg/dataset/actionRecognition/)
2. **NTU-RGB+D 120** dataset from [https://rose1.ntu.edu.sg/dataset/actionRecognition/](https://rose1.ntu.edu.sg/dataset/actionRecognition/)
3. **Toyota-Smarthome** dataset from [https://project.inria.fr/toyotasmarthome/](https://project.inria.fr/toyotasmarthome/)

# Process Dataset
```bash
cd Process_data
```
1. Download pretrained weights from [Google](https://drive.google.com/file/d/1MtljnHRv9R6F1ixMfIS0nqvLDyL2fe8a/view?usp=sharing)
2. Unzip and move
```bash
unzip SimCC_Pose_weights.zip
mv SimCC_Pose_weights/hrnet_w32-36af842e.pth ./pretrained
mv SimCC_Pose_weights/pose_hrnet_w48_256x192_split2_sigma4.pth ./pretrained
mv SimCC_Pose_weights/yolov5m.pt ./pretrained
mv SimCC_Pose_weights/fast_res50_256x192.pth ./models/sppe
mv SimCC_Pose_weights/fast_res101_320x256.pth ./models/sppe
```
3. Extract person frames
```bash
python Etract_person_from_video_xxx.py --sample_name_path <your_sample_name_path> --video_path <your_video_path> --output_path <your_output_path> --device <your_device>

# Example
python Etract_person_from_video_ntu.py --sample_name_path ./sample_txt/test.txt --video_path ./data/videos --output_path ./output/Person_Frame_224 --device 0
```
4. Extract heatmap pooling feature
```bash
python Etract_person_from_video_xxx.py --sample_txt <your_sample_txt> --videos_path <videos_path> --save_path <your_save_path> --device <your_device>

# Example
python Etract_person_from_video_ntu.py --sample_txt ./sample_txt/test.txt --videos_path ./data/videos --save_path ./output/pooling_feature --device 0
```

# Train

# Test

## Ensemble
1. Ensemble one-stream (1s)
```bash
cd ensemble
python ensemble_1s \
--gpu1_Score <gpu1_Score_path> \
--gpu2_Score <gpu2_Score_path> \
--gpu3_Score <gpu3_Score_path> \
--gpu4_Score <gpu4_Score_path> \
--gpu1_Name <gpu1_Name_path> \
--gpu2_Name <gpu2_Name_path> \
--gpu3_Name <gpu3_Name_path> \
--gpu4_Name <gpu4_Name_path> \
--val_sample <val_sample_path> \
--benchmark <NTU60XSub, NTU60XView, NTU120XSub, NTU120XSet, Smarthome_CS, Smarthome_CV1, Smarthome_CV2>

# demo example:
python ensemble_1s.py \
--gpu1_Score ./Smarthome_CS/0_best_score.npy \
--gpu2_Score ./Smarthome_CS/1_best_score.npy \
--gpu3_Score ./Smarthome_CS/2_best_score.npy \
--gpu4_Score ./Smarthome_CS/3_best_score.npy \
--gpu1_Name ./Smarthome_CS/0_best_name.txt \
--gpu2_Name ./Smarthome_CS/1_best_name.txt \
--gpu3_Name ./Smarthome_CS/2_best_name.txt \
--gpu4_Name ./Smarthome_CS/3_best_name.txt \
--val_sample ./Smarthome_CS/test_CS.txt \
--benchmark Smarthome_CS
```
2. Ensemble multi-stream
```bash
cd ensemble
# set rate (x.x, x.x, x.x, x.x, x.x) # J B JM BM HP-Net
python ensemble \
--gpu1_Score <gpu1_Score_path> \
--gpu2_Score <gpu2_Score_path> \
--gpu3_Score <gpu3_Score_path> \
--gpu4_Score <gpu4_Score_path> \
--gpu1_Name <gpu1_Name_path> \
--gpu2_Name <gpu2_Name_path> \
--gpu3_Name <gpu3_Name_path> \
--gpu4_Name <gpu4_Name_path> \
--J_Score <J_Score_path> \
--B_Score <B_Score_path> \
--JM_Score <JM_Score_path> \
--BM_Score <BM_Score_path> \
--val_sample <val_sample_path> \
--benchmark <NTU60XSub, NTU60XView, NTU120XSub, NTU120XSet, Smarthome_CS, Smarthome_CV1, Smarthome_CV2>

# demo example:
# set rate (0.1, 0.1, 0.1, 0.1, 4.0) # HP-Net J B JM BM
python ensemble \
--gpu1_Score ./Smarthome_CV1/0_best_score.npy \
--gpu2_Score ./Smarthome_CV1/1_best_score.npy \
--gpu3_Score ./Smarthome_CV1/2_best_score.npy \
--gpu4_Score ./Smarthome_CV1/3_best_score.npy \
--gpu1_Name ./Smarthome_CV1/0_best_name.txt \
--gpu2_Name ./Smarthome_CV1/1_best_name.txt \
--gpu3_Name ./Smarthome_CV1/2_best_name.txt \
--gpu4_Name ./Smarthome_CV1/3_best_name.txt \
--J_Score ./Smarthome_CV1/J_epoch1_test_score.pkl \
--B_Score ./Smarthome_CV1/B_epoch1_test_score.pkl \
--JM_Score ./Smarthome_CV1/JM_epoch1_test_score.pkl \
--BM_Score ./Smarthome_CV1/BM_epoch1_test_score.pkl \
--val_sample ./Smarthome_CV1/test_CV1.txt \
--benchmark Smarthome_CV1
```


# Contact
For any questions, feel free to contact: ```liujf69@gmail.com``` or ```jinfullliu@tencent.com```