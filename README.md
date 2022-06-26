# Improving YOLO for Video Object Detection with Sparse Sampling and Seq-NMS 


## Installation

```
# create conda environment:

conda create -n seq-yolo python=3.7
pip install -r requirements.txt

# install mish-cuda

cd mish-cuda
python setup.py install build

# go to code folder
cd ..
mkdir weights
```

## Make dataset
### test
```
# for testing, you should construct the structure as follow:

# original
/path/Test/
    Apr/frames/
       date/clip/
           images_xxxx.jpg
           
# for testing, you can run 'python create_testset.py' after changing the /path
/path/together/
    Test/split/
        Apr/images/
           date_clip_xxxx.jpg

```
### train
####We randomly select 3289 iamges from validation images to evaluate during training, you can download it from [eval.zip](https://drive.google.com/file/d/1o060lx2FpvMeDIsNNSB01gZ3hddUok_B/view?usp=sharing)

```
# for training, you should construct the structure as follow:

# original
/path/Train/
    Day/
        frames/
           date/clip/
               images_xxxx.jpg
        annotations/
           date/clip/
               annotations_xxxx.txt
    Week/
        frames/
           date/clip/
               images_xxxx.jpg
        annotations/
           date/clip/
               annotations_xxxx.txt
    Month/
        frames/
           date/clip/
               images_xxxx.jpg
        annotations/
           date/clip/
               annotations_xxxx.txt
           
# for training, you can run 'python create_trainset.py' after changing 'F_PATH' to your own path and 'SPLIT' to 'Day' or 'Week' or 'Month' in create_trainset.py

/path/together/
    Train/
        Day/
            images/
               date_clip_xxxx.jpg
            labels/
               date_clip_xxxx.txt
            all.txt
            Day_all_e10_bic10_mot.txt
            Day_all_e5.txt
        Week/
            images/
               date_clip_xxxx.jpg
            labels/
               date_clip_xxxx.txt
            all.txt
            Week_all_e5.txt
        Month/
            images/
               date_clip_xxxx.jpg
            labels/
               date_clip_xxxx.txt
            all.txt
            Month_all_e10_bic10_mot10.txt
            Month_all_e10_bic10_mot10_a.txt
            Month_all_e10.txt
            Month_all_e15.txt
        eval/
            images/
                xxxx.jpg
            labels/
                xxxx.txt
                
# all txt files containing the path of our training frames are provided in txt_files, just copy them to /path/together/Train/Day(or Week or Month) 

```


## Testing
###day-level
####[Day_a_e10_bic10_mot_p6_ep100_bs32_img1280_scratch_59-99-5_ms_51.pt](https://drive.google.com/file/d/1XJdOTZYDwSI8CYguR-Bhs3tjF4l2XnLc/view?usp=sharing)
####[Day_a_e5_p7_ep100_bs16_img1536_scratch_29-99-8_ms_51_aug_seqnms_0.7_0.3_0.6_0.3.pt](https://drive.google.com/file/d/1C0vBgDljJMPN30GSrR__BndFsY_a4Clx/view?usp=sharing)
```
## for reproduce the results of ECCV'22 ChaLearn Seasons in Drift Challenge (track 1: day level). 
# Firstly, change the test dir in data/Chalearn_test_{month}.yaml

python test.py --img-size 1280 --conf-thres 0.1 --iou-thres 0.5 --batch 32 --device 0 --data data/Chalearn_test_Sep.yaml --month Sep --weights weights/Day_a_e10_bic10_mot_p6_ep100_bs32_img1280_scratch_59-99-5_ms_51.pt --exist-ok --save-pkl --name Day/a_e10

# you will get one pkl file startswith 'Sep' in runs/test/Day/a_e10, change 'pkl_path' in remove_scores.py to 'runs/test/Day/a_e10', change 'score_threshold' to 0.1 in remove_scores.py, and change path of sub-fuction 'sup_object' of remove_scores.py to your own dir, then run:

python remove_scores.py

# you will get 'Sep.pkl' in runs/test/Day/a_e10

python test.py --img-size 1536 --conf-thres 0.1 --iou-thres 0.5 --batch 32 --device 0 --data data/Chalearn_test_Apr.yaml --month Apr --weights weights/Day_a_e5_p7_ep100_bs16_img1536_scratch_29-99-8_ms_51_aug_seqnms_0.7_0.3_0.6_0.3.pt --exist-ok --save-pkl --name Day/a_e5 --augment
python test.py --img-size 1536 --conf-thres 0.1 --iou-thres 0.5 --batch 32 --device 0 --data data/Chalearn_test_Aug.yaml --month Aug --weights weights/Day_a_e5_p7_ep100_bs16_img1536_scratch_29-99-8_ms_51_aug_seqnms_0.7_0.3_0.6_0.3.pt --exist-ok --save-pkl --name Day/a_e5 --augment
python test.py --img-size 1536 --conf-thres 0.1 --iou-thres 0.5 --batch 32 --device 0 --data data/Chalearn_test_Jan.yaml --month Jan --weights weights/Day_a_e5_p7_ep100_bs16_img1536_scratch_29-99-8_ms_51_aug_seqnms_0.7_0.3_0.6_0.3.pt --exist-ok --save-pkl --name Day/a_e5 --augment
python test.py --img-size 1536 --conf-thres 0.1 --iou-thres 0.5 --batch 32 --device 0 --data data/Chalearn_test_Jul.yaml --month Jul --weights weights/Day_a_e5_p7_ep100_bs16_img1536_scratch_29-99-8_ms_51_aug_seqnms_0.7_0.3_0.6_0.3.pt --exist-ok --save-pkl --name Day/a_e5 --augment
python test.py --img-size 1536 --conf-thres 0.1 --iou-thres 0.5 --batch 32 --device 0 --data data/Chalearn_test_Jun.yaml --month Jun --weights weights/Day_a_e5_p7_ep100_bs16_img1536_scratch_29-99-8_ms_51_aug_seqnms_0.7_0.3_0.6_0.3.pt --exist-ok --save-pkl --name Day/a_e5 --augment
python test.py --img-size 1536 --conf-thres 0.1 --iou-thres 0.5 --batch 32 --device 0 --data data/Chalearn_test_Mar.yaml --month Mar --weights weights/Day_a_e5_p7_ep100_bs16_img1536_scratch_29-99-8_ms_51_aug_seqnms_0.7_0.3_0.6_0.3.pt --exist-ok --save-pkl --name Day/a_e5 --augment
python test.py --img-size 1536 --conf-thres 0.1 --iou-thres 0.5 --batch 32 --device 0 --data data/Chalearn_test_May.yaml --month May --weights weights/Day_a_e5_p7_ep100_bs16_img1536_scratch_29-99-8_ms_51_aug_seqnms_0.7_0.3_0.6_0.3.pt --exist-ok --save-pkl --name Day/a_e5 --augment

# you will get seven pkl files in runs/test/Day/a_e5
# use seq-nms to the seven pkl files in runs/test/Day/a_e5, you should change path of sub-fuction 'VN' of oneseqnms.py to your own dir, change the 'SRC_PATH' in oneseqnms.py to runs/test/Day/a_e5, change 'sn_mons' to ['Apr', 'Aug', 'Jan', 'Jul', 'Jun', 'Mar', 'May'], and change 'CONF_THRESH' in seq_nms.py to 0.7, then run:

python oneseqnms.py

# you will get post-processing results in runs/test/Day/a_e5/src_split_seq_0.7_0.3_0.6_0.3_hb/src
# copy the pkl file 'Sep.pkl' in runs/test/Day/a_e10 to runs/test/Day/a_e5/src_split_seq_0.7_0.3_0.6_0.3_hb/src, then change 'path' in submit.py to 'runs/test/Day/a_e5/src_split_seq_0.7_0.3_0.6_0.3_hb', and change path of sub-fuction 'sup_object' of submit.py to your own dir,  then run:

python submit.py

# you will get the final pkl file 'predictions.pkl' in runs/test/Day/a_e5/src_split_seq_0.7_0.3_0.6_0.3_hb, which achieved the-state-of-art in ECCV'22 ChaLearn Seasons in Drift Challenge (track 1: day level).

----------------------------------------------------------------------------------------------
|mAP weighted|   mAP  |   Jan  |   Mar  |   Apr  |   May  |   Jun  |   Jul  |   Aug  |   Sep  |
----------------------------------------------------------------------------------------------
|  0.279846  | 0.2832 | 0.3048 | 0.3021 | 0.3073 | 0.2674 | 0.2748 | 0.2306 | 0.2829 | 0.2955 |
----------------------------------------------------------------------------------------------

```

###week-level
####[Week_a_e5_p7_ep100_bs16_img1536_scratch_14-29-54-69-79-ms_51_aug_seqnms_0.6_0.3_0.6_0.3.pt](https://drive.google.com/file/d/1QcMoseTQUjhWKMXaqUQLtODEk6wx-8J7/view?usp=sharing)
####[Week_a_e5_p7_ep100_bs16_img1536_scratch_9-14-49-64-69-5_ms_51_seqnms_0.6_0.3_0.6_0.3.pt](https://drive.google.com/file/d/1UywEOtynrsqytPhjzS1M07gIo-_NY15j/view?usp=sharing)
####[Week_a_e5_p7_ep100_bs16_img1536_scratch_14-29-59-64-74-ms_56.pt](https://drive.google.com/file/d/1ORv_9EjSIkpHbgBBWI_hmtWtQq3KJR-7/view?usp=sharing)
```
## for reproduce the results of ECCV'22 ChaLearn Seasons in Drift Challenge (track 1: week level). 
# Firstly, change the test dir in data/Chalearn_test_{month}.yaml

python test.py --img-size 1536 --conf-thres 0.1 --iou-thres 0.5 --batch 32 --device 0 --data data/Chalearn_test_Apr.yaml --month Apr --weights weights/Week_a_e5_p7_ep100_bs16_img1536_scratch_14-29-54-69-79-ms_51_aug_seqnms_0.6_0.3_0.6_0.3.pt --exist-ok --save-pkl --name Week/a_e5_5 --augment
python test.py --img-size 1536 --conf-thres 0.1 --iou-thres 0.5 --batch 32 --device 0 --data data/Chalearn_test_Aug.yaml --month Aug --weights weights/Week_a_e5_p7_ep100_bs16_img1536_scratch_14-29-54-69-79-ms_51_aug_seqnms_0.6_0.3_0.6_0.3.pt --exist-ok --save-pkl --name Week/a_e5_5 --augment
python test.py --img-size 1536 --conf-thres 0.1 --iou-thres 0.5 --batch 32 --device 0 --data data/Chalearn_test_Jan.yaml --month Jan --weights weights/Week_a_e5_p7_ep100_bs16_img1536_scratch_14-29-54-69-79-ms_51_aug_seqnms_0.6_0.3_0.6_0.3.pt --exist-ok --save-pkl --name Week/a_e5_5 --augment
python test.py --img-size 1536 --conf-thres 0.1 --iou-thres 0.5 --batch 32 --device 0 --data data/Chalearn_test_Jul.yaml --month Jul --weights weights/Week_a_e5_p7_ep100_bs16_img1536_scratch_14-29-54-69-79-ms_51_aug_seqnms_0.6_0.3_0.6_0.3.pt --exist-ok --save-pkl --name Week/a_e5_5 --augment
python test.py --img-size 1536 --conf-thres 0.1 --iou-thres 0.5 --batch 32 --device 0 --data data/Chalearn_test_Jun.yaml --month Jun --weights weights/Week_a_e5_p7_ep100_bs16_img1536_scratch_14-29-54-69-79-ms_51_aug_seqnms_0.6_0.3_0.6_0.3.pt --exist-ok --save-pkl --name Week/a_e5_5 --augment

# you will get five pkl files in runs/test/Week/a_e5_5
# use seq-nms to the five pkl files in runs/test/Week/a_e5_5, you should change the 'SRC_PATH' in oneseqnms.py to runs/test/Week/a_e5_5, change 'sn_mons' to ['Apr', 'Aug', 'Jan', 'Jul', 'Jun'], and change 'CONF_THRESH' in seq_nms.py to 0.6, then run:

python oneseqnms.py

# you will get post-processing results in runs/test/Week/a_e5_5/src_split_seq_0.6_0.3_0.6_0.3_hb/src

python test.py --img-size 1536 --conf-thres 0.1 --iou-thres 0.5 --batch 32 --device 0 --data data/Chalearn_test_Sep.yaml --month Sep --weights weights/Week_a_e5_p7_ep100_bs16_img1536_scratch_9-14-49-64-69-5_ms_51_seqnms_0.6_0.3_0.6_0.3.pt --exist-ok --save-pkl --name Week/a_e5_1

# you will get one pkl file startswith 'Sep' in runs/test/Week/a_e5_1
# use seq-nms to the pkl file in runs/test/Week/a_e5_1, you should change the 'SRC_PATH' in oneseqnms.py to runs/test/Week/a_e5_1, change 'sn_mons' to ['Sep'], and change 'CONF_THRESH' in seq_nms.py to 0.6, then run:

python oneseqnms.py

# you will get a post-processing result in runs/test/Week/a_e5_1/src_split_seq_0.6_0.3_0.6_0.3_hb/src

python test.py --img-size 1536 --conf-thres 0.6 --iou-thres 0.5 --batch 32 --device 0 --data data/Chalearn_test_Mar.yaml --month Mar --weights weights/Week_a_e5_p7_ep100_bs16_img1536_scratch_14-29-59-64-74-ms_56.pt --exist-ok --save-pkl --name Week/a_e5_2
python test.py --img-size 1536 --conf-thres 0.6 --iou-thres 0.5 --batch 32 --device 0 --data data/Chalearn_test_May.yaml --month May --weights weights/Week_a_e5_p7_ep100_bs16_img1536_scratch_14-29-59-64-74-ms_56.pt --exist-ok --save-pkl --name Week/a_e5_2

# you will get two pkl files in runs/test/Week/a_e5_2, change 'pkl_path' in remove_scores.py to 'runs/test/Week/a_e5_2', change 'score_threshold' to 0.6 in remove_scores.py, then run:

python remove_scores.py

# you will get 'Mar.pkl, May.pkl' in runs/test/Week/a_e5_2
# copy the pkl file 'Sep.pkl' in runs/test/Week/a_e5_1/src_split_seq_0.6_0.3_0.6_0.3_hb/src and 'Mar.pkl, May.pkl' in runs/test/Week/a_e5_2 to runs/test/Week/a_e5_5/src_split_seq_0.6_0.3_0.6_0.3_hb/src, then change 'path' in submit.py to 'runs/test/Week/a_e5_5/src_split_seq_0.6_0.3_0.6_0.3_hb' and run:

python submit.py

# you will get the final pkl file 'predictions.pkl' in runs/test/Week/a_e5_5/src_split_seq_0.6_0.3_0.6_0.3_hb, which achieved the-state-of-art in ECCV'22 ChaLearn Seasons in Drift Challenge (track 1: week level).

----------------------------------------------------------------------------------------------
|mAP weighted|   mAP  |   Jan  |   Mar  |   Apr  |   May  |   Jun  |   Jul  |   Aug  |   Sep  |
----------------------------------------------------------------------------------------------
|  0.323652  | 0.3305 | 0.3708 | 0.3502 | 0.3323 | 0.2774 | 0.2924 | 0.2506 | 0.3162 | 0.4542 |
----------------------------------------------------------------------------------------------

```

###month-level
####[Month_a_e10_bic10_mot_p6_ep100_bs32_img1280_scratch_39-79-5_ms_51_seqnms_0.6_0.3_0.6_0.3.pt](https://drive.google.com/file/d/1WlyCinouCQxOqhXsAfZpgmlqOVcGs0lP/view?usp=sharing)
####[Month_a_e10_p6_ep100_bs32_img1280_scratch_29-69-5_ms_51_aug_seqnms_0.5_0.3_0.6_0.3.pt](https://drive.google.com/file/d/1fgVZgQz5DYEyNUKApF8eTk2lppc6-Htn/view?usp=sharing)
####[Month_a_e10_p7_ep100_bs16_img1536_scratch_24-39-59-79-84-ms_51_aug_seqnms_0.5_0.3_0.6_0.3.pt](https://drive.google.com/file/d/17pUafzBPNfMfhcmnCfyK1fOJ1QyEEVeN/view?usp=sharing)
####[Month_a_e15_p6_ep100_bs64_img1280_scratch_59-99-5_ms_51.pt](https://drive.google.com/file/d/1jgkG_RU2lNIg6lAuhhUCRLNK2oOwNYfA/view?usp=sharing)
```
## for reproduce the results of ECCV'22 ChaLearn Seasons in Drift Challenge (track 1: month level). 
# Firstly, change the test dir in data/Chalearn_test_{month}.yaml

python test.py --img-size 1280 --conf-thres 0.1 --iou-thres 0.5 --batch 32 --device 0 --data data/Chalearn_test_Jan.yaml --month Jan --weights weights/Month_a_e10_bic10_mot_p6_ep100_bs32_img1280_scratch_39-79-5_ms_51_seqnms_0.6_0.3_0.6_0.3.pt --exist-ok --save-pkl --name Month/a_e10_bic10_mot

# you will get one pkl file in runs/test/Month/a_e10_bic10_mot
# use seq-nms to the pkl file in runs/test/Month/a_e10_bic10_mot, you should change the 'SRC_PATH' in oneseqnms.py to runs/test/Month/a_e10_bic10_mot, change 'sn_mons' to ['Jan'], and change 'CONF_THRESH' in seq_nms.py to 0.6, then run:

python oneseqnms.py

# you will get one post-processing result in runs/test/Month/a_e10_bic10_mot/src_split_seq_0.6_0.3_0.6_0.3_hb/src

python test.py --img-size 1280 --conf-thres 0.1 --iou-thres 0.5 --batch 32 --device 0 --data data/Chalearn_test_Mar.yaml --month Mar --weights weights/Month_a_e10_p6_ep100_bs32_img1280_scratch_29-69-5_ms_51_aug_seqnms_0.5_0.3_0.6_0.3.pt --exist-ok --save-pkl --name Month/a_e10_1 --augment

# you will get one pkl file in runs/test/Month/a_e10_1
# use seq-nms to the pkl file in runs/test/Month/a_e10_1, you should change the 'SRC_PATH' in oneseqnms.py to runs/test/Month/a_e10_1, change 'sn_mons' to ['Mar'], and change 'CONF_THRESH' in seq_nms.py to 0.5, then run:

python oneseqnms.py

# you will get a post-processing result in runs/test/Month/a_e10_1/src_split_seq_0.5_0.3_0.6_0.3_hb/src

python test.py --img-size 1536 --conf-thres 0.1 --iou-thres 0.5 --batch 32 --device 0 --data data/Chalearn_test_Apr.yaml --month Apr --weights weights/Month_a_e10_p7_ep100_bs16_img1536_scratch_24-39-59-79-84-ms_51_aug_seqnms_0.5_0.3_0.6_0.3.pt --exist-ok --save-pkl --name Month/a_e10_4 --augment
python test.py --img-size 1536 --conf-thres 0.1 --iou-thres 0.5 --batch 32 --device 0 --data data/Chalearn_test_Jun.yaml --month Jun --weights weights/Month_a_e10_p7_ep100_bs16_img1536_scratch_24-39-59-79-84-ms_51_aug_seqnms_0.5_0.3_0.6_0.3.pt --exist-ok --save-pkl --name Month/a_e10_4 --augment
python test.py --img-size 1536 --conf-thres 0.1 --iou-thres 0.5 --batch 32 --device 0 --data data/Chalearn_test_May.yaml --month May --weights weights/Month_a_e10_p7_ep100_bs16_img1536_scratch_24-39-59-79-84-ms_51_aug_seqnms_0.5_0.3_0.6_0.3.pt --exist-ok --save-pkl --name Month/a_e10_4 --augment
python test.py --img-size 1536 --conf-thres 0.1 --iou-thres 0.5 --batch 32 --device 0 --data data/Chalearn_test_Jul.yaml --month Jul --weights weights/Month_a_e10_p7_ep100_bs16_img1536_scratch_24-39-59-79-84-ms_51_aug_seqnms_0.5_0.3_0.6_0.3.pt --exist-ok --save-pkl --name Month/a_e10_4 --augment

# you will get four pkl files in runs/test/Month/a_e10_4
# use seq-nms to the pkl files in runs/test/Month/a_e10_4, you should change the 'SRC_PATH' in oneseqnms.py to runs/test/Month/a_e10_4, change 'sn_mons' to ['Apr', 'Jun', 'May', 'Jul'], and change 'CONF_THRESH' in seq_nms.py to 0.5, then run:

python oneseqnms.py

# you will get the four post-processing results in runs/test/Month/a_e10_4/src_split_seq_0.5_0.3_0.6_0.3_hb/src

python test.py --img-size 1536 --conf-thres 0.1 --iou-thres 0.5 --batch 32 --device 0 --data data/Chalearn_test_Aug.yaml --month Aug --weights weights/Month_a_e10_p7_ep100_bs16_img1536_scratch_24-39-59-79-84-ms_51_aug_seqnms_0.5_0.3_0.6_0.3.pt --exist-ok --save-pkl --name Month/a_e10_a1

# you will get a pkl file in runs/test/Month/a_e10_a1
# use seq-nms to the pkl file in runs/test/Month/a_e10_a1, you should change the 'SRC_PATH' in oneseqnms.py to runs/test/Month/a_e10_a1, change 'sn_mons' to ['Aug'], and change 'CONF_THRESH' in seq_nms.py to 0.5, then run:

python oneseqnms.py

# you will get the post-processing result in runs/test/Month/a_e10_a1/src_split_seq_0.5_0.3_0.6_0.3_hb/src

python test.py --img-size 1280 --conf-thres 0.1 --iou-thres 0.5 --batch 32 --device 0 --data data/Chalearn_test_Sep.yaml --month Sep --weights weights/Month_a_e15_p6_ep100_bs64_img1280_scratch_59-99-5_ms_51.pt --exist-ok --save-pkl --name Month/a_e15

# you will get a pkl file in runs/test/Month/a_e15
# use seq-nms to the pkl file in runs/test/Month/a_e15, you should change the 'SRC_PATH' in oneseqnms.py to runs/test/Month/a_e15, change 'sn_mons' to ['Sep'], and change 'CONF_THRESH' in seq_nms.py to 0.6, then run:

python oneseqnms.py

# you will get the post-processing result in runs/test/Month/a_e15/src_split_seq_0.6_0.3_0.6_0.3_hb/src

# copy the pkl file 'Sep.pkl' in runs/test/Month/a_e15/src_split_seq_0.6_0.3_0.6_0.3_hb/src and 'Aug.pkl' in runs/test/Month/a_e10_a1/src_split_seq_0.5_0.3_0.6_0.3_hb/src and 'Jan.pkl' in runs/test/Month/a_e10_bic10_mot/src_split_seq_0.6_0.3_0.6_0.3_hb/src and 'Mar.pkl' in runs/test/Month/a_e10_1/src_split_seq_0.5_0.3_0.6_0.3_hb/src to runs/test/Month/a_e10_4/src_split_seq_0.5_0.3_0.6_0.3_hb/src, then change 'path' in submit.py to 'runs/test/Month/a_e10_4/src_split_seq_0.5_0.3_0.6_0.3_hb' and run:

python submit.py

# you will get the final pkl file 'predictions.pkl' in runs/test/Month/a_e10_4/src_split_seq_0.5_0.3_0.6_0.3_hb, which achieved the-state-of-art in ECCV'22 ChaLearn Seasons in Drift Challenge (track 1: month level).

----------------------------------------------------------------------------------------------
|mAP weighted|   mAP  |   Jan  |   Mar  |   Apr  |   May  |   Jun  |   Jul  |   Aug  |   Sep  |
----------------------------------------------------------------------------------------------
|  0.337645  | 0.3464 | 0.4142 | 0.3729 | 0.3414 | 0.3032 | 0.2933 | 0.2567 | 0.3112 | 0.4779 |
----------------------------------------------------------------------------------------------

```

## Training
####[yolov4-p6.pt](https://drive.google.com/file/d/1aB7May8oPYzBqbgwYSZHuATPXyxh9xnf/view?usp=sharing)
####[yolov4-p7.pt](https://drive.google.com/file/d/18fGlzgEJTkUEiBG4hW00pyedJKNnYLP3/view?usp=sharing)
###day-level
```
## all experiments on day-level are trained on 4 Tesla V100-32G
# First, change the train dir to /path/together/Train/Day/Day_all_e10_bic10_mot.txt in data/Chalearn_Day_all.yaml, then run:

python -m torch.distributed.launch --nproc_per_node 4 train.py --batch-size 32 --epochs 100 --img-size 1280 1280 --data data/Chalearn_Day_all.yaml --weights weights/yolov4-p6.pt --hyp data/hyp.scratch.yaml --sync-bn --device 0,1,2,3 --logdir runs/train/Day/all_e10_bic10_mot/p6_ep100_bs32_img1280_scratch

# After training, you will get some weights in runs/train/Day/all_e10_bic10_mot/p6_ep100_bs32_img1280_scratch/weights. You can choose five best (mAP@0.5:0.95) weights according to the result txt file in runs/train/Day/all_e10_bic10_mot/p6_ep100_bs32_img1280_scratch/results.txt, and put their names to 'w_list' in model_soups.py, change 'PATH' to 'runs/train/Day/all_e10_bic10_mot/p6_ep100_bs32_img1280_scratch' and 'STRUCTURE' to 'models/yolov4-p6.yaml' in model_soups.py. Or you can just choose the weights(last_059.pt, last_069.pt, last_079.pt, last_089.pt, last_099.pt), then run:

python model_soups.py

# you will get an ensemble model in runs/train/Day/all_e10_bic10_mot/p6_ep100_bs32_img1280_scratch, and use it to test. The test hyper-parameters should be same as the above Testing.
# Second, change the train dir to /path/together/Train/Day/Day_all_e5.txt in data/Chalearn_Day_all.yaml, then run:

python -m torch.distributed.launch --nproc_per_node 4 train.py --batch-size 16 --epochs 100 --img-size 1536 1536 --data data/Chalearn_Day_all.yaml --weights weights/yolov4-p7.pt --hyp data/hyp.scratch.yaml --sync-bn --device 0,1,2,3 --logdir runs/train/Day/all_e5/p7_ep100_bs16_img1536_scratch

# After training, you will get some weights in runs/train/Day/all_e5/p7_ep100_bs16_img1536_scratch/weights. You can choose eight best (mAP@0.5:0.95) weights according to the result txt file in runs/train/Day/all_e5/p7_ep100_bs16_img1536_scratch/results.txt, and put their names to 'w_list' in model_soups.py, change 'PATH' to 'runs/train/Day/all_e5/p7_ep100_bs16_img1536_scratch' and 'STRUCTURE' to 'models/yolov4-p7.yaml' in model_soups.py. Or you can just choose the weights(last_029.pt, last_039.pt, last_049.pt, last_059.pt, last_069.pt, last_079.pt, last_089.pt, last_099.pt), then run:

python model_soups.py 

# you will get an ensemble model in runs/train/Day/all_e5/p7_ep100_bs16_img1536_scratch, and use it to test. The test hyper-parameters should be same as the above Testing.
```

###week-level
```
## all experiments on week-level are trained on 4 Tesla V100-32G
# First, change the train dir to /path/together/Train/Week/Week_all_e5.txt in data/Chalearn_Week_all.yaml, then run:

python -m torch.distributed.launch --nproc_per_node 4 train.py --batch-size 16 --epochs 100 --img-size 1536 1536 --data data/Chalearn_Week_all.yaml --weights weights/yolov4-p7.pt --hyp data/hyp.scratch.yaml --sync-bn --device 0,1,2,3 --logdir runs/train/Week/all_e5/p7_ep100_bs16_img1536_scratch

# After training, you will get some weights in runs/train/Week/all_e5/p7_ep100_bs16_img1536_scratch/weights. You should choose five best (mAP@0.5:0.95) weights according to the result txt file in runs/train/Week/all_e5/p7_ep100_bs16_img1536_scratch/results.txt, and put their names to 'w_list' in model_soups.py, change 'PATH' to 'runs/train/Week/all_e5/p7_ep100_bs16_img1536_scratch' and 'STRUCTURE' to 'models/yolov4-p7.yaml' in model_soups.py, then run:

python model_soups.py

# you will get an ensemble model in runs/train/Week/all_e5/p7_ep100_bs16_img1536_scratch, and use it to test. The test hyper-parameters should be same as the above Testing. Besides, to find five best weights performed better on Sep, we used the trained models to only evaluate on Sep of validation images.

```

###month-level
```
## all experiments on month-level are trained on 4 Tesla V100-32G or 8 Tesla V100-32G
# First, change the train dir to /path/together/Train/Month/Month_all_e15.txt in data/Chalearn_Month_all.yaml, then run:

python -m torch.distributed.launch --nproc_per_node 8 train.py --batch-size 64 --epochs 100 --img-size 1280 1280 --data data/Chalearn_Month_all.yaml --weights weights/yolov4-p6.pt --hyp data/hyp.scratch.yaml --sync-bn --device 0,1,2,3,4,5,6,7 --logdir runs/train/Month/all_e15/p6_ep100_bs64_img1280_scratch

# After training, you will get some weights in runs/train/Month/all_e15/p6_ep100_bs64_img1280_scratch/weights. You can choose five best (mAP@0.5:0.95) weights according to the result txt file in runs/train/Month/all_e15/p6_ep100_bs64_img1280_scratch/results.txt, and put their names to 'w_list' in model_soups.py, change 'PATH' to 'runs/train/Month/all_e15/p6_ep100_bs64_img1280_scratch' and 'STRUCTURE' to 'models/yolov4-p6.yaml' in model_soups.py. Or you can just choose the weights(last_059.pt, last_069.pt, last_079.pt, last_089.pt, last_099.pt), then run:

python model_soups.py

# you will get an ensemble model in runs/train/Month/all_e15/p6_ep100_bs64_img1280_scratch, and use it to test. The test hyper-parameters should be same as the above Testing.
# Second, change the train dir to /path/together/Train/Month/Month_all_e10.txt in data/Chalearn_Month_all.yaml, then run:

python -m torch.distributed.launch --nproc_per_node 4 train.py --batch-size 16 --epochs 100 --img-size 1536 1536 --data data/Chalearn_Month_all.yaml --weights weights/yolov4-p7.pt --hyp data/hyp.scratch.yaml --sync-bn --device 0,1,2,3 --logdir runs/train/Month/all_e10/p7_ep100_bs16_img1536_scratch

# After training, you will get some weights in runs/train/Month/all_e10/p7_ep100_bs16_img1536_scratch/weights. You should choose five best (mAP@0.5:0.95) weights according to the result txt file in runs/train/Month/all_e10/p7_ep100_bs16_img1536_scratch/results.txt, and put their names to 'w_list' in model_soups.py, change 'PATH' to 'runs/train/Month/all_e10/p7_ep100_bs16_img1536_scratch' and 'STRUCTURE' to 'models/yolov4-p7.yaml' in model_soups.py, then run:

python model_soups.py 

# you will get an ensemble model in runs/train/Month/all_e10/p7_ep100_bs16_img1536_scratch, and use it to test. The test hyper-parameters should be same as the above Testing.
# Third, change the train dir to /path/together/Train/Month/Month_all_e10.txt in data/Chalearn_Month_all.yaml, then run:

python -m torch.distributed.launch --nproc_per_node 4 train.py --batch-size 32 --epochs 100 --img-size 1280 1280 --data data/Chalearn_Month_all.yaml --weights weights/yolov4-p6.pt --hyp data/hyp.scratch.yaml --sync-bn --device 0,1,2,3 --logdir runs/train/Month/all_e10/p6_ep100_bs32_img1280_scratch

# After training, you will get some weights in runs/train/Month/all_e10/p6_ep100_bs32_img1280_scratch/weights. You can choose five best (mAP@0.5:0.95) weights according to the result txt file in runs/train/Month/all_e10/p6_ep100_bs32_img1280_scratch/results.txt, and put their names to 'w_list' in model_soups.py, change 'PATH' to 'runs/train/Month/all_e10/p6_ep100_bs32_img1280_scratch' and 'STRUCTURE' to 'models/yolov4-p6.yaml' in model_soups.py. Or you can just choose the weights(last_029.pt, last_039.pt, last_049.pt, last_059.pt, last_069.pt), then run:

python model_soups.py 

# you will get an ensemble model in runs/train/Month/all_e10/p6_ep100_bs32_img1280_scratch, and use it to test. The test hyper-parameters should be same as the above Testing.
# Fourth, change the train dir to /path/together/Train/Month/Month_all_e10_bic10_mot10.txt (or Month_all_e10_bic10_mot10_a.txt, I forgot which one I used for trianing, sorry) in data/Chalearn_Month_all.yaml, then run:

python -m torch.distributed.launch --nproc_per_node 4 train.py --batch-size 32 --epochs 100 --img-size 1280 1280 --data data/Chalearn_Month_all.yaml --weights weights/yolov4-p6.pt --hyp data/hyp.scratch.yaml --sync-bn --device 0,1,2,3 --logdir runs/train/Month/all_e10_bic10_mot10/p6_ep100_bs32_img1280_scratch

# After training, you will get some weights in runs/train/Month/all_e10_bic10_mot10/p6_ep100_bs32_img1280_scratch/weights. You can choose five best (mAP@0.5:0.95) weights according to the result txt file in runs/train/Month/all_e10_bic10_mot10/p6_ep100_bs32_img1280_scratch/results.txt, and put their names to 'w_list' in model_soups.py, change 'PATH' to 'runs/train/Month/all_e10_bic10_mot10/p6_ep100_bs32_img1280_scratch' and 'STRUCTURE' to 'models/yolov4-p6.yaml' in model_soups.py. Or you can just choose the weights(last_039.pt, last_049.pt, last_059.pt, last_069.pt, last_079.pt), then run:

python model_soups.py 

# you will get an ensemble model in runs/train/Month/all_e10_bic10_mot10/p6_ep100_bs32_img1280_scratch, and use it to test. The test hyper-parameters should be same as the above Testing.

```

If your training process stucks, it due to bugs of the python.
Just `Ctrl+C` to stop training and resume training by:
```
python -m torch.distributed.launch --nproc_per_node 4 train.py --batch-size 32 --img 1280 1280 --data data/Chalearn_Day_all.yaml --weights 'runs/train/Day/all_e10_bic10_mot/p6_ep100_bs32_img1280_scratch/weights/last.pt' --sync-bn --device 0,1,2,3 --resume
```



## Acknowledgements

<details><summary> <b>Expand</b> </summary>

* [https://github.com/WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)

</details>
