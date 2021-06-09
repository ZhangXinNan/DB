
# 0 prepare
sudo apt install gcc g++

python setup.py build_ext --inplace

pip install opencv-python 


# 1 demo
CUDA_VISIBLE_DEVICES=0 python demo.py \
    experiments/seg_detector/totaltext_resnet18_deform_thre.yaml \
    --image_path datasets/total_text/test_images/img10.jpg \
    --resume model/totaltext_resnet18 \
    --polygon --box_thresh 0.7 --visualize


CUDA_VISIBLE_DEVICES=0 python demo.py \
    experiments/seg_detector/ic15_resnet18_deform_thre.yaml \
    --image_path datasets/icdar2015/test_images/img_10.jpg \
    --resume models/ic15_resnet18 \
    --polygon --box_thresh 0.7 --visualize

# 2 Evaluate the performance

CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/totaltext_resnet18_deform_thre.yaml --resume models/totaltext_resnet18 --polygon --box_thresh 0.7

CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/totaltext_resnet50_deform_thre.yaml --resume models/totaltext_resnet50 --polygon --box_thresh 0.6

CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/td500_resnet18_deform_thre.yaml --resume models/td500_resnet18 --box_thresh 0.5

CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/td500_resnet50_deform_thre.yaml --resume models/td500_resnet50 --box_thresh 0.5

# short side 736, which can be changed in base_ic15.yaml
CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/ic15_resnet18_deform_thre.yaml --resume models/ic15_resnet18 --box_thresh 0.55

[INFO] [2021-06-09 17:46:33,076] precision : 0.877384 (500)
[INFO] [2021-06-09 17:46:33,076] recall : 0.775156 (500)
[INFO] [2021-06-09 17:46:33,076] fmeasure : 0.823108 (1)


# short side 736, which can be changed in base_ic15.yaml
CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/ic15_resnet50_deform_thre.yaml --resume models/ic15_resnet50 --box_thresh 0.6

[INFO] [2021-06-09 17:50:48,104] precision : 0.912924 (500)
[INFO] [2021-06-09 17:50:48,105] recall : 0.802600 (500)
[INFO] [2021-06-09 17:50:48,105] fmeasure : 0.854215 (1)


# short side 1152, which can be changed in base_ic15.yaml
CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/ic15_resnet50_deform_thre.yaml --resume models/ic15_resnet50 --box_thresh 0.6

[INFO] [2021-06-09 17:56:28,652] precision : 0.907388 (500)
[INFO] [2021-06-09 17:56:28,652] recall : 0.839673 (500)
[INFO] [2021-06-09 17:56:28,652] fmeasure : 0.872218 (1)


## ValueError: cannot reshape array of size 1 into shape (4,2)


```python
 for line in data['lines']:
            texts.append(line['text'])
            for p in line['poly']:
                keypoints.append(imgaug.Keypoint(p[0], p[1]))

        keypoints = aug.augment_keypoints([imgaug.KeypointsOnImage(keypoints=keypoints, shape=shape)])[0].keypoints
        new_polys = np.array([p.x, p.y] for p in keypoints).reshape([-1, 4, 2])
```
-->>
```python
        new_polys = []

        for line in data['lines']:
            texts.append(line['text'])
            new_poly = []
            for p in line['poly']:
                new_poly.append((p[0], p[1]))
                keypoints.append(imgaug.Keypoint(p[0], p[1]))
            new_polys.append(new_poly)

        if not self.only_resize:        
            keypoints = aug.augment_keypoints([imgaug.KeypointsOnImage(keypoints=keypoints, shape=shape)])[0].keypoints
            new_polys = np.array([[p.x, p.y] for p in keypoints]).reshape((-1, 4, 2))
```

# 3 Training


CUDA_VISIBLE_DEVICES=0 python train.py experiments/seg_detector/ic15_resnet18_deform_thre.yaml --num_gpus 1



