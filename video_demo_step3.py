# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import cv2
import mmcv
import torch
import numpy as np

from ultralytics import SAM
from ultralytics.models.sam import Predictor as SAMPredictor

from mmdet.apis import inference_detector, init_detector
#from projects import *


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection video demo step1')
    parser.add_argument('video', help='Video file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    parser.add_argument('--out', type=str, help='Output video file')
    parser.add_argument('--show', action='store_true', help='Show video')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=1,
        help='The interval of show (s), 0 is block')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.out or args.show, \
        ('Please specify at least one operation (save/show the '
         'video) with the argument "--out" or "--show"')
    
    # Load a model sam
    model_sam = SAM('sam_b.pt')

    model = init_detector(args.config, args.checkpoint, device=args.device)

    chosen_class_id = 2 # Defina o ID da classe do objeto desejado apos olha os pessos do arquivo (por exemplo, carro)

    checkpoint = torch.load(args.checkpoint)
    print(checkpoint['meta']) # restorna todas informaçoes armazenado dentro do modelo salvo, para então pode escolher a classe
    print(checkpoint['meta']['CLASSES'])
    #('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
    print(checkpoint['meta']['CLASSES'][chosen_class_id]) ## Classe carro

    Classes_ = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


    video_reader = mmcv.VideoReader(args.video)
    video_writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            args.out, fourcc, video_reader.fps,
            (video_reader.width, video_reader.height))
        

    for frame in mmcv.track_iter_progress(video_reader):
        result = inference_detector(model, frame)
        image_copy = frame.copy()
        # Filtrar apenas as detecções do objeto desejado (carro)
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            print(bbox_result)
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None 
            bboxes = np.vstack(bbox_result)

        labels = [np.full(res.shape[0], i, dtype=np.int32) for i, res in enumerate(bbox_result)]
        labels = np.concatenate(labels)

        count_car = np.count_nonzero(labels == chosen_class_id)

        for labels_, box in zip(labels,bboxes):

            if(labels_ == chosen_class_id):
                x1, y1, x2, y2, z = map(int, box)
                res = model_sam(frame, bboxes=box[0:4].numpy())
                masks = res['segmentation']
                cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 1) 
                cv2.putText(image_copy, Classes_[labels_], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


                # Sobreponha a máscara na imagem
                for mask in masks:
                    mask = np.array(mask, dtype=np.uint8)
                    mask = cv2.resize(mask, (x2 - x1, y2 - y1))
                    roi = image_copy[y1:y2, x1:x2]

                    # Aplique a máscara apenas na região da caixa delimitada
                    result = cv2.addWeighted(roi, 1, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.5, 0)
                    image_copy[y1:y2, x1:x2] = result

    print(f"Total de carros detectados por frame: {count_car}")
    #cv2_imshow(image_copy)
    cv2.imshow('Detecções Faster R-CNN', image_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
