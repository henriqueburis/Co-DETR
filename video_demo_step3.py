# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import cv2
import mmcv
import torch

from pathlib import Path
import matplotlib.pyplot as plt
from functools import reduce
import numpy as np

from ultralytics import SAM
from ultralytics.models.sam import Predictor as SAMPredictor

from mmdet.apis import inference_detector, init_detector
#from projects import *


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection video demo step3')
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

    # Display model information (optional)
    #model_sam.info()

    model = init_detector(args.config, args.checkpoint, device=args.device)

    chosen_class_id = 2 # Defina o ID da classe do objeto desejado apos olha os pessos do arquivo (por exemplo, carro)

    checkpoint = torch.load(args.checkpoint)
    print(checkpoint['meta']) # restorna todas informaçoes armazenado dentro do modelo salvo, para então pode escolher a classe
    print(checkpoint['meta']['CLASSES'])
    #('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
    print(checkpoint['meta']['CLASSES'][chosen_class_id]) ## Classe carro

    video_reader = mmcv.VideoReader(args.video)
    video_writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            args.out, fourcc, video_reader.fps,
            (video_reader.width, video_reader.height))
        

    count_car = 0  # Inicializa a contagem de carros
    bbox_list = []  # Inicializa uma lista para armazenar as BBoxes dos resultados selecionados

    for frame in mmcv.track_iter_progress(video_reader):
        result = inference_detector(model, frame)
        # Filtrar apenas as detecções do objeto desejado (carro)

        if isinstance(result, tuple):
            bbox_result, segm_result = result
            print(bbox_result)
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None 
            bboxes = np.vstack(bbox_result)

        labels = [np.full(res.shape[0], i, dtype=np.int32) for i, res in enumerate(result)]
        #labels [ 0  0  0  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2 2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2 2  7  7  7 13 13 28 56 56]
        r = [np.full(res.shape[2], i, dtype=np.int32) for i, res in enumerate(result)]

        filtered_labels = [x for x in labels if x == chosen_class_id]
        
        result = [detection for detection in result[0] if detection[4] == chosen_class_id]

         # Armazena as BBoxes dos resultados selecionados na lista
        bbox_list.extend([detection[:4] for detection in result])

        # Adiciona o número de detecções de carros ao contador
        count_car += len(result)

        res = model_sam(frame, bboxes=bbox_list.numpy())

        # Obtém as máscaras de segmentação a partir do resultado SAM na documentação do modelo tem todas as keys.
        masks = res['segmentation']

        # Adiciona sobreposições visuais das máscaras no quadro original
        for mask in masks:
            frame = cv2.addWeighted(frame, 1, mask, 0.5, 0) ###<<<<<<<<<<<<<<<<<<<<<==== muito lento, mas prova de conceito ok

        frame = model.show_result(frame, result, score_thr=args.score_thr)
        if args.show:
            cv2.namedWindow('video', 0)
            mmcv.imshow(frame, 'video', args.wait_time)
        if args.out:
            video_writer.write(frame)

    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()


    print(f"Total de carros detectados: {count_car}")


if __name__ == '__main__':
    main()
