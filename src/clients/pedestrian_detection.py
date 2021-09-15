import os
import sys

import numpy as np
from PIL import Image
from attrdict import AttrDict

import tritonclient.grpc.model_config_pb2 as mc
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from tritonclient.utils import triton_to_np_dtype

import time

import onnxruntime
import torch, torchvision
import cv2
import numpy as np
import argparse

def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def clip_coords(boxes, shape):
    if isinstance(boxes, torch.Tensor): 
        boxes[:, 0].clamp_(0, shape[1])  
        boxes[:, 1].clamp_(0, shape[0])  
        boxes[:, 2].clamp_(0, shape[1])  
        boxes[:, 3].clamp_(0, shape[0])  
    else:
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])

def scale_coords(img1_shape, coords, img0_shape, ratio_pad = None):
    if ratio_pad is None:  
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  
    coords[:, [1, 3]] -= pad[1]  
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def letterbox(im, new_shape = (640, 640), color = (114, 114, 114), auto = True, scaleFill = False, scaleup = True, stride = 32):
    shape = im.shape[:2]
    new_shape = (640, 640)

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  
    elif scaleFill:  
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  

    dw /= 2  
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation = cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value = color) 

    return im

def box_iou(box1, box2):

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)

def non_max_suppression(
    prediction,
    conf_thres = 0.25,
    iou_thres = 0.45,
    classes = None,
    agnostic = False,
    multi_label = False,
    labels = (),
    max_det = 300
):
    prediction = torch.tensor(prediction)

    nc = prediction.shape[2] - 5
    xc = prediction[..., 4] > conf_thres

    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    min_wh, max_wh = 2, 4096
    max_nms = 30000
    time_limit = 10.0
    redundant = True
    multi_label &= nc > 1
    merge = False

    t = time.time()
    output = [torch.zeros((0, 6), device = prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]

        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device = x.device)
            v[:, :4] = l[:, 1:5]
            v[:, 4] = 1.0
            v[range(len(l)), l[:, 0].long() + 5] = 1.0
            x = torch.cat((x, v), 0)

        if not x.shape[0]:
            continue

        x[:, 5:] *= x[:, 4:5]

        box = xywh2xyxy(x[:, :4])

        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        c = x[:, 5:6] * (0 if agnostic else max_wh) 
        boxes, scores = x[:, :4] + c, x[:, 4] 
        
        i = torchvision.ops.nms(boxes, scores, iou_thres)

        if i.shape[0] > max_det:
            i = i[:max_det]
        if merge and (1 < n < 3E3): 
            iou = box_iou(boxes[i], boxes) > iou_thres  
            weights = iou * scores[None]  
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True) 
            if redundant:
                i = i[iou.sum(1) > 1]  

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break 

    return output

def parse_model(model_metadata, model_config):
    input_metadata = model_metadata.inputs[0]
    input_config = model_config.input[0]
    output_metadata = model_metadata.outputs[0]
    
    input_batch_dim = model_config.max_batch_size > 0

    if input_config.format == mc.ModelInput.FORMAT_NHWC:
        h = input_metadata.shape[1 if input_batch_dim else 0]
        w = input_metadata.shape[2 if input_batch_dim else 1]
        c = input_metadata.shape[3 if input_batch_dim else 2]
    else:
        c = input_metadata.shape[1 if input_batch_dim else 0]
        h = input_metadata.shape[2 if input_batch_dim else 1]
        w = input_metadata.shape[3 if input_batch_dim else 2]

    return (
        model_config.max_batch_size,
        input_metadata.name,
        output_metadata.name,
        c,
        h,
        w,
        input_config.format,
        input_metadata.datatype
    )

def preprocess(img, input_format, input_dtype, c, h, w):
    img = letterbox(img, (h, w), stride = 64, auto = False)

    img = np.ascontiguousarray(img)

    img = img / 255.0

    if img.ndim == 2:
        img = np.expand_dims(img, -1)

    typed = img.astype(triton_to_np_dtype(input_dtype))

    preprocessed_image = np.transpose(typed, (2, 0, 1))

    return preprocessed_image

def postprocess(original_images, outputs, conf_thres = 0.25, iou_thres = 0.45, agnostic_nms = False, max_det = 300):
    outputs = non_max_suppression(outputs, conf_thres, iou_thres, [0], agnostic_nms, max_det = max_det)

    for idx, (original_image, output) in enumerate(zip(original_images, outputs)):
        if len(output):
            output[:, :4] = scale_coords((640, 640), output[:, :4], original_image.shape).round()

            for *xyxy, _, _ in reversed(output):
                c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                cv2.rectangle(original_image, c1, c2, [0, 0, 255], thickness = 3, lineType = cv2.LINE_AA)
            
        cv2.imwrite(f"outputs/{idx}.jpg", cv2.cvtColor(original_image, 4))

        # yield original_image

def request_generator(input_images, input_name, input_dtype, input_format, max_batch_size, input_shape):
    preprocessed_images = []

    for image in input_images:
        preprocessed_images.append(preprocess(image, input_format, input_dtype, *input_shape))

        if len(preprocessed_images) % max_batch_size == 0:
            input_batch = np.stack(preprocessed_images)

            inputs = [httpclient.InferInput(input_name, input_batch.shape, input_dtype)]
            inputs[0].set_data_from_numpy(input_batch)

            del preprocessed_images
    
            preprocessed_images = []
            
            yield inputs

    if len(preprocessed_images) != 0:
        input_batch = np.stack(preprocessed_images)

        inputs = [httpclient.InferInput(input_name, input_batch.shape, input_dtype)]
        inputs[0].set_data_from_numpy(np.stack(preprocessed_images))
        
        del preprocessed_images
        
        preprocessed_images = []
        
        yield inputs


def main():
    url = "localhost:8000"
    verbose = False
    concurrency = 10
    image_filename = "./src/inputs"
    model_name = "pedestrian_detection"
    model_version = "1"

    try:
        triton_client = httpclient.InferenceServerClient(
            url = url, 
            verbose = verbose, 
            concurrency = concurrency
        )
    except Exception as e:
        print("\n[========= Triton client initialization failed =========]")
        print(e, '\n')
        sys.exit(1)

    try:
        model_metadata = triton_client.get_model_metadata(
            model_name = model_name,
            model_version = model_version
        )
    except InferenceServerException as e:
        print("\n[========= Failed to retrieve the model metadata =========]")
        print(e, '\n')
        sys.exit(1)

    try:
        model_config = triton_client.get_model_config(
            model_name = model_name,
            model_version = model_version
        )
    except InferenceServerException as e:
        print("\n[========= Failed to retrieve the model config =========]")
        print(e, '\n')
        sys.exit(1)
        
    model_metadata, model_config = AttrDict(model_metadata), AttrDict(model_config)

    max_batch_size, input_name, output_name, c, h, w, input_format, input_dtype = parse_model(model_metadata, model_config)

    print("[============= Parsed Model ==============]")
    print(f'max_batch_size: {max_batch_size}')
    print(f'input_name: {input_name}')
    print(f'output_name: {output_name}')
    print(f'c: {c}')
    print(f'h: {h}')
    print(f'w: {w}')
    print(f'input_format: {input_format}')
    print(f'input_dtype: {input_dtype}\n')
    

    if os.path.isdir(image_filename):
        input_images = [
            np.array(Image.open(os.path.join(image_filename, f)).convert('RGB'))
            for f in os.listdir(image_filename)
            if os.path.isfile(os.path.join(image_filename, f))
        ]

    print(f'Total {len(input_images)} images to run inference for...\n')
    
    try:
        inference_start_time = time.time()

        print('[============ Running Infering ============]')

        for request_id, inputs in enumerate(request_generator(input_images, input_name, input_dtype, input_format, max_batch_size, (c, h, w))):
            print('Batch shape:', inputs[0].shape())

            request_start_time = time.time()

            response = triton_client.infer(
                model_name,
                inputs,
                request_id = str(request_id),
                model_version = model_version
            )

            postprocess(
                input_images,
                response.as_numpy(output_name),
                conf_thres = 0.28, 
                iou_thres = 0.45,
                max_det = 1000,
                agnostic_nms = False,
            )

            request_end_time = time.time()

            print(f'Request done in {request_end_time - request_start_time}\n')

        inference_end_time = time.time()

        print(f'\nInference done in {inference_end_time - inference_start_time} seconds\n')
    
    except InferenceServerException as e:
        print("\n[========= Failed to retrieve the model config =========]")
        print(e, '\n')
        sys.exit(1)

if __name__ == "__main__":
    main()