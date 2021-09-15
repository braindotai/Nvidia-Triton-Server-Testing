import os
import onnxruntime
from typing import *
import numpy as np
from PIL import Image
import time

def preprocess(img, c = 3, h = 112, w = 112):
    if c == 1:
        sample_img = img.convert('L')
    else:
        sample_img = img.convert('RGB')

    resized_img = sample_img.resize((w, h), Image.BILINEAR)
    resized = np.array(resized_img)

    if resized.ndim == 2:
        resized = np.expand_dims(resized, -1)

    typed = resized.astype('float32')

    ordered = np.transpose(typed, (2, 0, 1))
    
    return ordered

max_batch_size = 128

def request_generator(inputs_files):
    preprocessed_images = []

    for filename in inputs_files:
        img = Image.open(filename)
        preprocessed_images.append(preprocess(img))

        if len(preprocessed_images) % max_batch_size == 0:
            input_batch = np.stack(preprocessed_images)

            del preprocessed_images
    
            preprocessed_images = []
            
            yield input_batch

def inference(session: onnxruntime.InferenceSession, input_batch: np.ndarray) -> np.ndarray:
	'''
	Arguments:
		session		: onnxruntime.InferenceSession -> Onnx session
		images		: Union[List[np.ndarray], np.ndarray] -> Shape = (N, 128, 128, 3)
	'''
	return session.run(None, {'input': input_batch})[0]

image_filename = "./src/inputs"

filenames = []
if os.path.isdir(image_filename):
    filenames = [
        os.path.join(image_filename, f)
        for f in os.listdir(image_filename)
        if os.path.isfile(os.path.join(image_filename, f))
    ] * 1300
else:
    filenames = [
        image_filename,
    ]
filenames.sort()

inference_start_time = time.time()

session = onnxruntime.InferenceSession('./src/models/face_recognition/1/model.onnx')

print('[============ Running Infering ============]')

for request_id, inputs in enumerate(request_generator(filenames)):
    request_start_time = time.time()

    print('Batch shape:', inputs.shape)

    request_output = inference(session, inputs)

    request_end_time = time.time()

    print(f'Request done in {request_end_time - request_start_time}\n')

inference_end_time = time.time()

print(f'\nInference done in {inference_end_time - inference_start_time} seconds\n')