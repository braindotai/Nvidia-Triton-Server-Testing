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
    if c == 1:
        sample_img = img.convert('L')
    else:
        sample_img = img.convert('RGB')

    resized_img = sample_img.resize((w, h), Image.BILINEAR)
    resized = np.array(resized_img)

    if resized.ndim == 2:
        resized = np.expand_dims(resized, -1)

    npdtype = triton_to_np_dtype(input_dtype)
    typed = resized.astype(npdtype)

    ordered = np.transpose(typed, (2, 0, 1))

    # if input_format == 'FORMAT_NCHW':
    #     ordered = np.transpose(typed, (2, 0, 1))
    # else:
    #     ordered = typed

    return ordered

def request_generator(inputs_files, input_name, input_dtype, input_format, max_batch_size, input_shape):
    preprocessed_images = []

    for filename in inputs_files:
        img = Image.open(filename)
        preprocessed_images.append(preprocess(img, input_format, input_dtype, *input_shape))

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
    model_name = "face_recognition"
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

    try:
        inference_start_time = time.time()

        print('[============ Running Infering ============]')

        for request_id, inputs in enumerate(request_generator(filenames, input_name, input_dtype, input_format, max_batch_size, (c, h, w))):
            request_start_time = time.time()

            print('Batch shape:', inputs[0].shape())

            response = triton_client.infer(
                model_name,
                inputs,
                request_id = str(request_id),
                model_version = model_version
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