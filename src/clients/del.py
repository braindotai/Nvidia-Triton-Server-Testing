import os
import sys

from PIL import Image
import numpy as np
from attrdict import AttrDict

import tritonclient.grpc.model_config_pb2 as mc
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from tritonclient.utils import triton_to_np_dtype
import time


def parse_model(model_metadata, model_config):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)
    """
    if len(model_config.input) != 1:
        raise Exception(
            "expecting 1 input in model configuration, got {}".format(
                len(model_config.input)))

    input_metadata = model_metadata.inputs[0]
    input_config = model_config.input[0]
    output_metadata = model_metadata.outputs[0]

    if output_metadata.datatype != "FP32":
        raise Exception("expecting output datatype to be FP32, model '" +
                        model_metadata.name + "' output type is " +
                        output_metadata.datatype)

    # Output is expected to be a vector. But allow any number of
    # dimensions as long as all but 1 is size 1 (e.g. { 10 }, { 1, 10
    # }, { 10, 1, 1 } are all ok). Ignore the batch dimension if there
    # is one.
    output_batch_dim = (model_config.max_batch_size > 0)
    non_one_cnt = 0
    for dim in output_metadata.shape:
        if output_batch_dim:
            output_batch_dim = False
        elif dim > 1:
            non_one_cnt += 1
            if non_one_cnt > 1:
                raise Exception("expecting model output to be a vector")

    # Model input must have 3 dims, either CHW or HWC (not counting
    # the batch dimension), either CHW or HWC
    input_batch_dim = (model_config.max_batch_size > 0)
    expected_input_dims = 3 + (1 if input_batch_dim else 0)
    if len(input_metadata.shape) != expected_input_dims:
        raise Exception(
            "expecting input to have {} dimensions, model '{}' input has {}".
            format(expected_input_dims, model_metadata.name,
                   len(input_metadata.shape)))

    if type(input_config.format) == str:
        FORMAT_ENUM_TO_INT = dict(mc.ModelInput.Format.items())
        input_config.format = FORMAT_ENUM_TO_INT[input_config.format]

    if ((input_config.format != mc.ModelInput.FORMAT_NCHW) and
        (input_config.format != mc.ModelInput.FORMAT_NHWC)):
        raise Exception("unexpected input format " +
                        mc.ModelInput.Format.Name(input_config.format) +
                        ", expecting " +
                        mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NCHW) +
                        " or " +
                        mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NHWC))

    if input_config.format == mc.ModelInput.FORMAT_NHWC:
        h = input_metadata.shape[1 if input_batch_dim else 0]
        w = input_metadata.shape[2 if input_batch_dim else 1]
        c = input_metadata.shape[3 if input_batch_dim else 2]
    else:
        c = input_metadata.shape[1 if input_batch_dim else 0]
        h = input_metadata.shape[2 if input_batch_dim else 1]
        w = input_metadata.shape[3 if input_batch_dim else 2]

    return (model_config.max_batch_size, input_metadata.name,
            output_metadata.name, c, h, w, input_config.format,
            input_metadata.datatype)

def preprocess(img, format, dtype, c, h, w):
    """
    Pre-process an image to meet the size, type and format
    requirements specified by the parameters.
    """
    if c == 1:
        sample_img = img.convert('L')
    else:
        sample_img = img.convert('RGB')

    resized_img = sample_img.resize((w, h), Image.BILINEAR)
    resized = np.array(resized_img)
    if resized.ndim == 2:
        resized = resized[:, :, np.newaxis]

    npdtype = triton_to_np_dtype(dtype)
    typed = resized.astype(npdtype)

    # if scaling == 'face_recognition':
    #     scaled = (typed / 127.5) - 1
    # elif scaling == 'VGG':
    #     if c == 1:
    #         scaled = typed - np.asarray((128,), dtype=npdtype)
    #     else:
    #         scaled = typed - np.asarray((123, 117, 104), dtype=npdtype)
    # else:
    #     scaled = typed
    if format == mc.ModelInput.FORMAT_NCHW:
        ordered = np.transpose(typed, (2, 0, 1))
    else:
        ordered = typed
    # ordered = scaled

    # Channels are in RGB order. Currently model configuration data
    # doesn't provide any information as to other channel orderings
    # (like BGR) so we just assume RGB.
    return ordered


def requestGenerator(batched_image_data, input_name, output_name, dtype):
    client = httpclient
    # Set the input data
    inputs = [client.InferInput(input_name, batched_image_data.shape, dtype)]
    inputs[0].set_data_from_numpy(batched_image_data)
    model_name = "face_recognition"
    model_version = "1"

    yield inputs, model_name, model_version


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
        print("[========= Triton client initialization failed =========]")
        print(e)
        sys.exit(1)

    try:
        model_metadata = triton_client.get_model_metadata(
            model_name = model_name,
            model_version = model_version
        )
    except InferenceServerException as e:
        print("[========= Failed to retrieve the model metadata =========]")
        print(e)
        sys.exit(1)

    try:
        model_config = triton_client.get_model_config(
            model_name = model_name,
            model_version = model_version
        )
    except InferenceServerException as e:
        print("[========= Failed to retrieve the model config =========]")
        print(e)
        sys.exit(1)
    
    def convert_http_metadata_config(_metadata, _config):
        _model_metadata = AttrDict(_metadata)
        _model_config = AttrDict(_config)

        return _model_metadata, _model_config
        
    model_metadata, model_config = convert_http_metadata_config(model_metadata, model_config)
    print(model_metadata.inputs)
# 
    # def convert_http_metadata_config(_metadata, _config):
    #     _model_metadata = AttrDict(_metadata)
    #     _model_config = AttrDict(_config)

    #     return _model_metadata, _model_config

    # model_metadata, model_config = convert_http_metadata_config(model_metadata, model_config)

    # print(model_metadata.inputs)
    
    max_batch_size, input_name, output_name, c, h, w, format, dtype = parse_model(model_metadata, model_config)

    # filenames = []
    # if os.path.isdir(image_filename):
    #     filenames = [
    #         os.path.join(image_filename, f)
    #         for f in os.listdir(image_filename)
    #         if os.path.isfile(os.path.join(image_filename, f))
    #     ] * 256
    # else:
    #     filenames = [
    #         image_filename,
    #     ]

    # filenames.sort()

    # image_data = []
    # for filename in filenames:
    #     img = Image.open(filename)
    #     image_data.append(preprocess(img, format, dtype, c, h, w))

    # responses = []
    # sent_count = 0
    # last_request = False
    # batch_size = 128
    # image_idx = 0

    # while not last_request:
    #     input_filenames = []
    #     repeated_image_data = []

    #     for idx in range(batch_size):
    #         input_filenames.append(filenames[image_idx])
    #         repeated_image_data.append(image_data[image_idx])
    #         image_idx = (image_idx + 1) % len(image_data)
    #         if image_idx == 0:
    #             last_request = True

    #     if max_batch_size > 0:
    #         batched_image_data = np.stack(repeated_image_data, axis=0)
    #     else:
    #         batched_image_data = repeated_image_data[0]

    #     try:
    #         start_time = time.time()
    #         for inputs, model_name, model_version in requestGenerator(batched_image_data, input_name, output_name, dtype):
    #             sent_count += 1
    #             print('Infering.................')
    #             responses.append(triton_client.infer(model_name,
    #                                                 inputs,
    #                                                 request_id=str(sent_count),
    #                                                 model_version=model_version))
    #             print('done..........')
    #         end_time = time.time()
        
    #     except InferenceServerException as e:
    #             print("inference failed: " + str(e))
    #             sys.exit(1)

    # for response in responses:
    #     this_id = response.get_response()["id"]
    #     # print(response.as_numpy(output_name))
    #     print("Request {}, batch size {}".format(this_id, batch_size))
    #     print(response.as_numpy(output_name).shape)
    #     print(end_time-start_time)


if __name__ == "__main__":
    main()