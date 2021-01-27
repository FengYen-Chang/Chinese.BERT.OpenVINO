import os
import argparse
import time
import numpy as np
import logging as log
formatter = '[%(levelname)s] %(asctime)s %(message)s'
log.basicConfig(level=log.INFO, format=formatter)

from openvino.inference_engine import IECore
import tokenization_utils

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--device", default="CPU", type=str)
parser.add_argument("-v", "--vocab", required=True, type=str)
parser.add_argument("-m", "--model", required=True, type=str)
parser.add_argument("-i", "--input-data", required=True, type=str)

parser.add_argument("--max-seq-length", type=int, default=256)
parser.add_argument("--doc-stride", type=int, default=128)
parser.add_argument("--max-query-length", type=int, default=64)

args = parser.parse_args()


def main():
    log.info("Initializing Inference Engine")
    ie = IECore()
    version = ie.get_versions(args.device)[args.device]
    version_str = "{}.{}.{}".format(version.major, version.minor, version.build_number)
    log.info("Plugin version is {}".format(version_str))

    # read IR
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    ie_encoder = ie.read_network(model=model_xml, weights=model_bin)

    # load model to the device
    log.info("Loading model to the {}".format(args.device))
    ie_encoder_exec = ie.load_network(network=ie_encoder, device_name=args.device)

    # check input and output names
    input_names = list(ie_encoder.input_info.keys())
    output_names = list(ie_encoder.outputs.keys())
    input_info_text = "Inputs number: {}".format(len(ie_encoder.input_info.keys()))
    for input_key in ie_encoder.input_info:
        input_info_text += "\n\t- {} : {}".format(input_key, ie_encoder.input_info[input_key].input_data.shape)
    log.info(input_info_text)
    output_info_text = "Outputs number: {}".format(len(ie_encoder.outputs.keys()))
    for output_key in ie_encoder.outputs:
        output_info_text += "\n\t- {} : {}".format(output_key, ie_encoder.outputs[output_key].shape)
    log.info(output_info_text)

    # tokenization
    feature = tokenization_utils.export_feature(
        vocab_file = args.vocab, 
        data_file = args.input_data, 
        do_lower_case = False, 
        max_seq_length = args.max_seq_length, 
        doc_stride = args.doc_stride, 
        max_query_length = args.max_query_length, 
    )

    inputs = {
        input_names[0]: np.array([feature.input_ids], dtype=np.int32),
        input_names[1]: np.array([feature.input_mask], dtype=np.int32),
        input_names[2]: np.array([feature.segment_ids], dtype=np.int32),
    }

    # infer by IE
    t_start = time.perf_counter()
    res = ie_encoder_exec.infer(inputs=inputs)
    t_end = time.perf_counter()
    log.info("Inference time : {:0.2} sec".format(t_end - t_start))

    start_logits = res[output_names[0]].flatten()
    end_logits = res[output_names[1]].flatten()

    start_index = np.argmax(start_logits)
    end_index = np.argmax(end_logits)
    print(start_index, end_index)

    tok_tokens = feature.tokens[start_index:(end_index + 1)]
    print("".join(tok_tokens))

if __name__ == "__main__":
    main()
