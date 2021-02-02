import os
import argparse
import time
import numpy as np
import logging as log
formatter = '[%(levelname)s] %(asctime)s %(message)s'
log.basicConfig(level=log.INFO, format=formatter)

from openvino.inference_engine import IECore
import tokenization_utils
import json

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--device", default="CPU", type=str)
parser.add_argument("-v", "--vocab", required=True, type=str)
parser.add_argument("-m", "--model", required=True, type=str)
parser.add_argument("-i", "--input-data", required=True, type=str)

parser.add_argument("--max-seq-length", type=int, default=256)
parser.add_argument("--doc-stride", type=int, default=128)
parser.add_argument("--max-query-length", type=int, default=64)
parser.add_argument("-mal", "--max_answer_length", default=30, type=int)
parser.add_argument("-nbest", "--num_of_best_set", default=10, type=int)


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
    examples, features = tokenization_utils.export_feature(
        vocab_file = args.vocab, 
        data_file = args.input_data, 
        do_lower_case = False, 
        max_seq_length = args.max_seq_length, 
        doc_stride = args.doc_stride, 
        max_query_length = args.max_query_length, 
    )

    total_examples = len(examples)
    results = {}

    for example_id in range(total_examples):
        infer_feature = []
        for _idx, _ftr in enumerate(features):
            if _ftr.example_index == example_id :
                infer_feature.append(_ftr)

        n_best_results = []

        for i, feature in enumerate(infer_feature):
            inputs = {
                input_names[0]: np.array([feature.input_ids], dtype=np.int32),
                input_names[1]: np.array([feature.input_mask], dtype=np.int32),
                input_names[2]: np.array([feature.segment_ids], dtype=np.int32),
            }

            t_start = time.perf_counter()
            res = ie_encoder_exec.infer(inputs=inputs)
            t_end = time.perf_counter()

            start_logits = res[output_names[0]].flatten()
            end_logits = res[output_names[1]].flatten()

            start_logits = start_logits - np.log(np.sum(np.exp(start_logits)))
            end_logits = end_logits - np.log(np.sum(np.exp(end_logits)))

            sorted_start_index = np.argsort(-start_logits)
            sorted_end_index = np.argsort(-end_logits)

            token_length = len(feature.tokens)

            for _s_idx in sorted_start_index[:args.num_of_best_set]:
                for _e_idx in sorted_end_index[:args.num_of_best_set]:
                    if _s_idx >= token_length:
                        continue
                    if _e_idx >= token_length:
                        continue
                    if _s_idx not in feature.token_to_orig_map:
                        continue
                    if _e_idx not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(_s_idx, False):
                        continue
                    if _e_idx < _s_idx:
                        continue
                    length = _e_idx - _s_idx + 1
                    if length > args.max_answer_length:
                        continue
                    n_best_results.append((start_logits[_s_idx] +  end_logits[_e_idx], 
                        "".join(examples[example_id].doc_tokens[feature.token_to_orig_map[_s_idx]:feature.token_to_orig_map[_e_idx] + 1])))

        max_prob = -100000
        best_result = ""
        for _res in n_best_results:
            _prob, _text = _res
            if _prob > max_prob:
                max_prob = _prob
                best_result = _text

        results[examples[example_id].qas_id] = best_result
    
    with open('openvino_output.json', 'w') as json_writer:
        json.dump(results, json_writer, indent=4, ensure_ascii=False)

    
if __name__ == "__main__":
    main()
