import sys
import argparse
import logging
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)


import torch
from datetime import datetime


current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Create the main parser
parser = argparse.ArgumentParser(description="CCViM Pytorch Inference")
parser.add_argument('--gpu', default='1', help='GPU list')
parser.add_argument('--nr_types', type=int, default=None, help='Number of nuclei types to predict')
parser.add_argument('--model_path', default='', help='Path to saved checkpoint')
parser.add_argument('--model_mode', default='fast', choices=['original', 'fast'], help='Model mode: original or fast')
parser.add_argument('--nr_inference_workers', type=int, default=8, help='Number of workers during inference')
parser.add_argument('--nr_post_proc_workers', type=int, default=16, help='Number of workers during post-processing')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size per GPU')
parser.add_argument('--input_dir', default="/cpm17/test/Images", help='Path to input data directory')
parser.add_argument('--output_dir', default= f'/CCViM-nuclei/CPM17/infer_output/CCM_UNET_{current_time}', help='Path to output directory')
parser.add_argument('--mem_usage', type=float, default=0.2, help='Memory (physical + swap) to be used for caching')
parser.add_argument('--draw_dot', action='store_true', help='Draw nuclei centroid on overlay')
parser.add_argument('--save_qupath', action='store_true', help='Save QuPath v0.2.3 compatible format')
parser.add_argument('--save_raw_map', action='store_true', default=True,help='Save raw prediction')


# Parse the arguments
args = parser.parse_args()


logging.basicConfig(
    level=logging.INFO,
    format='|%(asctime)s.%(msecs)03d| [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d|%H:%M:%S',
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)
method_args = {
    'method': {
        'model_args': {
            'nr_types': args.nr_types,
        },
        'model_path': args.model_path,
    },
    'type_info_path': None if args.type_info_path == '' \
        else args.type_info_path,
}

# ***
run_args = {
    'batch_size': int(args.batch_size),
    'mem_usage': args.mem_usage,
    'input_dir': args.input_dir,
    'output_dir': args.output_dir,
    'draw_dot': args.draw_dot,
    'patch_input_shape': int(256),
    'patch_output_shape': int(164),
    'save_raw_map': args.save_raw_map,
    'save_qupath': int(args.save_qupath),
    'nr_post_proc_workers': int(args.nr_post_proc_workers),
    'nr_inference_workers': int(args.nr_inference_workers),
}
nr_gpus = torch.cuda.device_count()
logging.info('Detect #GPUS: %d' % nr_gpus)

print(f'***input_dir {args.input_dir}')
# Process tile using InferManager
if __name__ == '__main__':
    print(f'papaent {parent_dir}')
    from infer.tile import InferManager

    infer = InferManager(**method_args)
    infer.process_file_list(run_args)
