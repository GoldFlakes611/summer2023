'''
Name: convert.py
Description: Convert between model formats, supports torch to mobile, onnx to tflite
Date: 2023-08-25
Last Modified: 2023-08-25
'''

import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #this hides tensorflow output spam on import
import sys
import tempfile
from models import get_model

from klogs import kLogger
TAG = "CONVERT"
log = kLogger(TAG)


def torch_to_mobile(pytorch_model : str, output_mobile : str, model_name : str) -> None:
    '''
    Convert a pytorch model to a mobile model

    Args:
        pytorch_model (str): path to pytorch model
        output_mobile (str): path to output mobile model
        model_name (str): name of model to convert
    
    Returns:
        None

    Examples:
        python convert.py --torch_to_mobile -i model.pt -o model.ptl --model_name resnet34
    '''
    model_weights = torch.load(pytorch_model, map_location=torch.device('cpu'))['state']
    model = get_model(model_name)().to('cpu')
    model.load_state_dict(model_weights)
    model.eval()
    example = torch.rand(1,3,224,224)
    log.info(f"Model inference on example: {model(example)}")
    traced_script_module = torch.jit.script(model)
    traced_script_module_optimized = optimize_for_mobile(traced_script_module)
    traced_script_module_optimized._save_for_lite_interpreter(output_mobile)

def onnx_to_tflite(onnx_model : 'onnx.ModelProto', output_tflite : str) -> None:
    '''
    Convert an onnx model to a tflite model

    Args:
        onnx_model (str): path to onnx model
        output_tflite (str): path to output tflite model

    Returns:
        None

    Examples:
        python convert.py --onnx_to_tflite -i model.onnx -o model.tflite
    '''
    tf_rep = prepare(onnx_model)

    with tempfile.TemporaryDirectory() as tmp:
        tf_rep.export_graph(f"{tmp}/model")
        converter = tf.lite.TFLiteConverter.from_saved_model(f"{tmp}/model")
        # Convert the model
        tflite_model = converter.convert()

    # Save the model
    with open(output_tflite, 'wb') as f:
        f.write(tflite_model)


if __name__ == "__main__":
    '''
    Examples:
        To convert from onnx to tflite:
            python convert.py --onnx_to_tflite -i model.onnx -o model.tflite
        To convert from torch to mobile:
            python convert.py --torch_to_mobile -i model.pt -o model.ptl --model_name resnet34
    '''
    argparser = argparse.ArgumentParser(description='Convert between model formats')
    
    argparser.add_argument('-l','--level', type=str, default='info', help='Level of debug statement can be \
                           info, debug, warning, error, critical')
    argparser.add_argument('--verbose', action='store_true', help='Verbose output')

    #- ONNX to TFLite -
    argparser.add_argument('--onnx2tflite', action='store_true', help='Convert ONNX to TFLite')
    argparser.add_argument('-i', '--input', type=str, help='Path to the input model', required=True)
    argparser.add_argument('-o', '--output', type=str, help='Path to the output model', required=True)

    #- PyTorch to Pytorch Mobile -
    argparser.add_argument('--torch2mobile', action='store_true', help='Convert PyTorch to PyTorch Mobile')
    argparser.add_argument('--model_name', type=str, help='Name of the model', required="--torch2mobile" in sys.argv)

    args = argparser.parse_args()
    log.setLevel(args.level)

    if args.onnx2tflite:
        import tensorflow as tf
        from onnx_tf.backend import prepare
        import onnx
        try:
            onnx_to_tflite(onnx.load(args.input), args.output)
        except ImportError:
            log.error("Please install onnx and tensorflow packages")
            sys.exit(1)

    if args.torch2mobile:
        import torch
        from torch.utils.mobile_optimizer import optimize_for_mobile
        try:
            torch_to_mobile(args.input, args.output, args.model_name)
        except ImportError:
            log.error("Please install torch and torchvision packages")
            sys.exit(1)
