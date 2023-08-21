import argparse
import sys
import tempfile

import onnx
import tensorflow as tf
from onnx_tf.backend import prepare


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Convert ONNX to TFLite')
    argparser.add_argument('input_onnx', type=str, help='Path to the ONNX model')
    argparser.add_argument('output_tflite', type=str, help='Path to the TFLite model')
    args = argparser.parse_args()

    onnx_model = onnx.load(args.input_onnx)
    tf_rep = prepare(onnx_model)

    with tempfile.TemporaryDirectory() as tmp:
        tf_rep.export_graph(f"{tmp}/model")
        converter = tf.lite.TFLiteConverter.from_saved_model(f"{tmp}/model")
        # Convert the model
        tflite_model = converter.convert()

    # Save the model
    with open(args.output_tflite, 'wb') as f:
        f.write(tflite_model)
