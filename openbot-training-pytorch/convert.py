import sys
import tempfile

import onnx
import tensorflow as tf
from onnx_tf.backend import prepare


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"{sys.argv[0]} input_onnx output_tflite")

    onnx_model = onnx.load(sys.argv[1])
    tf_rep = prepare(onnx_model)

    with tempfile.TemporaryDirectory() as tmp:
        tf_rep.export_graph(f"{tmp}/model")
        converter = tf.lite.TFLiteConverter.from_saved_model(f"{tmp}/model")
        # Convert the model
        tflite_model = converter.convert()

    # Save the model
    with open(sys.argv[2], 'wb') as f:
        f.write(tflite_model)
