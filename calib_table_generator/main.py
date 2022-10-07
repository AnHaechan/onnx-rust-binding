from email.message import Message
import onnxruntime as ort
import onnxruntime.quantization as quant
import numpy as np
import onnx
import os

def get_input_name_shape(model_path):
    model = onnx.load(model_path)
    initializers =  [node.name for node in model.graph.initializer]
    inputs = []
    for node in model.graph.input:
        if node.name not in initializers:
            inputs.append(node)
    input_name = inputs[0].name
    input_shape_protobuf = inputs[0].type.tensor_type.shape.dim
    input_shape = [d.dim_value for d in input_shape_protobuf]
    return (input_name, input_shape)

class CSPDarkNet53DataReader(quant.CalibrationDataReader):
    def __init__(self, input_name, input_shape):
        self.input_name = input_name
        self.input_shape = input_shape

        self.num_inputs = 100
        self.curr_input_idx = 0

    def get_next(self):
        if self.curr_input_idx < self.num_inputs:
            self.curr_input_idx += 1
            NHWC_data = np.float32(np.random.random(self.input_shape))
            return {self.input_name: NHWC_data}
        else:
            return None

if __name__ == "__main__":
    model_path = "CSPDarkNet53_f32.onnx"
    input_name, input_shape = get_input_name_shape(model_path)
    calibrator = quant.create_calibrator(model_path)
    calibrator.set_execution_providers(["CUDAExecutionProvider"])
    data_reader = CSPDarkNet53DataReader(input_name, input_shape)
    calibrator.collect_data(data_reader)

    os.chdir("CSPDarkNet53")
    quant.write_calibration_table(calibrator.compute_range())
    os.chdir("..")