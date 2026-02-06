from aimet_onnx.quantsim import QuantizationSimModel
from aimet_common.defs import QuantScheme
from aimet_onnx import utils

import onnx
from onnxsim import simplify

def main():
    # Load simplified ONNX
    model = onnx.load('simplified.onnx')
    model_simp, check = simplify(model)
    onnx.save(model_simp, 'simplified_clean.onnx')

    # Initialize quantsim with per-channel INT8
    sim = QuantizationSimModel(
        model='simplified_clean.onnx',
        quant_scheme=QuantScheme.post_training_tf_enhanced,
        default_param_bw=8,
        default_activation_bw=8,
        per_channel=True,               # per-channel weights
        use_symmetric_encodings=True
    )

    # Apply cross-layer equalization (optional but recommended)
    sim.compute_encodings(lambda: None, forward_fn=None)  # dummy forward pass

    # Calibration with valence-aware data
    sim.compute_encodings(
        forward_fn=None,
        data_reader=ValenceCalibrationDataReader('calibration_data/high_valence_gestures/')
    )

    # AdaRound (optional – learnable rounding)
    from aimet_onnx.adaround import Adaround
    adaround_sim = Adaround.apply_adaround(
        sim.model,
        'calibration_data/high_valence_gestures/',
        num_iterations=10000,
        default_num_iterations=10000,
        default_reg_param=0.01
    )

    # Export final quantized model
    adaround_sim.export('quantized_model', 'qint8_perchannel_aimet')

    print("AIMET PTQ complete – model saved to quantized_model/")

if __name__ == '__main__':
    main()
