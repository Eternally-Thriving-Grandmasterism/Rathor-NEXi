import numpy as np
import json

class ValenceCalibrationDataReader:
    def __init__(self, calibration_files):
        self.calibration_files = calibration_files
        self.iterator = iter(calibration_files)

    def get_next(self):
        try:
            file_path = next(self.iterator)
            with open(file_path, 'rb') as f:
                data = np.load(f)
            return {'input': data}
        except StopIteration:
            return None
