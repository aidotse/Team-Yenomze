import numpy as np
from monai.transforms import Compose


def find_first_occurance(tuple_list, target_str):
    for i, t in enumerate(tuple_list):
        if target_str in t[0]:
            return i
        
        
def split_train_val(data_list, N_valid_per_magn=1, is_val_split=True):
    indexes = [
        find_first_occurance(data_list, mag_lev)
        for mag_lev in ["20x"]#["20x", "40x", "60x"]
    ]
    indexes = [i for initial in indexes for i in range(initial, initial+N_valid_per_magn)]
    train_split = [data_list[i] for i in range(len(data_list)) if i not in indexes]
    val_split = [data_list[i] for i in indexes]
    
    if is_val_split:
        return train_split, val_split
    else:
        return train_split + val_split, val_split
    

def get_mag_level(img_file_path):
    if "20x" in img_file_path:
        return "20x"
    elif "40x" in img_file_path:
        return "40x"
    else:
        return "60x"


class MozartTheComposer(Compose):
    def __call__(self, input_):        
        vol=input_
        for t in self.transforms:
            vol = t(vol)
        return vol
    
    
def preprocess(img, mag_level, channel):
    std_dict = {"20x": {"C01": 515.0, "C02": 573.0, "C03": 254.0, "C04": 974.0}, 
                "40x": {"C01": 474.0, "C02": 513.0, "C03": 146.0, "C04": 283.0}, 
                "60x": {"C01": 379.0, "C02": 1010.0, "C03": 125.0, "C04": 228.0}}

    threshold_99_dict = {"20x": {"C01": 5.47, "C02": 4.08, "C03": 5.95, "C04": 7.28}, 
                         "40x": {"C01": 5.81, "C02": 3.97, "C03": 6.09, "C04": 7.16}, 
                         "60x": {"C01": 5.75, "C02": 3.88, "C03": 6.27, "C04": 6.81}}
    
    max_log_value_dict = {"C01": 1.92, "C02": 1.63, "C03": 1.99, "C04": 2.12}

    normalized_img = img/std_dict[mag_level][channel]
    clipped_img = np.clip(normalized_img, None, threshold_99_dict[mag_level][channel])
    log_transform_img = np.log(1 + clipped_img)
    standardized_img = log_transform_img / max_log_value_dict[channel]
    
    return standardized_img


def adjust_intensity(img, mag_level, channel):
    slope_dict = {"20x": {"C01": 1.0, "C02": 1.27, "C03": 1.1}, 
                  "40x": {"C01": 1.0, "C02": 2.39, "C03": 1.7}, 
                  "60x": {"C01": 1.0, "C02": 2.4, "C03": 0.8}}
    
    intercept_dict = {"20x": {"C01": 0.0, "C02": 14.0, "C03": 320.0}, 
                      "40x": {"C01": 0.0, "C02": -427.0, "C03": 74.0}, 
                      "60x": {"C01": 0.0, "C02": -887.0, "C03": 128.0}}
    
    adjusted_img = img * slope_dict[mag_level][channel] + intercept_dict[mag_level][channel]
        
    return adjusted_img


def postprocess(img, mag_level, channel):
    std_dict = {"20x": {"C01": 515.0, "C02": 573.0, "C03": 254.0, "C04": 974.0}, 
                "40x": {"C01": 474.0, "C02": 513.0, "C03": 146.0, "C04": 283.0}, 
                "60x": {"C01": 379.0, "C02": 1010.0, "C03": 125.0, "C04": 228.0}}

    threshold_99_dict = {"20x": {"C01": 5.47, "C02": 4.08, "C03": 5.95, "C04": 7.28}, 
                         "40x": {"C01": 5.81, "C02": 3.97, "C03": 6.09, "C04": 7.16}, 
                         "60x": {"C01": 5.75, "C02": 3.88, "C03": 6.27, "C04": 6.81}}
    
    max_log_value_dict = {"C01": 1.92, "C02": 1.63, "C03": 1.99, "C04": 2.12}
    
    log_transform_img = img * max_log_value_dict[channel]
    normalized_img = np.exp(log_transform_img - 1)
    final_img = normalized_img * std_dict[mag_level][channel]
    
    final_adjusted_img = adjust_intensity(final_img, mag_level, channel)
    
    return final_adjusted_img
