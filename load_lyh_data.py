import os
import numpy as np

def read_folder_npy_data(folder_path=None, selected_file=None):
    '''
    加载数据
    folder_path: 数据文件所在目录
    selected_file: 需要的数据文件
    return---
    data_dict, 字典类型, key为文件名, value为相应内容
    '''

    data_dict = {}

    if not folder_path:
        folder_path = os.getcwd() + "/Emotiv_dataloader-main/data"
    if not selected_file:
        selected_file = ["nothing1_orginal.npy", "nothing2_orginal.npy", "nothing3_orginal.npy", \
                        "left1_orginal.npy", "left2_orginal.npy", "left3_orginal.npy",
                        "right1_orginal.npy", "right2_orginal.npy", "right3_orginal.npy",
                        "leg1_orginal.npy", "leg2_orginal.npy", "leg3_orginal.npy"]

    # Load .npy files
    for filename in selected_file:
        file_path = os.path.join(folder_path, filename)
        
        try:
            # Attempt to load the file
            content = np.load(file_path, allow_pickle=True)

            # Print information about the loaded file
            print(f"Loaded {filename}: Shape={content.shape}, Dtype={content.dtype}")
            
            # Add to the dictionary
            data_dict[filename] = content
        
        except Exception as e:
            # Print an error message if loading fails
            print(f"Error loading {filename}: {e}")

    return data_dict