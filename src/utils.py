import os
import sys

import pickle

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dirname = os.path.dirname(file_path)
        os.makedirs(dirname, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)   