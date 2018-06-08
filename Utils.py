import dlib
import cv2
import os

class Utils():
    @staticmethod
    def mkdir_if_not_exist(path):
        if not os.path.exists(path):
                os.mkdir(path)
