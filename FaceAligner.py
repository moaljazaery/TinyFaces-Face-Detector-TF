import dlib
import cv2
import os

class FaceAligner():
    def __init__(self,face_size,face_padding,predictor_5_face_landmarks,out_dir=None):
        self.face_size=face_size
        self.face_padding=face_padding
        self.predictor_5_face_landmarks=predictor_5_face_landmarks
        self.out_dir=out_dir

    def align_face(self,img_path,face_rec,out_file_name):
        bgr_img = cv2.imread(img_path)
        # Convert to RGB since dlib uses RGB images
        img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        landmarks=self.predictor_5_face_landmarks(img, face_rec)
        faces = dlib.full_object_detections()
        faces.append(landmarks)
        aligned_face = dlib.get_face_chips(img, faces, size=self.face_size, padding=self.face_padding)
        aligned_face = cv2.cvtColor(aligned_face[0], cv2.COLOR_RGB2BGR)
        if self.out_dir is not None:
            to=os.path.join(self.out_dir , out_file_name)
            cv2.imwrite(to, aligned_face)
        return aligned_face