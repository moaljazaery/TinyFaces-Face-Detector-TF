from TinyFacesDetector import TinyFacesDetector
import dlib
from Utils import Utils
from FaceAligner import FaceAligner

faces_out_folder = "./output/"
image_path="sample.jpg"
model_pkl="weights.pkl"

Utils.mkdir_if_not_exist(faces_out_folder)

tiny_faces_detector = TinyFacesDetector(model_pkl,use_gpu=True)

#init the face landmarks detector
predictor_5_face_landmarks = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")

# tight face aligner : padding = 0.2
aligner_tight = FaceAligner(face_size=112, face_padding=0.2, predictor_5_face_landmarks=predictor_5_face_landmarks)



face_rects=tiny_faces_detector.detect(image_path,nms_thresh=0.1,prob_thresh=0.5,min_conf=0.9)
face_indx=0
for rect in face_rects:
    face_indx+=1
    aligner_tight.out_dir= faces_out_folder
    aligner_tight.align_face(image_path,rect,str(face_indx)+'.jpg')
