import SimpleITK
import numpy as np
import cv2
from pandas import DataFrame
from pathlib import Path
from scipy.ndimage import center_of_mass, label
from pathlib import Path
from evalutils import DetectionAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    DataFrameValidator,
)
from typing import (Tuple)
from evalutils.exceptions import ValidationError
import random
import json

from ultralytics import YOLO

####
# Toggle the variable below to debug locally. The final container would need to have execute_in_docker=True
####
execute_in_docker = True


class VideoLoader():
    def load(self, *, fname):
        path = Path(fname)
        print('File found: ' + str(path))
        if ((str(path)[-3:])) == 'mp4':
            if not path.is_file():
                raise IOError(
                    f"Could not load {fname} using {self.__class__.__qualname__}."
                )
                #cap = cv2.VideoCapture(str(fname))
            #return [{"video": cap, "path": fname}]
            return [{"path": fname}]

# only path valid
    def hash_video(self, input_video):
        pass


class UniqueVideoValidator(DataFrameValidator):
    """
    Validates that each video in the set is unique
    """

    def validate(self, *, df: DataFrame):
        try:
            hashes = df["video"]
        except KeyError:
            raise ValidationError("Column `video` not found in DataFrame.")

        if len(set(hashes)) != len(hashes):
            raise ValidationError(
                "The videos are not unique, please submit a unique video for "
                "each case."
            )

class Surgtoolloc_det(DetectionAlgorithm):
    def __init__(self):
        super().__init__(
            index_key='input_video',
            file_loaders={'input_video': VideoLoader()},
            input_path=Path("/input/") if execute_in_docker else Path("./test/"),
            output_file=Path("/output/surgical-tools.json") if execute_in_docker else Path(
                            "./output/surgical-tools.json"),
            validators=dict(
                input_video=(
                    #UniqueVideoValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )
        
        ###                                                                                                     ###
        ###  TODO: adapt the following part for creating your model and loading weights
        ###                                                                                                     ###
        self.model = YOLO("bestl.pt")
        
        self.tool_list = ['needle_driver','monopolar_curved_scissor', 'force_bipolar', 'clip_applier', 'cadiere_forceps', 'bipolar_forceps'
                         , 'vessel_sealer', 'permanent_cautery_hook_spatula', 'prograsp_forceps', 'stapler'
                         , 'grasping_retractor', 'tip_up_fenestrated_grasper']
        
        # self.tool_list = ['needle_driver','monopolar_curved_scissors', 'force_bipolar', 'clip_applier', 'cadiere_forceps', 'bipolar_forceps'
        #                  , 'vessel_sealer', 'permanent_cautery_hook/spatula', 'prograsp_forceps', 'stapler'
        #                  , 'grasping_retractor', 'tip-up_fenestrated_grasper']

    def process_case(self, *, idx, case):
        # Input video would return the collection of all frames (cap object)
        input_video_file_path = case #VideoLoader.load(case)
        # Detect and score candidates
        scored_candidates = self.predict(case.path) #video file > load evalutils.py

        # Write resulting candidates to result.json for this case
        return dict(type="Multiple 2D bounding boxes", boxes=scored_candidates, version={"major": 1, "minor": 0})

    def save(self):
        with open(str(self._output_file), "w") as f:
            json.dump(self._case_results[0], f)

    # def generate_bbox(self, frame_id):
    #     # bbox coordinates are the four corners of a box: [x, y, 0.5]
    #     # Starting with top left as first corner, then following the clockwise sequence
    #     # origin is defined as the top left corner of the video frame
    #     num_predictions = 2
    #     predictions = []
    #     for n in range(num_predictions):
    #         name = f'slice_nr_{frame_id}_' + self.tool_list[n]
    #         bbox = [[54.7, 95.5, 0.5],
    #                 [92.6, 95.5, 0.5],
    #                 [92.6, 136.1, 0.5],
    #                 [54.7, 136.1, 0.5]]
    #         score = np.random.rand()
    #         prediction = {"corners": bbox, "name": name, "probability": score}
    #         predictions.append(prediction)
    #     return predictions

    def predict(self, fname) -> DataFrame:
        """
        Inputs:
        fname -> video file path
        
        Output:
        tools -> list of prediction dictionaries (per frame) in the correct format as described in documentation 
        """
        print('Video file to be loaded: ' + str(fname))
        cap = cv2.VideoCapture(str(fname))
        # num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ###                                                                     ###
        ###  TODO: adapt the following part for YOUR submission: make prediction
        ###                                                                     ###
        frame_count = 0
        all_frames_predicted_outputs = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            results = self.model([frame])
            # results = self.model.predict(frame, imgsz=680, conf=0.5)
            boxes = results[0].boxes
            boxes = boxes.numpy()
            
            for i in range(len(boxes)):
                x_c, y_c, w, h = boxes.xywh[i]
                x_c = x_c.item()
                y_c = y_c.item()
                w = w.item()
                h = h.item()
                label = boxes.cls[i]
                probability = boxes.conf[i].item()
                corners=[[x_c-(w/2), y_c-(h/2), 0.5], [x_c+(w/2), y_c-(h/2), 0.5], [x_c+(w/2), y_c+(h/2), 0.5], [x_c-(w/2), y_c+(h/2), 0.5]]
                name = f"slice_nr_{frame_count}_" + self.tool_list[int(label)]
                dict = {'corners':corners,
                        'name': name,
                        'probability': probability}
                all_frames_predicted_outputs.append(dict)

            frame_count+=1
            
        return all_frames_predicted_outputs

            
        # all_frames_predicted_outputs = []
        # for fid in range(num_frames):
        #     tool_detections = self.generate_bbox(fid)
        #     all_frames_predicted_outputs += tool_detections



if __name__ == "__main__":
    Surgtoolloc_det().process()
