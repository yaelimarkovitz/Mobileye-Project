from dataclasses import dataclass
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

from model.candidates import Candidates
from model.find_lights import find_tfl_lights
from initialization.create_data_set import crop_image
from model.calc_distance import calc_TFL_dist, get_foe_rotate
from view.visualation import visual


class FrameContainer(object):
    def __init__(self, img_path, traffic_lights):
        self.img = plt.imread(img_path)
        self.traffic_light = traffic_lights
        self.traffic_lights_3d_location = []
        self.EM = []
        self.corresponding_ind = []
        self.valid = []

#TODO merge the frame_conteiner with the candidates

@dataclass
class TflManager:
    def __init__(self, pp, focal, egomotion):
        self.principle_point = pp
        self.focal = focal
        self.em = egomotion
        self.my_model = load_model("./model.h5")
        self.prev_candidates = Candidates("", [], [])

    def on_frame(self, frame_path, index):
        lights_candidates, tfl_candidates = self.find_tfl(frame_path)

        if index == 0:
            distances , rot_pts ,foe =0,0,0
        else:
            distances, foe, rot_pts = self.calc_distance(tfl_candidates, index)
        visual(lights_candidates, tfl_candidates, distances, rot_pts, foe)
        self.prev_candidates = tfl_candidates

    def find_tfl(self, frame_path):
        can, aux = self.find_lights(frame_path)
        lights_candidates = Candidates(frame_path, can, aux)
        can, aux = self.recognize_tfl(lights_candidates)
        tfl_candidates = Candidates(frame_path, can, aux)
        return lights_candidates, tfl_candidates

    def find_lights(self, frame) -> (list, list):
        image = np.array(Image.open(frame))
        return find_tfl_lights(image)

    def recognize_tfl(self, candidate: Candidates) -> (list, list):

        croped_images = [crop_image(candidate.frame_path, point[0], point[1]) for point in candidate.points]
        predictions = self.my_model.predict(np.array(croped_images))
        tfl_array = []
        auxiliary = []
        for index, predict in enumerate(predictions[:, 1]):
            if predict > 0.5:
                tfl_array.append(candidate.points[index])
                auxiliary.append(candidate.auxiliary[index])
        return tfl_array, auxiliary

    def calc_distance(self, cur_frame: Candidates, index: int) -> float:
        prev_container = FrameContainer(self.prev_candidates.frame_path, np.array(self.prev_candidates.points))
        curr_container = FrameContainer(cur_frame.frame_path, np.array(cur_frame.points))
        curr_container.EM = self.em[index - 1]
        z = calc_TFL_dist(prev_container, curr_container, self.focal, self.principle_point)
        foe, rot_pts = get_foe_rotate(prev_container, curr_container, self.focal, self.principle_point)
        return z, foe, rot_pts
