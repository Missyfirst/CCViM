import re
import glob
import os
import tqdm
import pathlib
from misc.utils import rm_n_mkdir
import json

from dataset import get_dataset

import cv2
import math
import random
import colorsys
import numpy as np
from misc.utils import get_bounding_box



####
def random_colors(N, bright=True):
    """Generate random colors.

    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors
####

def visualize_instances_map(
    input_image, inst_map, type_map=None, type_colour=None, line_thickness=2
):
    """Overlays segmentation results on image as contours.

    Args:
        input_image: input image
        inst_map: instance mask with unique value for every object
        type_map: type mask with unique value for every class
        type_colour: a dict of {type : colour} , `type` is from 0-N
                     and `colour` is a tuple of (R, G, B)
        line_thickness: line thickness of contours

    Returns:
        overlay: output image with segmentation overlay as contours
    """
    overlay = np.copy((input_image).astype(np.uint8))

    inst_list = list(np.unique(inst_map))  # get list of instances
    inst_list.remove(0)  # remove background

    inst_rng_colors = random_colors(len(inst_list))
    inst_rng_colors = np.array(inst_rng_colors) * 255
    inst_rng_colors = inst_rng_colors.astype(np.uint8)

    for inst_idx, inst_id in enumerate(inst_list):
        inst_map_mask = np.array(inst_map == inst_id, np.uint8)  # get single object
        y1, y2, x1, x2 = get_bounding_box(inst_map_mask)
        y1 = y1 - 2 if y1 - 2 >= 0 else y1
        x1 = x1 - 2 if x1 - 2 >= 0 else x1
        x2 = x2 + 2 if x2 + 2 <= inst_map.shape[1] - 1 else x2
        y2 = y2 + 2 if y2 + 2 <= inst_map.shape[0] - 1 else y2
        inst_map_crop = inst_map_mask[y1:y2, x1:x2]
        contours_crop = cv2.findContours(
            inst_map_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        # only has 1 instance per map, no need to check #contour detected by opencv
        contours_crop = np.squeeze(
            contours_crop[0][0].astype("int32")
        )  # * opencv protocol format may break
        contours_crop += np.asarray([[x1, y1]])  # index correction
        if type_map is not None:
            type_map_crop = type_map[y1:y2, x1:x2]

            type_list, type_pixels = np.unique(type_map_crop, return_counts=True)
            type_list = list(zip(type_list, type_pixels))
            type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
            if type_list[0][0] == 0:
                type_id = type_list[1][0]
            else:
                type_id = type_list[0][0]

            # type_id = np.unique(type_map_crop).max()  # non-zero
            inst_colour = type_colour[type_id][1]
        else:
            inst_colour = (inst_rng_colors[inst_idx]).tolist()
        cv2.drawContours(overlay, [contours_crop], -1, inst_colour, line_thickness)
    return overlay


if __name__ == "__main__":
    dataset_name = "consep"
    save_root = "/opt/data/private/zhuyun/dataset/consep_extracted/processed/type_renew/%s" % dataset_name
    type_classification = True
    # a dictionary to specify where the dataset path should be
    dataset_info = {
        "valid": {
            "img": (".png", "/opt/data/private/zhuyun/dataset/consep_extracted/CoNSeP/Test/Images/"),
            "ann": (".mat", "/opt/data/private/zhuyun/dataset/consep_extracted/CoNSeP/Test/Labels/"),
        },
    }
    ###### color #########
    nr_types = 5
    type_info_path = "/opt/data/private/zhuyun/MedImage/HoVCoCs/type_consep.json"
    type_info_dict = json.load(open(type_info_path, "r"))
    type_info_dict = {
        int(k): (v[0], tuple(v[1])) for k, v in type_info_dict.items()
    }
    # availability check
    for k in range(nr_types):
        if k not in type_info_dict:
            assert False, "Not detect type_id=%d defined in json." % k


    patterning = lambda x: re.sub("([\[\]])", "[\\1]", x)
    parser = get_dataset(dataset_name)
    for split_name, split_desc in dataset_info.items():
        img_ext, img_dir = split_desc["img"]
        ann_ext, ann_dir = split_desc["ann"]

        out_dir = "%s/%s/%s/%s/" % (
            save_root,
            dataset_name,
            split_name,
            'overlay'
        )
        file_list = glob.glob(patterning("%s/*%s" % (ann_dir, ann_ext)))
        file_list.sort()  # ensure same ordering across platform

        rm_n_mkdir(out_dir)

        pbar_format = "Process File: |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
        pbarx = tqdm.tqdm(
            total=len(file_list), bar_format=pbar_format, ascii=True, position=0
        )

        for file_idx, file_path in enumerate(file_list):
            base_name = pathlib.Path(file_path).stem

            img = parser.load_img("%s/%s%s" % (img_dir, base_name, img_ext))
            ann = parser.load_ann(
                "%s/%s%s" % (ann_dir, base_name, ann_ext), type_classification
            )
            type_map = (ann[..., 1]).copy()
            inst_map = (ann[..., 0]).copy()

            overlaid_img = visualize_instances_map(img,inst_map=inst_map,type_map=type_map,type_colour=type_info_dict)

            save_path = "%s/%s.png" % (out_dir, base_name)
            cv2.imwrite(save_path, cv2.cvtColor(overlaid_img, cv2.COLOR_RGB2BGR))




