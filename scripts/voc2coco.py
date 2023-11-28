#####################################
# Converte VOC format to COCO format
#####################################
import argparse
import os
import shutil
import json

from os import listdir
from os.path import join
from tqdm import tqdm
import xml.etree.ElementTree as ET

VOC_DIR = "/content/drive/My Drive/VOCdevkit/"
COCO_DIR = "/content/drive/My Drive/COCO/"

# LABEL_MAP = {
#     "aeroplane": 1,
#     "bicycle": 2,
#     "bird": 3,
# }

miss_cvt_list = [
    "/content/drive/MyDrive/cloud_pc/data/VOCdevkit/VOC2012/JPEGImages/2011_003353.jpg",
    "/content/drive/MyDrive/cloud_pc/data/VOCdevkit/VOC2012/JPEGImages/2011_006777.jpg",
]

def convert_voc_to_coco(voc_dir: str, coco_dir: str, label_map: dict, categories: list) -> None:
    """Convert VOC format data to COCO format.

    Args:
        voc_dir (str): Directory path to VOC data (base dir).
        coco_dir (str): Directory path to COCO data (dst dir).
        label_map (dict): The label map.
        categories (list): The list of categories.
    """
    # Create the directories for the COCO dataset
    os.makedirs(join(coco_dir, "images"), exist_ok=True)
    os.makedirs(join(coco_dir, "annotations"), exist_ok=True)

    # Get the list of all VOC images
    print("[On-going] curating all imgs from google drive")
    images = listdir(join(voc_dir, "VOC2012", "JPEGImages"))
    images = [image for image in images if image.endswith(".jpg")]
    print("[DONE] curating all imgs from google drive")
    print("number of images: ", len(images))

    # Iterate over all images
    for image in tqdm(images):
        # Get the image path
        image_path = join(voc_dir, "VOC2012", "JPEGImages", image)

        # Get the annotation path
        annotation_path = join(voc_dir, "VOC2012", "Annotations", image.replace(".jpg", ".xml"))

        # Check if the annotation file exists
        if not os.path.exists(annotation_path):
            continue

        # Convert the VOC annotation to COCO format
        coco_annotation = convert_voc_annotation_to_coco(image_path, annotation_path, label_map, categories)

        # Save the COCO annotation
        coco_anno_fp = join(coco_dir, "annotations", image.replace(".jpg", ".json"))
        os.makedirs(os.path.dirname(coco_anno_fp), exist_ok=True)
        with open(coco_anno_fp, "w") as f:
            f.write(coco_annotation)

        # Check if the image file exists
        if os.path.exists(join(coco_dir, "images", image)):
            continue
        # Copy the image to the COCO dataset
        shutil.copy(image_path, join(coco_dir, "images", image))

def convert_voc_annotation_to_coco(image_path: str, annotation_path: str, label_map: dict, categories: list) -> str:
    """Convert a VOC annotation to COCO anotation.

    Args:
        image_path (str): Path to image.
        annotation_path (str): path to annotation.
        label_map (dict): The label map.
        categories (list): The list of categories.

    Returns:
        str: The coco annotation.
    """
    anno_id = 01
    # Parse the VOC annotation
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    # Get the image size
    image_size = root.find("size")
    width = int(float(image_size.find("width").text))
    height = int(float(image_size.find("height").text))

    # Get the objects
    objects = root.findall("object")

    # Create the COCO annotation
    coco_annotation = {
        "info": {
            "description": "COCO dataset converted from VOC dataset",
            "url": "http://cocodataset.org/",
            "version": "1.0",
            "year": 2018,
            "contributor": "VOC Original",
            "date_created": "2018-09-29 00:00:00",
        },
        "images": [],
        "annotations": [],
        "categories": categories,
    }

    image_id = os.path.basename(image_path)
    image_id = image_id.split(".")[0]
    image_id = int(float(image_id))
    # Add the image to the COCO annotation
    coco_image = {
        "id": image_id,
        "width": width,
        "height": height,
        "file_name": image_path.split("/")[-1]
    }
    coco_annotation["images"].append(coco_image)

    # Add the objects to the COCO annotation
    for _object in objects:
        try:
            # Get the object name
            object_name = _object.find("name").text

            # Get the object bounding box
            object_bbox = _object.find("bndbox")
            left = int(float(object_bbox.find("xmin").text))
            top = int(float(object_bbox.find("ymin").text))
            right = int(float(object_bbox.find("xmax").text))
            bottom = int(float(object_bbox.find("ymax").text))

            # Check if the object is out of the image
            if left < 0:
                left = 0
            if top < 0:
                top = 0
            if right > width:
                right = width
            if bottom > height:
                bottom = height
            if left >= right or top >= bottom:
                continue

            # Create the COCO annotation
            _coco_annotation = {
                "id": anno_id,
                "image_id": image_id,
                "category_id": label_map[object_name],
                "bbox": [left, top, right - left, bottom - top],  # left, right, width, height
                "iscrowd": 0
            }
            anno_id += 1
            coco_annotation["annotations"].append(_coco_annotation)

            if "part" not in _object.attrib:
                continue

            # Get the object part
            object_parts = _object.find("part")
            for object_part in object_parts:
                # Get the object part name
                object_part_name = object_part.find("name").text

                # Get the object part bounding box
                object_part_bbox = object_part.find("bndbox")
                left = int(float(object_part_bbox.find("xmin").text))
                top = int(float(object_part_bbox.find("ymin").text))
                right = int(float(object_part_bbox.find("xmax").text))
                bottom = int(float(object_part_bbox.find("ymax").text))

                # Check if the object part is out of the image
                if left < 0:
                    left = 0
                if top < 0:
                    top = 0
                if right > width:
                    right = width
                if bottom > height:
                    bottom = height
                if left >= right or top >= bottom:
                    continue

                # Create the COCO annotation
                _coco_annotation = {
                    "id": anno_id,
                    "image_id": image_id,
                    "category_id": label_map[object_part_name],
                    "bbox": [left, top, right - left, bottom - top],  # left, right, width, height
                    "iscrowd": 0
                }
                anno_id += 1
                coco_annotation["annotations"].append(_coco_annotation)
        except Exception as e:
            print(e)
            print("ERROR occured at: {}".format(annotation_path))
            continue

    return json.dumps(coco_annotation, indent=4)


def args():
    p = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    p.add_argument("-lp", "--label_map_path", type=str, required=True, help="Path to labelmap.")
    p.add_argument("-cp", "--category_path", type=str, required=True, help="Path to categoriy's file.")
    p.add_argument("-voc", "--voc_dir", type=str, required=True, help="Path to VOC data directory.")
    p.add_argument("-coco", "--coco_dir", type=str, required=True, help="Path to COCO data directory.")
    return p.parse_args()


if __name__ == "__main__":
    options = args()
    label_map_path = options.label_map_path
    category_fp = options.category_path
    VOC_DIR = options.voc_dir
    COCO_DIR = options.coco_dir

    # Create the COCO dataset directory
    if not os.path.exists(COCO_DIR):
        os.makedirs(COCO_DIR)

    # Load the label map
    with open(label_map_path, "r") as f:
        label_map = json.load(f)

    # Load the categories
    with open(category_fp, "r") as f:
        categories = json.load(f)

    # Convert the VOC dataset to the COCO dataset
    convert_voc_to_coco(VOC_DIR, COCO_DIR, label_map, categories)
