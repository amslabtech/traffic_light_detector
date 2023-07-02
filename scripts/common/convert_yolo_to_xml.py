import os
import xml.etree.ElementTree as ET
import pandas as pd
import glob
from PIL import Image




def convert_yolo_to_xml(yolo_data, image_filename, image_size, class_names):
    root = ET.Element("annotation")

    folder = ET.SubElement(root, "folder")
    folder.text = "VIDVIP"

    filename = ET.SubElement(root, "filename")
    filename.text = image_filename

    source = ET.SubElement(root, "source")
    database = ET.SubElement(source, "database")
    database.text = "The VIDVIP Database"
    annotation = ET.SubElement(source, "annotation")
    annotation.text = "PASCAL VOC"
    image = ET.SubElement(source, "image")
    image.text = "flickr"

    size = ET.SubElement(root, "size")
    width = ET.SubElement(size, "width")
    width.text = str(image_size[0])
    height = ET.SubElement(size, "height")
    height.text = str(image_size[1])
    depth = ET.SubElement(size, "depth")
    depth.text = "3"

    segmented = ET.SubElement(root, "segmented")
    segmented.text = "0"

    for yolo_line in yolo_data:
        class_id, x, y, w, h = yolo_line.split()
        class_name = class_names[class_id]


        object_elem = ET.SubElement(root, "object")

        name = ET.SubElement(object_elem, "name")
        name.text = class_name

        pose = ET.SubElement(object_elem, "pose")
        pose.text = "Unspecified"

        truncated = ET.SubElement(object_elem, "truncated")
        truncated.text = "0"

        difficult = ET.SubElement(object_elem, "difficult")
        difficult.text = "0"

        bndbox = ET.SubElement(object_elem, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = str(int((float(x) - float(w) / 2) * image_size[0]))
        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = str(int((float(y) - float(h) / 2) * image_size[1]))
        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = str(int((float(x) + float(w) / 2) * image_size[0]))
        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = str(int((float(y) + float(h) / 2) * image_size[1]))

    tree = ET.ElementTree(root)
    return tree

def save_xml_file(xml_tree, xml_filename):
    xml_tree.write(xml_filename, encoding="utf-8", xml_declaration=True)


def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size

def convert_dataset(yolo_dataset_dir, xml_dataset_dir, class_names):
    if not os.path.exists(xml_dataset_dir):
        os.makedirs(xml_dataset_dir)

    for filename in sorted(os.listdir(yolo_dataset_dir)):
        if filename.endswith(".txt"):
            yolo_file_path = os.path.join(yolo_dataset_dir, filename)
            xml_file_path = os.path.join(xml_dataset_dir, filename[:-4] + ".xml")

            image_filename = filename[:-4] + ".jpg"
            image_path = os.path.join(yolo_dataset_dir, image_filename)

            image_size = get_image_size(image_path)

            with open(yolo_file_path, "r") as yolo_file:
                yolo_lines = yolo_file.readlines()

            xml_tree = convert_yolo_to_xml(yolo_lines, image_filename, image_size, class_names)
            save_xml_file(xml_tree, xml_file_path)

# Usage example

if __name__ == "__main__":

    class_names={}
    class_info=pd.read_csv('../../dataset/annotations.csv').values.tolist()

    for line in class_info[:-2]:
        class_names.update({line[0]: line[1]})

    
    dataset_root = '../../dataset'
    #txtデータの名前のリストを保存する
    path = glob.glob(os.path.join(dataset_root, 'vidvipo_full_2023_05_27/*.txt'))

        
yolo_dataset_dir = os.path.join(dataset_root, 'vidvipo_full_2023_05_27')
xml_dataset_dir = os.path.join(dataset_root, 'xml_annotations')

convert_dataset(yolo_dataset_dir, xml_dataset_dir, class_names)
