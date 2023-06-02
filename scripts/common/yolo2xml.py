
import xml.etree.ElementTree as ET
import xml.dom.minidom as md
import os
import numpy as np
import glob

from PIL import  Image

def create_xml(txt_path, object_number):

    #保存フォルダへ移動
    # os.chdir('../../dataset/converted_to_xml')
    xml_path = txt_path.replace('.txt', '.xml')
    # xml_path = os.path.join(dataset_root, 'converted_to_xml', xml_file)

    #xmlファイルを生成する
    annotation = ET.Element('annotation')
    folder = ET.SubElement(annotation, 'folder')
    filename = ET.SubElement(annotation, 'filename')
    path = ET.SubElement(annotation, 'path')
    source = ET.SubElement(annotation, 'source')
    database = ET.SubElement(source, 'database')
    size = ET.SubElement(annotation, 'size')
    width = ET.SubElement(size, 'width')
    height = ET.SubElement(size, 'height')
    depth = ET.SubElement(size, 'depth')
    segment = ET.SubElement(annotation, 'segment')

    for i in range(object_number):
        object = ET.SubElement(annotation, 'object')
        name = ET.SubElement(object, 'name')
        pose = ET.SubElement(object, 'pose')
        truncated = ET.SubElement(object, 'truncated')
        difficult = ET.SubElement(object, 'difficult')
        bndbox = ET.SubElement(object, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        ymin = ET.SubElement(bndbox, 'ymin')
        xmax = ET.SubElement(bndbox, 'xmax')
        ymax = ET.SubElement(bndbox, 'ymax')

    tree = ET.ElementTree(annotation)
    # fl = file_name
    tree.write(xml_path)

    # os.chdir('../../scripts/common')

    

    return xml_path

def add_object(name):

    os.chdir('./converted_to_xml')
    xml_name = name.replace('txt', 'xml')

    # xmlに新たな物体情報を追加する
    tree = ET.parse(xml_name)
    root = tree.getroot()

    for annotation in root.findall('object'):
        object = ET.SubElement(annotation, 'object')
        name = ET.SubElement(object, 'name')
        pose = ET.SubElement(object, 'pose')
        truncated = ET.SubElement(object, 'truncated')
        difficult = ET.SubElement(object, 'difficult')
        bndbox = ET.SubElement(object, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        ymin = ET.SubElement(bndbox, 'ymin')
        xmax = ET.SubElement(bndbox, 'xmax')
        ymax = ET.SubElement(bndbox, 'ymax')

    tree = ET.ElementTree(root)
    fl = xml_name
    tree.write(fl)

    os.chdir('../')

    return fl

def read_txt(xml_path, txt_path):
    # txt_path = os.path.join('anottation_data/', name)
    # txt_path = os.path.join(dataset_root, "vidvipo_full_2023_05_27", name)
    # xml_path = os.path.join(dataset_root, "converted_to_xml", name)
    # xml_path.replace('.txt', '.xml')

    with open(txt_path) as f:
        l_strip = [s.strip() for s in f.readlines()]

        #画像中の物体の個数を出力
        #print(len(l_strip))
        #物体が２つ以上ならxmlファイルを書き換える
        if len(l_strip) != '1':
            # xml_name = name.replace('.txt', '.xml')
            create_xml(xml_path, len(l_strip))
        #print(l_strip)
        #txtデータ読み込み
        for i, j in enumerate(l_strip):

            #データを分割する
            # splitは番号、ｘ座標、y座標、Width、Heightの順に格納されているリスト型
            split = j.split()

            #辞書型に代入する
            data = {'number':0,'x_coordinate':0, 'y_coordinate':0, 'width':0, 'height':0}
            data['number'] = split[0]
            data['x_coordinate'] = split[1]
            data['y_coordinate'] = split[2]
            data['width'] = split[3]
            data['height'] = split[4]

            #作成したxmlへ代入
            # xml_name = xml_path.replace('.txt', '.xml')
            yolo_to_xml(xml_path, txt_path, data, i)

    # return name

def yolo_to_xml(xml_path, txt_path, data, object_counter):

    #保存用の辞書型生成
    coordinated_data = {'x_min': 0, 'x_max': 0, 'y_min': 0, 'y_max': 0}

    #画像データの取得
    # os.chdir('../../dataset/vidvipo_full_2023_05_27')
    # img_path = os.path.join(dataset_root, "vidvipo_full_2023_05_27", name)
    img_path = xml_path.replace('.xml', '.jpg')

    # img_path.replace('.txt', '.jpg')
    im = np.array(Image.open(img_path))

    #object_nameの変更(任意の物体名を追加可能)
    if data['number'] == '0':
        data['number'] = 'person'
    elif data['number'] == '1':
        data['number'] = 'bicycle'
    elif data['number'] == '2':
        data['number'] = 'car'
    elif data['number'] == '3':
        data['number'] = 'motorbike'
    elif data['number'] == '4':
        data['number'] = 'bus'
    elif data['number'] == '5':
        data['number'] = 'train'
    elif data['number'] == '6':
        data['number'] = 'truck'
    elif data['number'] == '7':
        data['number'] = 'boat'
    elif data['number'] == '8':
        data['number'] = 'traffic_light'
    elif data['number'] == '9':
        data['number'] = 'bicycler'
    elif data['number'] == '10':
        data['number'] = 'braille_block'
    elif data['number'] == '11':
        data['number'] = 'guardrail'
    elif data['number'] == '12':
        data['number'] = 'white_line'
    elif data['number'] == '13':
        data['number'] = 'crosswalk'
    elif data['number'] == '14':
        data['number'] = 'signal_button'
    elif data['number'] == '15':
        data['number'] = 'signal_red'
    elif data['number'] == '16':
        data['number'] = 'signal_blue'
    elif data['number'] == '17':
        data['number'] = 'stairs'
    elif data['number'] == '18':
        data['number'] = 'handrail'
    elif data['number'] == '19':
        data['number'] = 'steps'
    elif data['number'] == '20':
        data['number'] = 'faragates'
    elif data['number'] == '21':
        data['number'] = 'train_ticket_machine'
    elif data['number'] == '22':
        data['number'] = 'shrubs'
    elif data['number'] == '23':
        data['number'] = 'tree'
    elif data['number'] == '24':
        data['number'] = 'vending_machine'
    elif data['number'] == '25':
        data['number'] = 'bathroom'
    elif data['number'] == '26':
        data['number'] = 'door'
    elif data['number'] == '27':
        data['number'] = 'elevator'
    elif data['number'] == '28':
        data['number'] = 'escalator'
    elif data['number'] == '29':
        data['number'] = 'bollard'
    elif data['number'] == '30':
        data['number'] = 'bus_stop_sign'
    elif data['number'] == '31':
        data['number'] = 'pole'
    elif data['number'] == '32':
        data['number'] = 'monument'
    elif data['number'] == '33':
        data['number'] = 'fence'
    elif data['number'] == '34':
        data['number'] = 'wall'
    elif data['number'] == '35':
        data['number'] = 'signboard'
    elif data['number'] == '36':
        data['number'] = 'flag'
    elif data['number'] == '37':
        data['number'] = 'postbox'
    elif data['number'] == '38':
        data['number'] = 'safety_cone'

    #座標の変更(非正規化)
    data['x_coordinate'] = int(float(data['x_coordinate']) * im.shape[1])
    data['y_coordinate'] = int(float(data['y_coordinate']) * im.shape[0])
    data['width'] = int(float(data['width']) * im.shape[1])
    data['height'] = int(float(data['height']) * im.shape[0])
    coordinated_data['x_min'] = int(data['x_coordinate'] - (data['width']/2))
    coordinated_data['x_max'] = int(data['x_coordinate'] + (data['width']/2))
    coordinated_data['y_min'] = int(data['y_coordinate'] - (data['height']/2))
    coordinated_data['y_max'] = int(data['y_coordinate'] + (data['height']/2))

    for key, value in coordinated_data.items():
        print(key, value)

    #xmlへの書き込み開始
    # os.chdir('../../')
    # os.chdir('../converted_to_xml')
    # os.chdir('.')

    #xmlにデータを書き込んでいる
    tree = ET.parse(xml_path)
    root = tree.getroot()

    root.findall('filename')[0].text = img_path
    root.findall('./*/width')[0].text = str(im.shape[1])
    root.findall('./*/height')[0].text = str(im.shape[0])
    root.findall('./*/depth')[0].text = str(im.shape[2])
    root.findall('./*/name')[object_counter].text = data['number']
    root.findall('./*/*/xmin')[object_counter].text = str(coordinated_data['x_min'])
    root.findall('./*/*/ymin')[object_counter].text = str(coordinated_data['y_min'])
    root.findall('./*/*/xmax')[object_counter].text = str(coordinated_data['x_max'])
    root.findall('./*/*/ymax')[object_counter].text = str(coordinated_data['y_max'])


    tree.write(xml_path)

    # os.chdir('../')

    # return data


if __name__ == "__main__":

    #txtデータの名前のリストを保存する
    # path = os.listdir('./anottation_data')
    
    #xmlファイルを生成してtxtファイルの内容を書き込む
    dataset_root = '../../dataset'
    path = glob.glob(os.path.join(dataset_root, 'vidvipo_full_2023_05_27/*.txt'))
    for txt_path in path:
        # print(txt_path)
        xml_path = create_xml(txt_path, 1)
        read_txt(xml_path, txt_path)
        
    # txt_path = os.path.join(dataset_root, "vidvipo_full_2023_05_27", path)
    # xml_name = path.replace('.txt', '.xml')
    # xml_path = create_xml(txt_path, 1)
    # read_txt(xml_path, txt_path)

    # xmlファイル生成
    # for i, name in enumerate(path):
        # name = name.replace('.txt', '.xml')
    #     create_xml(name, 1)
    #     read_txt(path[i])



#     import glob
 
# path = './dir/*.txt'
 
# list1 = glob.glob(path);
 
# print(list1)

