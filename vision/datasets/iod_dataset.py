import numpy as np
import pathlib
import xml.etree.ElementTree as ET
import cv2
import pandas as pd
from vision.datasets.open_images import OpenImagesDataset


class IODDataset(OpenImagesDataset):
    def _read_data(self):
        boxes_num = 0
        data = []

        for i in range(1, 7):
            root = ET.parse(f'{self.root}annotation/annotation_s{i}.xml').getroot()
            for child in root:
                if child.tag != 'images':
                    continue

                images_tag = child
                for image_tag in images_tag:

                    image_file = f'{self.root}sequence_{i}/' + image_tag.attrib['file']

                    img = cv2.imread(image_file)

                    for box_tag in image_tag:
                        row = []
                        boxes_num += 1

                        row.append(image_file)

                        XMin = int(box_tag.attrib['left']) / img.shape[1]
                        row.append(XMin)
                        XMax = (int(box_tag.attrib['left']) + int(box_tag.attrib['width'])) / img.shape[1]
                        row.append(XMax)

                        YMin = int(box_tag.attrib['top']) / img.shape[0]
                        row.append(YMin)
                        YMax = (int(box_tag.attrib['top']) + int(box_tag.attrib['height'])) / img.shape[0]
                        row.append(YMax)

                        row.append(box_tag[0].text)

                        data.append(row)

        annotations = pd.DataFrame(index=np.arange(0, boxes_num),
                                   columns=['ImageID', 'XMin', 'XMax', 'YMin', 'YMax', 'ClassName'])
        for i in np.arange(0, boxes_num):
            annotations.loc[i] = data[i]

        class_names = ['BACKGROUND'] + sorted(list(annotations['ClassName'].unique()))
        class_dict = {class_name: i for i, class_name in enumerate(class_names)}
        data = []
        for image_id, group in annotations.groupby("ImageID"):
            boxes = group.loc[:, ["XMin", "YMin", "XMax", "YMax"]].values.astype(np.float32)
            labels = np.array([class_dict[name] for name in group["ClassName"]])
            data.append({
                'image_id': image_id,
                'boxes': boxes,
                'labels': labels
            })
        return data, class_names, class_dict

    def _read_image(self, image_id):
        image_file = image_id
        image = cv2.imread(str(image_file))
        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image





