import os
import argparse
import cv2
import glob
import numpy as np
import xml.etree.ElementTree as ET


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='data/train/images')
    parser.add_argument('--stride', type=int, default=0.5)
    #  parser.add_argument('--crop_image_size', type=int, default=224)
    parser.add_argument('--xml_dir', type=str, required=True)
    args = parser.parse_args()
    return args


def read_content(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    target_list = []

    for boxes in root.iter('object'):
        ymin, xmin, ymax, xmax = None, None, None, None
        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)

        target_list.append([ymax - ymin, xmax - xmin])

    return target_list


def get_avg_shape(xml_dir):
    shape_list = []
    for r, d, f in os.walk(xml_dir):
        for ff in f:
            if ff.split('.')[-1].lower() == 'xml':
                xml_path = os.path.join(r, ff)
                single_shape_list = read_content(xml_path)
                shape_list.extend(single_shape_list)
    avg_h = sum([x[0] / len(shape_list) for x in shape_list])
    avg_w = sum([x[1] / len(shape_list) for x in shape_list])
    return [avg_h, avg_w]


def find_all_imgs(root_path):
    img_list = []
    for r, d, f in os.walk(root_path):
        for ff in f:
            if ff.split('.')[-1].lower() in ['jpg', 'jpeg', 'png']:
                img_list.append(os.path.join(r, ff))
    return img_list


def crop_save_one_img(img_path, save_dir, stride, crop_image_size):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    base_name = os.path.basename(img_path).split('.')[0]

    s_w, s_h = 0, 0
    count = 0
    while s_h + stride[0] < h:
        while s_w + stride[1] < w:
            new_img = img[s_h:s_h + crop_image_size[0],
                          s_w:s_w + crop_image_size[1], :]
            save_path = os.path.join(save_dir,
                                     base_name + '_' + str(count) + '.jpg')
            cv2.imwrite(save_path, new_img)
            count += 1
            s_w += stride[1]
        s_h += stride[0]


def main():
    args = parse_args()
    print("croped images will be saved in {}".format(args.save_dir))
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    img_list = find_all_imgs(args.image_dir)
    avg_shape = get_avg_shape(args.xml_dir)
    avg_shape = [x * 1.2 for x in avg_shape]
    stride = [x * args.stride for x in avg_shape]
    print('crop image size: h: {}, w:{}'.format(avg_shape[0], avg_shape[1]))
    print('stride for generating images: h:{}, w:{}'.format(
        stride[0], stride[1]))
    for x in img_list:
        print("{} start croping".format(x))
        crop_save_one_img(x, args.save_dir, stride, avg_shape)


if __name__ == "__main__":
    main()
