# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np

import xml.etree.ElementTree as ET

from ppdet.core.workspace import register, serializable

from .dataset import DetDataset

from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)


@register
@serializable
class LogoDataSet(DetDataset):
    """
    Load dataset with PascalVOC format.

    Notes:
    `anno_path` must contains xml file and image file path for annotations.

    Args:
        dataset_dir (str): root directory for dataset.
        image_dir (str): directory for images.
        anno_path (str): voc annotation file path.
        sample_num (int): number of samples to load, -1 means all.
        use_default_label (bool): whether use the default mapping of
            label to integer index. Default True.
        with_background (bool): whether load background as a class,
            default True.
        label_list (str): if use_default_label is False, will load
            mapping between category and class index.
    """

    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 sample_num=-1,
                 data_fields=['image'],
                 label_list=None):
        super(LogoDataSet, self).__init__(
            image_dir=image_dir,
            anno_path=anno_path,
            sample_num=sample_num,
            dataset_dir=dataset_dir,
            data_fields=data_fields)
        self.label_list = label_list

    def parse_dataset(self,):
        anno_path = os.path.join(self.dataset_dir, self.anno_path)
        label_path = os.path.join(self.dataset_dir, 'label_list.txt')
        #image_dir = os.path.join(self.dataset_dir, self.image_dir)

        # mapping category name to class id
        # if with_background is True:
        #   background:0, first_class:1, second_class:2, ...
        # if with_background is False:
        #   first_class:0, second_class:1, ...
        records = []
        tmp_records = {}
        ct = 0
        cname2cid = {}
        cid2cname = {}
        if not os.path.exists(anno_path):
            raise ValueError("label_list {} does not exists".format(
                anno_path))
        with open(label_path, 'r', encoding='utf-8') as fr:
            for line in fr.readlines():
                line = line.strip()
                if line not in cname2cid:
                    id = ct
                    name = line
                    cname2cid[name] = id
                    cid2cname[id] = name
                    ct += 1

        idx = 0
        with open(anno_path, 'r', encoding='utf-8') as fr:
            for line in fr.readlines():
                if not line:
                    break
                splitted_line = line.strip().split(',')
                img_file = splitted_line[0]
                if not os.path.exists(img_file):
                    print(
                        'Illegal image file: {}, and it will be ignored'.format(
                            img_file))
                    continue
                im_id = np.array([idx])
                idx += 1

                gt_bbox = []
                gt_class = []
                difficult = []

                x1 = float(splitted_line[3])
                y1 = float(splitted_line[4])
                x2 = float(splitted_line[5])
                y2 = float(splitted_line[6])

                if x2 > x1 and y2 > y1:
                    gt_bbox.append([x1, y1, x2, y2])
                    gt_class.append([0])
                    difficult.append([0])
                else:
                    print(
                        'Found an invalid bbox in annotations: img_file: {}'
                        ', x1: {}, y1: {}, x2: {}, y2: {}.'.format(
                            img_file, x1, y1, x2, y2))
                gt_bbox = np.array(gt_bbox).astype('float32')
                gt_class = np.array(gt_class).astype('int32')
                difficult = np.array(difficult).astype('int32')
                voc_rec = {
                    'im_file': img_file,
                    'im_id': im_id,
                    'gt_class': gt_class,
                    'gt_bbox': gt_bbox,
                    'difficult': difficult
                }

                if im_id[0] not in tmp_records:
                    tmp_records[im_id[0]] = voc_rec
                else:
                    for key in voc_rec:
                        if key != 'im_id' and key != 'im_file':
                            tmp_records[im_id[0]][key] = np.append(tmp_records[im_id[0]][key], voc_rec[key], axis=0)

        for key in tmp_records:
            records.append(tmp_records[key])
        assert len(records) > 0, 'not found any voc record in %s' % (
            self.anno_path)
        print('{} samples in file {}'.format(ct, anno_path))
        self.roidbs, self.cname2cid = records, sorted(cname2cid)


    def get_label_list(self):
        return os.path.join(self.dataset_dir, 'label_list.txt')
