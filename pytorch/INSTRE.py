# -*- coding: utf-8 -*-
# @Author:Liu Chuanlong
# -*- coding: utf-8 -*-
# @Author:Liu Chuanlong
import csv
import os
import os.path
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np

object_categories = ['01a_canada_book',
                     '01b_panda_book',
                     '02a_parchis',
                     '02b_plate',
                     '03a_toy_oldman',
                     '03b_toy_taobao',
                     '04a_matryoshka',
                     '04b_toy_youth',
                     '05a_foxtoy',
                     '05b_notebook',
                     '06a_caterpillar',
                     '06b_red_car',
                     '07a_toy_phone',
                     '07b_toy_cow',
                     '08a_crab',
                     '08b_DDog',
                     '09a_toy_deer',
                     '09b_clay_figurine',
                     '10a_hello_kitty',
                     '10b_prince_book',
                     '11a_crunch',
                     '11b_postcard',
                     '12a_poker',
                     '12b_mizimizi',
                     '13a_toy_car',
                     '13b_toy_man',
                     '14a_pocky',
                     '14b_Nestle',
                     '15a_dwarf',
                     '15b_porcelain_cat',
                     '16a_toy_woman',
                     '16b_taiwan101',
                     '17a_flowerpot',
                     '17b_toy_cat',
                     '18a_Doraemon',
                     '18b_toy_dog',
                     '19a_correction_tape',
                     '19b_dog_sharpener',
                     '20a_coconut_juice',
                     '20b_pechoin',
                     '21a_deli_penguin',
                     '21b_crest',
                     '22a_Linden_honey',
                     '22b_Oreo_cookies',
                     '23a_Yunnan_Baiyao_Aerosol',
                     '23b_Verbatim',
                     '24a_huiyuan_juice',
                     '24b_fish_sticker',
                     '25a_Six_walnut',
                     '25b_nescafe',
                     '26a_hersheys',
                     '26b_run_hou_tang',
                     '27a_china_unicom_card',
                     '27b_china_mobile_card',
                     '28a_Thermometer',
                     '28b_TopStrong_Cup',
                     '29a_Extra_Chewing_Gum',
                     '29b_Pretz',
                     '30a_Nutrient_Book',
                     '30b_Corn_Shape_Pothook',
                     '31a_Aodiao_Chocolate_Roll',
                     '31b_Blue_notebook',
                     '32a_Glue_stick',
                     '32b_bus_card',
                     '33a_Alice_Guitar_String_Packaging',
                     '33b_Pills_package',
                     '34a_Caculator',
                     '34b_Chips_Ahoy',
                     '35a_toy_jeep',
                     '35b_toy_tractor',
                     '36a_disney_mat',
                     '36b_Lycium_chinensis',
                     '37a_crazybird_notes',
                     '37b_zhangjunya',
                     '38a_new_year_pic',
                     '38b_stamp',
                     '39a_CRISPY',
                     '39b_yuji',
                     '40a_toy_carman',
                     '40b_oldman_candle',
                     '41a_SevenUp',
                     '41b_mrbrown_coffee',
                     '42a_Luffy',
                     '42b_Chopper',
                     '43a_opera_face',
                     '43b_garden_expo',
                     '44a_red_army',
                     '44b_tiny_girl',
                     '45a_5yuan',
                     '45b_1yuan',
                     '46a_god_of_fortune',
                     '46b_smiling_boy',
                     '47a_toy_golden_fish',
                     '47b_wierd_fish',
                     '48a_502',
                     '48b_pear_book',
                     '49a_AstickMini',
                     '49b_HelloPanda',
                     '50a_toy_snowman',
                     '50b_sprike']

def write_object_labels_csv(file, labeled_data, classses):
    # write a csv file
    print('[dataset] write file %s' % file)
    with open(file, 'w') as csvfile:
        fieldnames = ['name']
        fieldnames.extend(object_categories)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for (name, labels) in labeled_data.items():
            example = {'name': name}
            for i in range(len(classses)):
                example[fieldnames[i + 1]] = int(labels[i])
            writer.writerow(example)

    csvfile.close()

def read_object_labels_csv(file, header=True):
    images = []
    num_categories = 0
    print('[dataset] read', file)
    with open(file, 'r') as f:
        reader = csv.reader(f)
        rownum = 0
        for row in reader:
            if header and rownum == 0:
                header = row
            else:
                if num_categories == 0:
                    num_categories = len(row) - 1
                name = row[0]
                labels = (np.asarray(row[1:num_categories + 1])).astype(np.float32)
                labels = torch.from_numpy(labels)
                item = (name, labels)
                images.append(item)
            rownum += 1
    return images

class INSTREclassification(data.Dataset):
    def __init__(self, root, set, transform=None, target_transform=None):

        assert(set in ['train', 'test'])
        self.root = root
        self.datapath = os.path.join(root, 'INSTRE_release', 'INSTRE-S1')
        self.datalist = os.listdir(self.datapath)
        self.label = sorted(self.datalist)
        self.labeldata = {}
        self.transform = transform
        self.target_transform = target_transform
        self.classes = object_categories

        for name in self.datalist:
            one = -np.ones(len(self.label))
            n = self.label.index(name)
            one[n] = 1.0
            tmppath = os.path.join(self.datapath, name)
            imagepath = os.listdir(tmppath)
            for image in imagepath:
                if image.split('.')[-1] == 'txt':
                    continue
                if set == 'train' and int(image.split('.')[-2]) < 80:
                    enimg = os.path.join(name, image)
                    self.labeldata[enimg] = one
                elif set == 'test' and int(image.split('.')[-2]) > 80:
                    enimg = os.path.join(name, image)
                    self.labeldata[enimg] = one

        # define path of csv file
        path_csv = os.path.join(self.root, 'DATAFILES')
        # define filename of csv file
        file_csv = os.path.join(path_csv, 'INSTREclassification_' + set + '.csv')

        # create the csv file if necessary
        if not os.path.exists(file_csv):
            if not os.path.exists(path_csv):  # create dir if necessary
                os.makedirs(path_csv)
            # write csv file
            write_object_labels_csv(file_csv, self.labeldata, classses=self.classes)

        self.images = read_object_labels_csv(file_csv)

        print('[dataset] INSTRE classification set=%s number of classes=%d  number of images=%d' % (
            set, len(self.classes), len(self.images)))

    def __getitem__(self, index):
        path, target = self.images[index]

        img = Image.open(os.path.join(self.datapath, path)).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (img, path), target

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return len(self.classes)