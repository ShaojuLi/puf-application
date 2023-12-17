# 第二步：
#       用于生成训练和测试文件，并放进对应的文件夹，可以选择使用何种方式生成训练的二值图像(直方图固定化或者自动阈值)
import os
import time

import cv2
import copy
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

from configuration import Config
from tools import PreprocessTools


class PreprocessImage:
    def __init__(self, original_path, generated_path):
        self.original_path = original_path
        self.generated_path = generated_path
        self.base_config = Config()
        self.tools = PreprocessTools()

    def __expand_sample(self, sample_path, config):
        one_class_of_samples = os.listdir(sample_path)

        test_template = None
        for one_img_path in one_class_of_samples:
            name, ext = os.path.splitext(one_img_path)
            if name[-1] == "C":
                # means template
                if config.sample_dir_name < config.template_range:
                    template_img_path = os.path.join(sample_path, one_img_path)
                    template_img = cv2.imread(template_img_path)
                    config.reset_img_pre()
                    config.need_resize = True
                    test_template = self.tools.img_pretreatment(template_img, config)
                    test_template_dir = os.path.join(config.template_path, str(config.sample_dir_name))
                    self.tools.create_folder(test_template_dir)
                    test_template_path = os.path.join(test_template_dir, "template.jpg")
                    cv2.imwrite(test_template_path, test_template)

                    test_threshold_file = open(os.path.join(test_template_dir, "threshold.txt"), 'w')
                    test_threshold_file.write(str(config.pre_threshold))
                break

        for one_img in one_class_of_samples:
            img_path = os.path.join(sample_path, one_img)
            name, ext = os.path.splitext(one_img)

            if os.path.isdir(img_path) or ext == ".txt" or name == "template_binary":
                continue

            config.reset_img_pre()
            config.need_resize = True
            config.need_border = True
            img = cv2.imread(img_path)
            img = self.tools.img_pretreatment(img, config)

            config.is_verify_img = True
            config.is_hs = False
            if config.sample_dir_name < config.template_range:
                if name[-1] == "C":
                    config.is_verify_img = False
                elif int(name[-1]) < config.verify_sample_min:
                    config.is_verify_img = False
                    if config.is_hs:
                        test_template_img = self.tools.img_pretreatment(test_template, config)
                        img = self.tools.histogram_specification(test_template_img, img)

            self.tools.change_angle(img, config)


    def __binary(self, config):
        expand_samples = os.listdir(config.expend_sample_dir)
        template_img = cv2.imread(config.template_img_path)

        config.reset_img_pre()
        config.need_gaussian = True
        config.need_border = True
        config.need_gray = True
        template_img = self.tools.img_pretreatment(template_img, config)

        ret, template_binary_img = cv2.threshold(template_img, config.pre_threshold, 255, cv2.THRESH_BINARY)

        template_img_mean_pixel = self.tools.compute_img_threshold_weight(template_binary_img)
        config.mean_pixel = template_img_mean_pixel
        for name in expand_samples:
            img_path = os.path.join(config.expend_sample_dir, name)
            img = cv2.imread(img_path)
            config.reset_img_pre()
            config.need_gaussian = True
            config.need_gray = True
            img = self.tools.img_pretreatment(img, config)
            binary_img = self.tools.auto_threshold(img, config)
            save_name = "binary_" + os.path.split(name)[1]
            save_path = config.train_sample_dir
            save = os.path.join(save_path, save_name)
            cv2.imwrite(save, binary_img)

    def generate_one(self, one_sample, config):
        """
        Args:
            one_sample:
            config:
        """
        print('processing the {}th folder'.format(one_sample))
        config.sample_dir_name = int(one_sample)

        one_sample_dir_path = os.path.join(self.original_path, one_sample)

        threshold_file = open(os.path.join(one_sample_dir_path, "threshold.txt"))
        config.pre_threshold = int(threshold_file.readlines()[0])
        self.__expand_sample(one_sample_dir_path, config)

        if config.sample_dir_name < config.template_range:

            config.expend_sample_dir = os.path.join(config.template_path, one_sample, "Expand")
            config.template_img_path = os.path.join(config.template_path, one_sample, "template.jpg")

            train_sample_dir = os.path.join(config.train_path, one_sample)
            self.tools.create_folder(train_sample_dir)
            config.train_sample_dir = train_sample_dir
            self.__binary(config)

    def generate(self):
        """

        """
        train_path = os.path.join(self.generated_path, "train")

        test_path = os.path.join(self.generated_path, "test")
        template_path = os.path.join(test_path, "template")
        verify_path = os.path.join(test_path, "verify")

        self.tools.create_folder(train_path)
        self.tools.create_folder(test_path)
        # self.tools.create_folder(template_path)
        self.tools.create_folder(verify_path)
        self.base_config.train_path = train_path
        self.base_config.test_path = test_path
        self.base_config.template_path = template_path
        self.base_config.verify_path = verify_path

        original_samples = os.listdir(self.original_path)

        with ThreadPoolExecutor(max_workers=self.base_config.workers) as executor:
            tasks = [executor.submit(self.generate_one, one_sample, copy.deepcopy(self.base_config))
                     for one_sample in original_samples]
            wait(tasks, return_when=ALL_COMPLETED)


if __name__ == '__main__':

    original = r"\your\original\samples"
    generated = r"\generate\dir"


    begin = time.time()
    image_processor = PreprocessImage(original, generated)
    image_processor.generate()

    end = time.time()
    time_consume = end - begin
    print('time consume：{}'.format(time_consume))
