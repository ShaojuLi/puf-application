import cv2
import numpy as np

from random import choice, uniform
import os


class PreprocessTools:

    @staticmethod
    def img_pretreatment(img, config):

        if config.need_resize:
            img = cv2.resize(img, (config.w_h, config.w_h))
        if config.need_border:
            img = cv2.copyMakeBorder(img, config.border_size, config.border_size,
                                     config.border_size, config.border_size, cv2.BORDER_CONSTANT, 0)

        if config.need_gaussian:
            img = cv2.GaussianBlur(img, config.kernel, 0)

        if config.need_gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def auto_threshold(self, img, config):
        pre_threshold = config.pre_threshold
        _, binary_img = cv2.threshold(img, pre_threshold, 255, cv2.THRESH_BINARY)
        mean_pixel = self.compute_img_threshold_weight(binary_img)

        count_times = config.count_times
        while count_times:
            if abs(mean_pixel - config.mean_pixel) > 0.5:
                count_times -= 1
                if mean_pixel > config.mean_pixel:
                    pre_threshold += 1
                    _, temp_binary_img = cv2.threshold(img, pre_threshold, 255, cv2.THRESH_BINARY)
                    mean_pixel = self.compute_img_threshold_weight(temp_binary_img)
                else:
                    pre_threshold -= 1
                    _, temp_binary_img = cv2.threshold(img, pre_threshold, 255, cv2.THRESH_BINARY)
                    mean_pixel = self.compute_img_threshold_weight(temp_binary_img)
            else:
                _, final_binary_img = cv2.threshold(img, pre_threshold, 255, cv2.THRESH_BINARY)
                return final_binary_img
        return binary_img

    @staticmethod
    def histogram_specification(template, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        template = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)

        color = ('h', 's', 'v')
        for i, col in enumerate(color):
            hist1, bins = np.histogram(img[:, :, i].ravel(), 256, [0, 256])
            hist2, bins = np.histogram(template[:, :, i].ravel(), 256, [0, 256])
            cdf1 = hist1.cumsum()
            cdf2 = hist2.cumsum()
            cdf1_hist = hist1.cumsum() / cdf1.max()
            cdf2_hist = hist2.cumsum() / cdf2.max()
            diff_cdf = [[0 for j in range(256)] for k in range(256)]
            for j in range(256):
                for k in range(256):
                    diff_cdf[j][k] = abs(cdf1_hist[j] - cdf2_hist[k])
            lut = [0 for j in range(256)]
            for j in range(256):
                min = diff_cdf[j][0]
                index = 0
                for k in range(256):
                    if min > diff_cdf[j][k]:
                        min = diff_cdf[j][k]
                        index = k
                lut[j] = ([j, index])

            h = int(img.shape[0])
            w = int(img.shape[1])
            for j in range(h):
                for k in range(w):
                    img[j, k, i] = lut[img[j, k, i]][1]

        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    @staticmethod
    def compute_img_threshold_weight(binary_img):
        sum_pixel = binary_img.sum()
        h, w = binary_img.shape
        mean_pixel = sum_pixel / (h * w)
        return mean_pixel

    def change_angle(self, img, config):
        if config.is_verify_img:
            save_path = os.path.join(config.verify_path, str(config.sample_dir_name))
        else:
            save_path = os.path.join(config.template_path, str(config.sample_dir_name), "Expand")
        self.create_folder(save_path)

        expend_times = config.expend_times

        w, h = img.shape[0:2]

        while expend_times:
            select_base = choice([0, 1])
            if select_base:
                angle_x = int(uniform(-config.sum_angle, config.sum_angle))
                abs_angle = config.sum_angle - abs(angle_x)
                angle_y = int(uniform(-abs_angle, abs_angle))
            else:
                angle_y = int(uniform(-config.sum_angle, config.sum_angle))
                abs_angle = config.sum_angle - abs(angle_y)
                angle_x = int(uniform(-abs_angle, abs_angle))

            angle_z = int(uniform(-config.angle_z, config.angle_z))
            fov = 42
            temp_res = self.get_perspective_img(img, w, h, angle_x, angle_y, angle_z, fov)

            save_name = "angle_xyz-{x}_{y}_{z}".format(x=angle_x, y=angle_y, z=angle_z) + \
                        str(uniform(-10, 10))+".jpg"
            save = os.path.join(save_path, save_name)
            cv2.imwrite(save, temp_res)
            expend_times -= 1

    def get_perspective_img(self, img, w, h, angle_x, angle_y, angle_z, fov):
        z = np.sqrt(w ** 2 + h ** 2) / 2 / np.tan(self.rad(fov / 2))
        # 齐次变换矩阵
        rx = np.array([[1, 0, 0, 0],
                       [0, np.cos(self.rad(angle_x)), -np.sin(self.rad(angle_x)), 0],
                       [0, -np.sin(self.rad(angle_x)), np.cos(self.rad(angle_x)), 0, ],
                       [0, 0, 0, 1]], np.float32)

        ry = np.array([[np.cos(self.rad(angle_y)), 0, np.sin(self.rad(angle_y)), 0],
                       [0, 1, 0, 0],
                       [-np.sin(self.rad(angle_y)), 0, np.cos(self.rad(angle_y)), 0, ],
                       [0, 0, 0, 1]], np.float32)

        rz = np.array([[np.cos(self.rad(angle_z)), np.sin(self.rad(angle_z)), 0, 0],
                       [-np.sin(self.rad(angle_z)), np.cos(self.rad(angle_z)), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]], np.float32)

        r = rx.dot(ry).dot(rz)

        # 四对点的生成
        pcenter = np.array([h / 2, w / 2, 0, 0], np.float32)

        p1 = np.array([0, 0, 0, 0], np.float32) - pcenter
        p2 = np.array([w, 0, 0, 0], np.float32) - pcenter
        p3 = np.array([0, h, 0, 0], np.float32) - pcenter
        p4 = np.array([w, h, 0, 0], np.float32) - pcenter

        dst1 = r.dot(p1)
        dst2 = r.dot(p2)
        dst3 = r.dot(p3)
        dst4 = r.dot(p4)

        list_dst = [dst1, dst2, dst3, dst4]

        org = np.array([[0, 0],
                        [w, 0],
                        [0, h],
                        [w, h]], np.float32)

        dst = np.zeros((4, 2), np.float32)

        # 投影至成像平面
        for i in range(4):
            dst[i, 0] = list_dst[i][0] * z / (z - list_dst[i][2]) + pcenter[0]
            dst[i, 1] = list_dst[i][1] * z / (z - list_dst[i][2]) + pcenter[1]

        warpR = cv2.getPerspectiveTransform(org, dst)

        result = cv2.warpPerspective(img, warpR, (h, w))
        return result

    @staticmethod
    def rad(x):
        return x * np.pi / 180

    @staticmethod
    def create_folder(path):
        if not os.path.exists(path):
            os.makedirs(path)

