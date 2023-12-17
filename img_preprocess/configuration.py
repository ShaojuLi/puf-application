# 配置文件


class Config:
    def __init__(self):
        # pretreatment
        self.workers = 8

        # for remove noise
        self.need_gaussian = False
        self.kernel = (11, 11)
        # is need resize
        self.need_resize = False
        self.w_h = 1024
        # is need border
        self.need_border = False
        self.border_size = 200
        # is need gray
        self.need_gray = False

        # for expend sample
        self.sum_angle = 15
        self.angle_z = 5
        self.expend_times = 30
        self.expend_sample_dir = None
        self.save_angle_path_compare = None
        self.train_sample_dir = None
        # auto threshold
        self.count_times = 200
        self.pre_threshold = 255
        self.template_img_path = None
        self.mean_pixel = 0
        # auto_threshold or hs
        self.is_hs = True
        self.is_auto = True
        # dir
        self.sample_dir_name = None
        self.train_path = None
        self.test_path = None
        # select verify sample
        self.verify_sample_max = 3
        self.verify_sample_min = 3
        self.template_range = 81
        self.is_template = True
        self.template_path = None
        self.verify_path = None
        self.is_verify_img = True
        self.test_template_path = None

        # for test
        self.template = None

    def reset_img_pre(self):
        self.need_gaussian = False
        self.need_gray = False
        self.need_resize = False
        self.need_border = False
