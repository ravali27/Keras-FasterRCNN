from keras import backend as K


class Config:

    def __init__(self):

        self.verbose = True

        # base CNN model
        self.network = 'resnet50'

        # setting for data augmentation
        self.use_horizontal_flips = False
        self.use_vertical_flips = False
        self.rot_90 = False
        self.use_freq_mask = False
        self.use_time_mask = False

        # anchor box scales
        self.anchor_box_scales = [128, 256, 512]

        # anchor box ratios
        self.anchor_box_ratios = [[1, 1], [1, 4], [4, 1]]

        # size to resize the smallest side of the image
        self.im_size = 600

        # image channel-wise mean to subtract
        self.img_channel_mean = [0, 0, 0]
        self.img_scaling_factor = 1.0

        # number of ROIs at once
        self.num_rois = 300

        # stride at the RPN (this depends on the network configuration)
        self.rpn_stride = 16

        self.balanced_classes = False

        # scaling the stdev
        self.std_scaling = 1.0
        self.classifier_regr_std = [8.0, 8.0, 4.0, 4.0]

        # overlaps for RPN
        self.rpn_min_overlap = 0.3
        self.rpn_max_overlap = 0.7

        # overlaps for classifier ROIs
        self.classifier_min_overlap = 0.1
        self.classifier_max_overlap = 0.5

        # placeholder for the class mapping, automatically generated by the parser
        self.class_mapping = None

        #location of pretrained weights for the base network
        # weight files can be found at:
        # https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels_notop.h5
        # https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5

        self.model_path = 'model_frcnn.vgg.hdf5'
