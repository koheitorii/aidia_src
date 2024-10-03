import re
import numpy as np
import tensorflow as tf

from aidia.ai.config import AIConfig
from aidia.ai.models.yolov4 import yolov4_utils as utils
from aidia.ai.models.yolov4 import yolov4_common as common
from aidia.ai.models.yolov4 import yolov4_backbone as backbone
from aidia.ai.models.yolov4.yolov4_config import YOLO_Config


class YOLO(tf.keras.Model):
    ################################
    ### Override Model Functions ###
    ################################
    def __init__(self, config:AIConfig):
        super().__init__()
        self.config = config
        self.is_tiny = True if self.config.MODEL.split("-")[-1] == "tiny" else False
        self.version = int(re.sub(r'[^0-9]', '', self.config.MODEL))

        yolo_config = YOLO_Config()
        self.NUM_CLASS = config.num_classes
        self.IOU_LOSS_THRESH = yolo_config.IOU_LOSS_THRESH
        self.STRIDES, self.ANCHORS, self.XYSCALE = yolo_config.get_yolo_params(self.is_tiny, self.version)
        self.ANCHOR_PER_SCALE = yolo_config.ANCHOR_PER_SCALE

        # for test
        self.IOU_THRESH = yolo_config.IOU_THRESHOLD
        self.SCORE_THRESH = yolo_config.SCORE_THRESHOLD

        if config.MODEL == "YOLOv4":
            self.freeze_layers = ['conv2d_93', 'conv2d_101', 'conv2d_109']
        elif config.MODEL == "YOLOv4-tiny":
            self.freeze_layers = ['conv2d_17', 'conv2d_20']
        elif config.MODEL == "YOLOv3":
            self.freeze_layers = ['conv2d_58', 'conv2d_66', 'conv2d_74']
        elif config.MODEL == "YOLOv3-tiny":
            self.freeze_layers = ['conv2d_9', 'conv2d_12']
        
        self.model = self.create_model()
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
     
    def create_model(self):
        input_layer = tf.keras.layers.Input([self.config.INPUT_SIZE, self.config.INPUT_SIZE, 3])

        if self.config.MODEL == "YOLOv4":
            feature_maps = self.YOLOv4(input_layer)
        elif self.config.MODEL == "YOLOv4-tiny":
            feature_maps = self.YOLOv4_tiny(input_layer)
        elif self.config.MODEL == "YOLOv3":
            feature_maps = self.YOLOv3(input_layer)
        elif self.config.MODEL == "YOLOv3-tiny":
            feature_maps = self.YOLOv3_tiny(input_layer)
        else:
            raise NotImplementedError

        if self.is_tiny:
            bbox_tensors = []
            for i, fm in enumerate(feature_maps):
                if i == 0:
                    bbox_tensor = self.decode_train(fm, self.config.INPUT_SIZE // 16, i)
                else:
                    bbox_tensor = self.decode_train(fm, self.config.INPUT_SIZE // 32, i)
                bbox_tensors.append(fm)
                bbox_tensors.append(bbox_tensor)
        else:
            bbox_tensors = []
            for i, fm in enumerate(feature_maps):
                if i == 0:
                    bbox_tensor = self.decode_train(fm, self.config.INPUT_SIZE // 8, i)
                elif i == 1:
                    bbox_tensor = self.decode_train(fm, self.config.INPUT_SIZE // 16, i)
                else:
                    bbox_tensor = self.decode_train(fm, self.config.INPUT_SIZE // 32, i)
                bbox_tensors.append(fm)
                bbox_tensors.append(bbox_tensor)
        
        return tf.keras.Model(input_layer, bbox_tensors)
        
    @property
    def metrics(self):
        return [self.loss_tracker]
    
    def train_step(self, data):
        inputs, targets = data
        with tf.GradientTape() as tape:
            pred_result = self.model(inputs, training=True)
            giou_loss = conf_loss = prob_loss = 0
            # optimizing process
            num_output = 2 if self.is_tiny else 3
            for i in range(num_output):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = self.compute_loss(pred, conv, targets[i][0], targets[i][1], i=i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]
            total_loss = giou_loss + conf_loss + prob_loss
            gradients = tape.gradient(total_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            self.loss_tracker.update_state(total_loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        inputs, targets = data
        pred_result = self.model(inputs, training=False) # if training=True, update BatchNormalization params during inferences
        giou_loss = conf_loss = prob_loss = 0
        num_output = 2 if self.is_tiny else 3
        for i in range(num_output):
            conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
            loss_items = self.compute_loss(pred, conv, targets[i][0], targets[i][1], i=i)
            giou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]
        total_loss = giou_loss + conf_loss + prob_loss
        self.loss_tracker.update_state(total_loss)
        return {"loss": self.loss_tracker.result()}
    
    def call(self, x):
        return self.model(x)
    
    def predict(self, img, score_thresh=0.5):
        org_image = np.copy(img)
        org_h, org_w, _ = org_image.shape

        image_data = utils.image_preprocess(img, self.config.image_size)
        image_data = image_data[np.newaxis, ...]

        # print(len(self.model(image_data)))
        if self.is_tiny:
            _, pred_mbbox, _, pred_lbbox = self.model(image_data)
        else:
            _, pred_sbbox, _, pred_mbbox, _, pred_lbbox = self.model(image_data)
        # print(pred_sbbox[-1].shape, pred_mbbox[-1].shape, pred_lbbox[-1].shape)

        if self.is_tiny:
            pred_bbox = np.concatenate([np.reshape(pred_mbbox, (-1, 5 + self.NUM_CLASS)),
                                        np.reshape(pred_lbbox, (-1, 5 + self.NUM_CLASS))], axis=0)
        else:
            pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.NUM_CLASS)),
                                        np.reshape(pred_mbbox, (-1, 5 + self.NUM_CLASS)),
                                        np.reshape(pred_lbbox, (-1, 5 + self.NUM_CLASS))], axis=0)
        bboxes = utils.postprocess_boxes(pred_bbox, (org_h, org_w), self.config.INPUT_SIZE, self.SCORE_THRESH)
        bboxes = utils.nms(bboxes, self.IOU_THRESH)
        
        return bboxes

    ###########################
    ### YOLO Implementation ###
    ###########################
    def _YOLO(self, input_layer, version=4, is_tiny=False):
        if is_tiny:
            if version == 4:
                return self.YOLOv4_tiny(input_layer)
            elif version == 3:
                return self.YOLOv3_tiny(input_layer)
        else:
            if version == 4:
                return self.YOLOv4(input_layer)
            elif version == 3:
                return self.YOLOv3(input_layer)

    def YOLOv3(self, input_layer):
        route_1, route_2, conv = backbone.darknet53(input_layer)

        conv = common.convolutional(conv, (1, 1, 1024, 512))
        conv = common.convolutional(conv, (3, 3, 512, 1024))
        conv = common.convolutional(conv, (1, 1, 1024, 512))
        conv = common.convolutional(conv, (3, 3, 512, 1024))
        conv = common.convolutional(conv, (1, 1, 1024, 512))

        conv_lobj_branch = common.convolutional(conv, (3, 3, 512, 1024))
        conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 1024, 3 * (self.NUM_CLASS + 5)), activate=False, bn=False)

        conv = common.convolutional(conv, (1, 1, 512, 256))
        conv = common.upsample(conv)

        conv = tf.concat([conv, route_2], axis=-1)

        conv = common.convolutional(conv, (1, 1, 768, 256))
        conv = common.convolutional(conv, (3, 3, 256, 512))
        conv = common.convolutional(conv, (1, 1, 512, 256))
        conv = common.convolutional(conv, (3, 3, 256, 512))
        conv = common.convolutional(conv, (1, 1, 512, 256))

        conv_mobj_branch = common.convolutional(conv, (3, 3, 256, 512))
        conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 512, 3 * (self.NUM_CLASS + 5)), activate=False, bn=False)

        conv = common.convolutional(conv, (1, 1, 256, 128))
        conv = common.upsample(conv)

        conv = tf.concat([conv, route_1], axis=-1)

        conv = common.convolutional(conv, (1, 1, 384, 128))
        conv = common.convolutional(conv, (3, 3, 128, 256))
        conv = common.convolutional(conv, (1, 1, 256, 128))
        conv = common.convolutional(conv, (3, 3, 128, 256))
        conv = common.convolutional(conv, (1, 1, 256, 128))

        conv_sobj_branch = common.convolutional(conv, (3, 3, 128, 256))
        conv_sbbox = common.convolutional(conv_sobj_branch, (1, 1, 256, 3 * (self.NUM_CLASS + 5)), activate=False, bn=False)

        return [conv_sbbox, conv_mbbox, conv_lbbox]

    def YOLOv4(self, input_layer):
        route_1, route_2, conv = backbone.cspdarknet53(input_layer)

        route = conv
        conv = common.convolutional(conv, (1, 1, 512, 256))
        conv = common.upsample(conv)
        route_2 = common.convolutional(route_2, (1, 1, 512, 256))
        conv = tf.concat([route_2, conv], axis=-1)

        conv = common.convolutional(conv, (1, 1, 512, 256))
        conv = common.convolutional(conv, (3, 3, 256, 512))
        conv = common.convolutional(conv, (1, 1, 512, 256))
        conv = common.convolutional(conv, (3, 3, 256, 512))
        conv = common.convolutional(conv, (1, 1, 512, 256))

        route_2 = conv
        conv = common.convolutional(conv, (1, 1, 256, 128))
        conv = common.upsample(conv)
        route_1 = common.convolutional(route_1, (1, 1, 256, 128))
        conv = tf.concat([route_1, conv], axis=-1)

        conv = common.convolutional(conv, (1, 1, 256, 128))
        conv = common.convolutional(conv, (3, 3, 128, 256))
        conv = common.convolutional(conv, (1, 1, 256, 128))
        conv = common.convolutional(conv, (3, 3, 128, 256))
        conv = common.convolutional(conv, (1, 1, 256, 128))

        route_1 = conv
        conv = common.convolutional(conv, (3, 3, 128, 256))
        conv_sbbox = common.convolutional(conv, (1, 1, 256, 3 * (self.NUM_CLASS + 5)), activate=False, bn=False)

        conv = common.convolutional(route_1, (3, 3, 128, 256), downsample=True)
        conv = tf.concat([conv, route_2], axis=-1)

        conv = common.convolutional(conv, (1, 1, 512, 256))
        conv = common.convolutional(conv, (3, 3, 256, 512))
        conv = common.convolutional(conv, (1, 1, 512, 256))
        conv = common.convolutional(conv, (3, 3, 256, 512))
        conv = common.convolutional(conv, (1, 1, 512, 256))

        route_2 = conv
        conv = common.convolutional(conv, (3, 3, 256, 512))
        conv_mbbox = common.convolutional(conv, (1, 1, 512, 3 * (self.NUM_CLASS + 5)), activate=False, bn=False)

        conv = common.convolutional(route_2, (3, 3, 256, 512), downsample=True)
        conv = tf.concat([conv, route], axis=-1)

        conv = common.convolutional(conv, (1, 1, 1024, 512))
        conv = common.convolutional(conv, (3, 3, 512, 1024))
        conv = common.convolutional(conv, (1, 1, 1024, 512))
        conv = common.convolutional(conv, (3, 3, 512, 1024))
        conv = common.convolutional(conv, (1, 1, 1024, 512))

        conv = common.convolutional(conv, (3, 3, 512, 1024))
        conv_lbbox = common.convolutional(conv, (1, 1, 1024, 3 * (self.NUM_CLASS + 5)), activate=False, bn=False)

        return [conv_sbbox, conv_mbbox, conv_lbbox]

    def YOLOv4_tiny(self, input_layer):
        route_1, conv = backbone.cspdarknet53_tiny(input_layer)

        conv = common.convolutional(conv, (1, 1, 512, 256))

        conv_lobj_branch = common.convolutional(conv, (3, 3, 256, 512))
        conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 512, 3 * (self.NUM_CLASS + 5)), activate=False, bn=False)

        conv = common.convolutional(conv, (1, 1, 256, 128))
        conv = common.upsample(conv)
        conv = tf.concat([conv, route_1], axis=-1)

        conv_mobj_branch = common.convolutional(conv, (3, 3, 128, 256))
        conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 256, 3 * (self.NUM_CLASS + 5)), activate=False, bn=False)

        return [conv_mbbox, conv_lbbox]

    def YOLOv3_tiny(self, input_layer):
        route_1, conv = backbone.darknet53_tiny(input_layer)

        conv = common.convolutional(conv, (1, 1, 1024, 256))

        conv_lobj_branch = common.convolutional(conv, (3, 3, 256, 512))
        conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 512, 3 * (self.NUM_CLASS + 5)), activate=False, bn=False)

        conv = common.convolutional(conv, (1, 1, 256, 128))
        conv = common.upsample(conv)
        conv = tf.concat([conv, route_1], axis=-1)

        conv_mobj_branch = common.convolutional(conv, (3, 3, 128, 256))
        conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 256, 3 * (self.NUM_CLASS + 5)), activate=False, bn=False)

        return [conv_mbbox, conv_lbbox]

    def decode(self, conv_output, output_size, i, FRAMEWORK='tf'):
        if FRAMEWORK == 'trt':
            return self.decode_trt(conv_output, output_size, i=i)
        elif FRAMEWORK == 'tflite':
            return self.decode_tflite(conv_output, output_size, i=i)
        else:
            return self.decode_tf(conv_output, output_size, i=i)

    def decode_train(self, conv_output, output_size, i=0):
        conv_output = tf.reshape(conv_output,
                                (tf.shape(conv_output)[0], output_size, output_size, self.ANCHOR_PER_SCALE, 5 + self.NUM_CLASS))

        conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (2, 2, 1, self.NUM_CLASS),
                                                                            axis=-1)

        xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
        xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
        xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [tf.shape(conv_output)[0], 1, 1, 3, 1])

        xy_grid = tf.cast(xy_grid, tf.float32)

        pred_xy = ((tf.sigmoid(conv_raw_dxdy) * self.XYSCALE[i]) - 0.5 * (self.XYSCALE[i] - 1) + xy_grid) * self.STRIDES[i]
        pred_wh = (tf.exp(conv_raw_dwdh) * self.ANCHORS[i])
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)

        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

    def decode_tf(self, conv_output, output_size, i=0):
        batch_size = tf.shape(conv_output)[0]
        conv_output = tf.reshape(conv_output,
                                (batch_size, output_size, output_size, 3, 5 + self.NUM_CLASS))

        conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (2, 2, 1, self.NUM_CLASS),
                                                                            axis=-1)

        xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
        xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
        xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [batch_size, 1, 1, 3, 1])

        xy_grid = tf.cast(xy_grid, tf.float32)

        pred_xy = ((tf.sigmoid(conv_raw_dxdy) * self.XYSCALE[i]) - 0.5 * (self.XYSCALE[i] - 1) + xy_grid) * self.STRIDES[i]
        pred_wh = (tf.exp(conv_raw_dwdh) * self.ANCHORS[i])
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)

        pred_prob = pred_conf * pred_prob
        pred_prob = tf.reshape(pred_prob, (batch_size, -1, self.NUM_CLASS))
        pred_xywh = tf.reshape(pred_xywh, (batch_size, -1, 4))

        return pred_xywh, pred_prob
        # return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

    def decode_tflite(self, conv_output, output_size, i=0):
        conv_raw_dxdy_0, conv_raw_dwdh_0, conv_raw_score_0,\
        conv_raw_dxdy_1, conv_raw_dwdh_1, conv_raw_score_1,\
        conv_raw_dxdy_2, conv_raw_dwdh_2, conv_raw_score_2 = tf.split(conv_output, (2, 2, 1+self.NUM_CLASS, 2, 2, 1+self.NUM_CLASS,
                                                                                    2, 2, 1+self.NUM_CLASS), axis=-1)

        conv_raw_score = [conv_raw_score_0, conv_raw_score_1, conv_raw_score_2]
        for idx, score in enumerate(conv_raw_score):
            score = tf.sigmoid(score)
            score = score[:, :, :, 0:1] * score[:, :, :, 1:]
            conv_raw_score[idx] = tf.reshape(score, (1, -1, self.NUM_CLASS))
        pred_prob = tf.concat(conv_raw_score, axis=1)

        conv_raw_dwdh = [conv_raw_dwdh_0, conv_raw_dwdh_1, conv_raw_dwdh_2]
        for idx, dwdh in enumerate(conv_raw_dwdh):
            dwdh = tf.exp(dwdh) * self.ANCHORS[i][idx]
            conv_raw_dwdh[idx] = tf.reshape(dwdh, (1, -1, 2))
        pred_wh = tf.concat(conv_raw_dwdh, axis=1)

        xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
        xy_grid = tf.stack(xy_grid, axis=-1)  # [gx, gy, 2]
        xy_grid = tf.expand_dims(xy_grid, axis=0)
        xy_grid = tf.cast(xy_grid, tf.float32)

        conv_raw_dxdy = [conv_raw_dxdy_0, conv_raw_dxdy_1, conv_raw_dxdy_2]
        for idx, dxdy in enumerate(conv_raw_dxdy):
            dxdy = ((tf.sigmoid(dxdy) * self.XYSCALE[i]) - 0.5 * (self.XYSCALE[i] - 1) + xy_grid) * self.STRIDES[i]
            conv_raw_dxdy[idx] = tf.reshape(dxdy, (1, -1, 2))
        pred_xy = tf.concat(conv_raw_dxdy, axis=1)
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
        return pred_xywh, pred_prob
        # return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

    def decode_trt(self, conv_output, output_size, i=0):
        batch_size = tf.shape(conv_output)[0]
        conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + self.NUM_CLASS))

        conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (2, 2, 1, self.NUM_CLASS), axis=-1)

        xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
        xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
        xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [batch_size, 1, 1, 3, 1])

        # x = tf.tile(tf.expand_dims(tf.range(output_size, dtype=tf.float32), axis=0), [output_size, 1])
        # y = tf.tile(tf.expand_dims(tf.range(output_size, dtype=tf.float32), axis=1), [1, output_size])
        # xy_grid = tf.expand_dims(tf.stack([x, y], axis=-1), axis=2)  # [gx, gy, 1, 2]
        # xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [tf.shape(conv_output)[0], 1, 1, 3, 1])

        xy_grid = tf.cast(xy_grid, tf.float32)

        # pred_xy = ((tf.sigmoid(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * \
        #           STRIDES[i]
        pred_xy = (tf.reshape(tf.sigmoid(conv_raw_dxdy), (-1, 2)) * self.XYSCALE[i] - 0.5 * (self.XYSCALE[i] - 1) + tf.reshape(xy_grid, (-1, 2))) * self.STRIDES[i]
        pred_xy = tf.reshape(pred_xy, (batch_size, output_size, output_size, 3, 2))
        pred_wh = (tf.exp(conv_raw_dwdh) * self.ANCHORS[i])
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)

        pred_prob = pred_conf * pred_prob

        pred_prob = tf.reshape(pred_prob, (batch_size, -1, self.NUM_CLASS))
        pred_xywh = tf.reshape(pred_xywh, (batch_size, -1, 4))
        return pred_xywh, pred_prob
        # return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)


    def filter_boxes(self, box_xywh, scores, score_threshold=0.4, input_shape = tf.constant([416,416])):
        scores_max = tf.math.reduce_max(scores, axis=-1)

        mask = scores_max >= score_threshold
        class_boxes = tf.boolean_mask(box_xywh, mask)
        pred_conf = tf.boolean_mask(scores, mask)
        class_boxes = tf.reshape(class_boxes, [tf.shape(scores)[0], -1, tf.shape(class_boxes)[-1]])
        pred_conf = tf.reshape(pred_conf, [tf.shape(scores)[0], -1, tf.shape(pred_conf)[-1]])

        print(class_boxes.shape)
        box_xy, box_wh = tf.split(class_boxes, (2, 2), axis=-1)

        input_shape = tf.cast(input_shape, dtype=tf.float32)

        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]

        box_mins = (box_yx - (box_hw / 2.)) / input_shape
        box_maxes = (box_yx + (box_hw / 2.)) / input_shape
        boxes = tf.concat([
            box_mins[..., 0:1],  # y_min
            box_mins[..., 1:2],  # x_min
            box_maxes[..., 0:1],  # y_max
            box_maxes[..., 1:2]  # x_max
        ], axis=-1)
        # return tf.concat([boxes, pred_conf], axis=-1)
        return (boxes, pred_conf)


    def compute_loss(self, pred, conv, label, bboxes, i=0):
        conv_shape  = tf.shape(conv)
        batch_size  = conv_shape[0]
        output_size = conv_shape[1]
        input_size  = self.STRIDES[i] * output_size
        conv = tf.reshape(conv, (batch_size, output_size, output_size, self.ANCHOR_PER_SCALE, 5 + self.NUM_CLASS))

        conv_raw_conf = conv[:, :, :, :, 4:5]
        conv_raw_prob = conv[:, :, :, :, 5:]

        pred_xywh     = pred[:, :, :, :, 0:4]
        pred_conf     = pred[:, :, :, :, 4:5]

        label_xywh    = label[:, :, :, :, 0:4]
        respond_bbox  = label[:, :, :, :, 4:5]
        label_prob    = label[:, :, :, :, 5:]

        giou = tf.expand_dims(utils.bbox_giou(pred_xywh, label_xywh), axis=-1)
        input_size = tf.cast(input_size, tf.float32)

        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
        giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

        iou = utils.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

        respond_bgd = (1.0 - respond_bbox) * tf.cast( max_iou < self.IOU_LOSS_THRESH, tf.float32 )

        conf_focal = tf.pow(tf.abs(respond_bbox - pred_conf), 2)

        conf_loss = conf_focal * (
                respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
                +
                respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
        )

        prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

        giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
        prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))

        return giou_loss, conf_loss, prob_loss
