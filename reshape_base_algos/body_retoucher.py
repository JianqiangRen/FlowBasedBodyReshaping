# coding=utf-8
# summary:
# author: Jianqiang Ren
# date:
import torch
import time
import cv2
import argparse
import numpy as np
from reshape_base_algos.logger import Logger

from reshape_base_algos.slim_utils import resize_on_long_side,  visualize_flow, vis_joints
from reshape_base_algos.person_info import PersonInfo
from reshape_base_algos.image_warp import image_warp_grid1

from config import test_config
from network.flow_generator import FlowGenerator
from pose_estimator.body import Body


class BodyRetoucher(object):
    def __init__(self):
        pass

    @classmethod
    def init(cls,
             reshape_ckpt_path,
             pose_estimation_ckpt,
             device,
             log_level='info',
             log_path='body_liquify.log',
             debug_level=0,
             network_input_H = 256,
             network_input_W = 256):
        
        cls._logger = Logger(filename=log_path, level=log_level)
        cls._logger.logger.info(time.strftime("init on %a %b %d %H:%M:%S %Y", time.localtime()))

        cls._debug_level = debug_level
        cls.pad_border = True

        cls.warp_lib = None

        if device < 0:
            cls.device = torch.device('cpu')
        else:
            cls.device = torch.device("cuda:{}".format(device) if torch.cuda.is_available() else 'cpu')

        cls.pose_esti = Body(pose_estimation_ckpt)

        cls.network_input_H = network_input_H
        cls.network_input_W = network_input_W

        cls.flow_generator = FlowGenerator(n_channels=16)
        cls.flow_generator.name = "FlowGenerator"

        cls.liquify_net = cls.flow_generator.to(cls.device)

        checkpoint = torch.load(reshape_ckpt_path, map_location=cls.device)

        cls.flow_generator.load_state_dict(checkpoint['state_dict'], strict=True)
        cls.flow_generator.eval()

        cls._logger.logger.info('init done')
        return 0

    @classmethod
    def reshape_body(cls, src_img, degree=1.0):
        t1 = time.time()

        cls._logger.logger.info('begin run')
        flow = cls.pred_flow(src_img)
        cls._logger.logger.info('time of inference: {}ms'.format(int((time.time() - t1) * 1000)))

        if flow is None:
            return src_img, None

        cls._logger.logger.info('time of run: {}ms'.format(int((time.time() - t1) * 1000)))
        cls._logger.logger.info("flow size:{}".format(flow.shape))
        cls._logger.logger.info("src_img size:{}".format(src_img.shape))

        assert flow.shape[:2] == src_img.shape[:2]

        if test_config.TESTCONFIG.suppress_bg:
            mag, ang = cv2.cartToPolar(flow[..., 0] + 1e-8, flow[..., 1] + 1e-8)
            mag -= 3
            mag[mag <= 0] = 0

            x, y = cv2.polarToCart(mag, ang, angleInDegrees=False)
            flow = np.dstack((x, y))

        flow *= degree
        t1 = time.time()

        cls._logger.logger.info('src_img.shape: {}ms'.format(src_img.shape))
        pred = cls.warp(src_img, flow)
        cls._logger.logger.info('time of warp: {}ms'.format(int((time.time() - t1) * 1000)))

        pred = np.clip(pred, 0, 255)
        return pred, flow

    @classmethod
    def warp(cls, src_img, flow, show_log=True):
        assert src_img.shape[:2] == flow.shape[:2]
        X_flow = flow[..., 0]
        Y_flow = flow[..., 1]

        X_flow = np.ascontiguousarray(X_flow)
        Y_flow = np.ascontiguousarray(Y_flow)

        pred = image_warp_grid1(X_flow, Y_flow, src_img, 1.0, 0, 0)
        if show_log:
            if cls._debug_level >= 1:
                flow_field_val_vis = visualize_flow(np.dstack((X_flow, Y_flow))) * 255
                cv2.imwrite('flow_all.jpg', flow_field_val_vis)

        return pred


    @classmethod
    def pred_joints(cls, img):
        if img is None:
            cls._logger.logger.error("Img is None")
            return None

        small_src, resize_scale = resize_on_long_side(img, 300)
        body_joints = cls.pose_esti(small_src)

        if body_joints.shape[0] >= 1:
            body_joints[:, :, :2] = body_joints[:, :, :2] / resize_scale

        return body_joints

    @classmethod
    def visualize(cls, img):
        body_joints = cls.pred_joints(img)
        out_img = vis_joints(img, body_joints[0], (255, 0, 0), True)
        return out_img


    @classmethod
    def pred_flow(cls, img):

        body_joints = cls.pred_joints(img)
        small_size = 1200

        if img.shape[0] > small_size or img.shape[1] > small_size:
            _img, _scale = resize_on_long_side(img, small_size)
            body_joints[:, :, :2] = body_joints[:, :, :2] * _scale
        else:
            _img = img

        if body_joints.shape[0] < 1:
            cls._logger.logger.info("noJointDetected")
            return None

        # in this demo, we only reshape one person
        person = PersonInfo(body_joints[0], cls._logger)

        cls._logger.logger.info("joint:shape:{}".format(body_joints.shape))

        with torch.no_grad():
            person_pred = person.pred_flow(_img, cls.flow_generator,  cls.device)

        flow = np.dstack((person_pred['rDx'], person_pred['rDy']))

        scale = img.shape[0] *1.0/ flow.shape[0]

        flow = cv2.resize(flow, (img.shape[1], img.shape[0]))
        flow *= scale

        if cls._debug_level >= 1:
            visual = person.visualize(_img, person_pred['rDx'],
                                      person_pred['rDy'],
                                      0, 0, person_pred['multi_bbox'])

            cv2.imwrite('x_fusion_map_{}.jpg'.format(1), person_pred['x_fusion_map'] * 255)
            cv2.imwrite('y_fusion_map_{}.jpg'.format(1), person_pred['y_fusion_map'] * 255)
            cv2.imwrite('flow_{}.jpg'.format(1), visual['flow'])
            cv2.imwrite('pred_{}.jpg'.format(1), visual['pred'])

        return flow

    @classmethod
    def release(cls):
        pass
