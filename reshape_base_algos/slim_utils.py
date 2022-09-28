# coding=utf-8
# summary:
# author: Jianqiang Ren
# date:

import math
import numpy as np
import cv2
import os
import numba
import time
import random
import torch

def set_valid_cpu_count(valid_cpu_count):
    pid = os.getpid()
    print(f'pid : {pid}')
    cpu_list = "0"
    for i in range(valid_cpu_count - 1):
        cpu_list += f",{i + 1}"
    print(f'cpu_list : {cpu_list}')
    
    os.system(f"taskset -cp {cpu_list} {pid}")


def calc_angle(vec_1, vec_2):
    inner_prod = vec_1[0]*vec_2[0] + vec_1[1]*vec_2[1]
    inner_prod = inner_prod/(math.sqrt(vec_1[0]**2 + vec_1[1]**2))/(math.sqrt(vec_2[0]**2 + vec_2[1]**2))
    return math.acos(inner_prod)/math.pi * 180


def interp_pts(pt_1, pt_2, degree=0.5):
        interp = [pt_1[0] * degree + pt_2[0]*(1 - degree), pt_1[1]* degree + pt_2[1]*(1 - degree)]
        interp = np.array(interp)
        return interp
        

def calc_distance(pt_1, pt_2):
    dis = pt_1 - pt_2
    return math.sqrt(dis[0]**2 + dis[1]**2)


def resize_on_long_side(img, long_side = 800):
    src_height = img.shape[0]
    src_width = img.shape[1]
    
    if src_height > src_width:
        scale = long_side*1.0/src_height
        _img = cv2.resize(img, (int(src_width * scale),  long_side),interpolation=cv2.INTER_LINEAR)
    else:
        scale = long_side * 1.0 / src_width
        _img = cv2.resize(img, (long_side, int(src_height * scale)), interpolation=cv2.INTER_LINEAR)

    return _img, scale

def BoxRotateAtiColockWise90(x, y, w, h, img_height, img_width):
    new_x = y
    new_y = img_width - (x + w)
    new_w = h
    new_h = w
    return new_x, new_y, new_w, new_h


def RotateAntiClockWise90(img):
    trans_img = cv2.transpose(img)
    new_img = cv2.flip(trans_img, 0)
    return new_img


def RotateClockWise90(img):
    trans_img = cv2.transpose(img)
    new_img = cv2.flip(trans_img, 1)
    return new_img


def recurve_search(root_path, all_paths, suffix=['.png']):
    for file in os.listdir(root_path):
        target_file = os.path.join(root_path, file)
        if os.path.isfile(target_file):
            (path, extension) = os.path.splitext(target_file)
            
            if extension in suffix:
                all_paths.append(target_file)
        
        else:
            recurve_search(target_file, all_paths, suffix)




def enlarge_box_tblr(roi_bbox, mask,ratio=0.4, use_long_side =True):
    if roi_bbox is None or None in roi_bbox:
        return [None, None, None, None]
    
    top = roi_bbox[0]
    bottom = roi_bbox[1]
    left = roi_bbox[2]
    right = roi_bbox[3]
    
    roi_width = roi_bbox[3] - roi_bbox[2]
    roi_height = roi_bbox[1] - roi_bbox[0]
    right = left + roi_width
    bottom = top + roi_height
    
    long_side = roi_width if roi_width > roi_height else roi_height
    
    if use_long_side:
        new_left = left - int(long_side * ratio)
    else:
        new_left = left - int(roi_width * ratio)
    new_left = 1 if new_left < 0 else new_left

    if use_long_side:
        new_top = top - int(long_side * ratio)
    else:
        new_top = top - int(roi_height * ratio)
    new_top = 1 if new_top < 0 else new_top

    if use_long_side:
        new_right = right + int(long_side * ratio)
    else:
        new_right = right + int(roi_width * ratio)
    new_right = mask.shape[1]-2 if new_right > mask.shape[1] else new_right

    if use_long_side:
        new_bottom = bottom + int(long_side * ratio)
    else:
        new_bottom = bottom + int(roi_height * ratio)
    new_bottom = mask.shape[0]-2 if new_bottom > mask.shape[0] else new_bottom
    
    bbox = [new_top, new_bottom, new_left, new_right]
    return bbox



def gen_PAF(image, joints):

    assert joints.shape[0] == 18
    assert joints.shape[1] == 3

    org_h = image.shape[0]
    org_w = image.shape[1]
    small_image, resize_scale = resize_on_long_side(image, 120)

    joints[:, :2] = joints[:, :2] * resize_scale

    joint_left = int(np.min(joints, axis=0)[0])
    joint_right = int(np.max(joints, axis=0)[0])
    joint_top = int(np.min(joints, axis=0)[1])
    joint_bottom = int(np.max(joints, axis=0)[1])

    limb_width = min(abs(joint_right - joint_left), abs(joint_bottom - joint_top)) // 6

    if limb_width % 2 ==0:
        limb_width += 1
    kernel_size = limb_width

    part_orders = [(5,11),(2,8), (5,6), (6,7), (2,3),(3,4),(11,12),(12,13),(8,9),(9,10)]

    map_list = []
    mask_list = []
    PAF_all = np.zeros(shape=(small_image.shape[0], small_image.shape[1], 2), dtype=np.float32)
    for c, pair in enumerate(part_orders):
        idx_a_name = pair[0]
        idx_b_name = pair[1]

        jointa = joints[idx_a_name]
        jointb = joints[idx_b_name]

        confidence_threshold = 0.05
        if jointa[2] > confidence_threshold and jointb[2] > confidence_threshold:
            canvas = np.zeros(shape =(small_image.shape[0], small_image.shape[1]), dtype=np.uint8 )

            canvas = cv2.line(
                    canvas,
                    (int(jointa[0] ), int(jointa[1])),
                    (int(jointb[0]), int(jointb[1])),
                    (255,255,255),
                    5
                )

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

            canvas = cv2.dilate(canvas, kernel, 1)
            canvas = cv2.GaussianBlur(canvas, (kernel_size, kernel_size), 0)
            canvas = canvas.astype(np.float32)/255
            PAF = np.zeros(shape=(small_image.shape[0], small_image.shape[1], 2), dtype=np.float32)
            PAF[..., 0] = jointb[0] - jointa[0]
            PAF[..., 1] = jointb[1] - jointa[1]
            mag, ang = cv2.cartToPolar(PAF[..., 0], PAF[..., 1])
            PAF /= (np.dstack((mag, mag)) + 1e-5)

            single_PAF = PAF * np.dstack((canvas,canvas))
            map_list.append(cv2.GaussianBlur(single_PAF, (kernel_size*3, kernel_size*3), 0))

            mask_list.append(cv2.GaussianBlur(canvas.copy(), (kernel_size*3, kernel_size*3), 0))
            PAF_all = PAF_all * (1.0 - np.dstack((canvas,canvas))) + single_PAF

    PAF_all = cv2.GaussianBlur(PAF_all, (kernel_size*3, kernel_size*3), 0)
    PAF_all = cv2.resize(PAF_all, (org_w, org_h), interpolation=cv2.INTER_LINEAR)
    map_list.append(PAF_all)
    return PAF_all, map_list, mask_list



def gen_skeleton_map(joints, stack_mode="column", input_roi_box = None):
    '''
    
    :param image:
    :param joints:
    :param stack_mode:
    :param confidence_threshold:
    :param input_roi_box: if not None, using and finally return it directly;
                          if None, calc it with joint confidence and enlarge it, then return it
    :return:
            joint_map: gray vlue 2.0 means skeleton , 0 means background
    '''
    t1 = time.time()
    if type(joints) == list:
        joints = np.array(joints)
    assert stack_mode == "column" or stack_mode == "depth"

    part_orders = [(2,5),(5,11),(2,8),(8,11),(5,6),(6,7),(2,3),(3,4),(11,12),(12,13),(8,9),(9,10)]

    def link(img, a, b, color, line_width, scale=1.0, x_offset=0,y_offset=0):
        jointa = joints[a]
        jointb = joints[b]

        cv2.line( img, (int((jointa[0] -x_offset)*scale), int((jointa[1]-y_offset)*scale)),
            (int((jointb[0]-x_offset)*scale), int((jointb[1]-y_offset)*scale)),
            color, line_width )


    roi_box = input_roi_box


    roi_box_width = roi_box[3] - roi_box[2]
    roi_box_height = roi_box[1] - roi_box[0]
    short_side_length = min(roi_box_width, roi_box_height)
    line_width = short_side_length // 30

    line_width = max(line_width, 2)
    # kernel_size = line_width * 4
    # kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    # kernel_size = int(kernel_size)
 
    map_cube = np.zeros(shape=(roi_box_height, roi_box_width, len(part_orders)+1),dtype=np.float32)

    use_line_width = min(5, line_width)
    fx = use_line_width * 1.0 / line_width  # fx 最大值为1

    # print(f'map_cube size before resize:{map_cube.shape}')
    if fx <0.99:
        map_cube = cv2.resize(map_cube, (0, 0), fx=fx, fy=fx)
    # print(f'fx:{fx},use_line_width:{use_line_width}')
    # print(f'map_cube size after resize:{map_cube.shape}')

    t1 =time.time()

    for c, pair in enumerate(part_orders):
        tmp = map_cube[..., c].copy()
        link(tmp, pair[0], pair[1], (2.0, 2.0, 2.0),use_line_width, scale=fx, x_offset=roi_box[2], y_offset=roi_box[0])
        map_cube[..., c] = tmp
        
        tmp = map_cube[..., -1].copy()
        link(tmp, pair[0], pair[1], (2.0, 2.0, 2.0),use_line_width, scale=fx, x_offset=roi_box[2], y_offset=roi_box[0])
        map_cube[..., -1] = tmp

    # print('time append: {}ms/frame'.format(int((time.time() - t1) * 1000)))
    # print('map_cube shape:{}'.format(map_cube.shape))
    # use_kernel_size = int(kernel_size * fx)
    # use_kernel_size = use_kernel_size if use_kernel_size % 2 == 1 else use_kernel_size - 1
    # print('time Gaussian Kernel Size: {}'.format(use_kernel_size))
    # t1 = time.time()
    # map_cube = cv2.GaussianBlur(map_cube, (use_kernel_size, use_kernel_size), 0)
    map_cube = cv2.resize(map_cube, (roi_box_width, roi_box_height))
    # print('time GaussianBlur: {}ms/frame'.format(int((time.time() - t1) * 1000)))

    if stack_mode =="depth":
        return map_cube, roi_box
    elif stack_mode =="column":
        joint_maps = []
        for c in range(len(part_orders)+1):
            joint_maps.append(map_cube[...,c])
        joint_map = np.column_stack(joint_maps)

        return joint_map, roi_box

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        

def draw_line(im, points, color, stroke_size=2, closed=False):
    points = points.astype(np.int32)
    for i in range(len(points) - 1):
        cv2.line(im, tuple(points[i]), tuple(points[i + 1]), color, stroke_size)
    if closed:
        cv2.line(im, tuple(points[0]), tuple(points[-1]), color, stroke_size)


def enlarged_bbox(bbox, img_width, img_height, enlarge_ratio=0.2):
    '''
    :param bbox: [xmin,ymin,xmax,ymax]
    :return: bbox: [xmin,ymin,xmax,ymax]
    '''
    
    left = bbox[0]
    top = bbox[1]
    
    right = bbox[2]
    bottom = bbox[3]
    
    roi_width = right - left
    roi_height = bottom - top
    
    new_left = left - int(roi_width * enlarge_ratio)
    new_left = 0 if new_left < 0 else new_left
    
    new_top = top - int(roi_height * enlarge_ratio)
    new_top = 0 if new_top < 0 else new_top
    
    new_right = right + int(roi_width * enlarge_ratio)
    new_right = img_width if new_right > img_width else new_right
    
    new_bottom = bottom + int(roi_height * enlarge_ratio)
    new_bottom = img_height if new_bottom > img_height else new_bottom
    
    bbox = [new_left, new_top, new_right, new_bottom]
    
    bbox = [int(x) for x in bbox]
    
    return bbox


def get_map_fusion_map_cuda(map_list, threshold=1, device=torch.device('cpu')):
    map_list_cuda = [torch.from_numpy(x).to(device) for x in map_list]
    map_concat = torch.stack(tuple(map_list_cuda),dim=-1)

    map_concat = torch.abs(map_concat)

    map_concat[map_concat < threshold] = 0
    map_concat[map_concat > 1e-5] = 1.0

    sum_map = torch.sum(map_concat, dim=2)
    a = torch.ones_like(sum_map)
    acc_map = torch.where(sum_map > 0, a*2.0, torch.zeros_like(sum_map))

    fusion_map = torch.where(sum_map < 0.5, a * 1.5, sum_map)

    fusion_map = fusion_map.float()
    acc_map = acc_map.float()

    fusion_map = fusion_map.cpu().numpy().astype(np.float32)
    acc_map = acc_map.cpu().numpy().astype(np.float32)

    return fusion_map, acc_map



def gen_border_shade(height, width, height_band, width_band):
    height_ratio = height_band * 1.0 / height
    width_ratio = width_band * 1.0 / width
    
    _height_band = int(256 * height_ratio)
    _width_band = int(256 * width_ratio)
    
    canvas = np.zeros((256, 256), dtype=np.float32)
    
    canvas[_height_band // 2:-_height_band // 2, _width_band // 2:-_width_band // 2] = 1.0
    
    canvas = cv2.blur(canvas, (_height_band, _width_band))
    
    canvas = cv2.resize(canvas, (width, height))
    
    return canvas


def get_mask_bbox(mask, threshold = 127):
    '''

    :param mask:
    :return: [x,y,w,h]
    '''

    ret, mask = cv2.threshold(mask, threshold, 1, 0)

    if cv2.countNonZero(mask) == 0:
        return [None, None, None, None]

    col_acc = np.sum(mask, 0)
    row_acc = np.sum(mask, 1)

    col_acc = col_acc.tolist()
    row_acc = row_acc.tolist()

    for x in range(len(col_acc)):
        if col_acc[x] > 0:
            left = x
            break

    for x in range(1,len(col_acc)):
        if col_acc[-x] > 0:
            right = len(col_acc) - x
            break

    for x in range(len(row_acc)):
        if row_acc[x] > 0:
            top = x
            break

    for x in range(1,len(row_acc)):
        if row_acc[-x] > 0:
            bottom = len(row_acc[::-1]) - x
            break
    return [top, bottom, left, right]


def visualize_flow(flow):
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 2] = 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    bgr = bgr*1.0/255
    return bgr.astype(np.float32)



def vis_joints(image, joints, color, show_text=True, confidence_threshold = 0.1):

    part_orders = [(2, 5), (5, 11), (2, 8), (8, 11), (5, 6), (6, 7), (2, 3), (3, 4), (11, 12), (12, 13), (8, 9),
                   (9, 10)]

    abandon_idxs = [0,1,14,15,16, 17]
    # draw joints
    for i, joint in enumerate(joints):
        if i in abandon_idxs:
            continue
        if joint[-1] > confidence_threshold:

            cv2.circle(image, (int(joint[0]), int(joint[1])), 1, color, 2)
            if show_text:
                cv2.putText(image, str(i)+"[{:.2f}]".format(joint[-1]), (int(joint[0]), int(joint[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # draw link
    for pair in part_orders:
        if joints[pair[0]][-1] > confidence_threshold and joints[pair[1]][-1] > confidence_threshold:
            cv2.line( image,
                (int(joints[pair[0]][0]), int(joints[pair[0]][1])),
                (int(joints[pair[1]][0]), int(joints[pair[1]][1])),
                color, 2 )
    return image




def get_heatmap_cv(img, magn, max_flow_mag):
    min_flow_mag = .5
    cv_magn = np.clip(
        255 * (magn - min_flow_mag) / (max_flow_mag - min_flow_mag+1e-7),
        a_min=0,
        a_max=255).astype(np.uint8)
    if img.dtype != np.uint8:
        img = (255 * img).astype(np.uint8)

    heatmap_img = cv2.applyColorMap(cv_magn, cv2.COLORMAP_JET)
    heatmap_img = heatmap_img[..., ::-1]

    h, w = magn.shape
    img_alpha = np.ones((h, w), dtype=np.double)[:, :, None]
    heatmap_alpha = np.clip( magn / (max_flow_mag +1e-7), a_min=1e-7, a_max=1)[:, :, None]**.7
    heatmap_alpha[heatmap_alpha < .2]**.5
    pm_hm = heatmap_img * heatmap_alpha
    pm_img = img * img_alpha
    cv_out = pm_hm + pm_img * (1 - heatmap_alpha)
    cv_out = np.clip(cv_out, a_min=0, a_max=255).astype(np.uint8)

    return cv_out

def save_heatmap_cv(img, flow,supression=2):

    flow_magn = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
    flow_magn -= supression
    flow_magn[flow_magn<=0]=0
    cv_out = get_heatmap_cv(img, flow_magn, np.max(flow_magn)*1.3)
    return cv_out


if __name__ == "__main__":
    print(calc_angle([-3,-3],[0,3]))
    
    img = cv2.imread('../0bc494749e33d195fac533c62ff8b7dd9423-photo.jpg')
    new = RotateAntiClockWise90(img)
    cv2.imwrite('pro.jpg', new)
