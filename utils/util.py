import os
import numpy as np
import random
import cv2
import torch
import torchvision
import glob
import torch.distributed as dist
from config.config import CONFIG
import torch.nn.functional as F
import math
import pytorch_ssim
import lpips
from reshape_base_algos.slim_utils import visualize_flow
loss_fn_alex = lpips.LPIPS(net='alex').cuda() # best forward scores
# loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization
criterion_ssim = pytorch_ssim.SSIM().cuda()

def rand_simple_crop(image, input_crop_size, label=None):

    image_h, image_w = image.shape[0], image.shape[1]

    if input_crop_size[0] >= image_w:
        x_start = -1
    else:
        x_start = random.randint(0, image_w - input_crop_size[0])

    if input_crop_size[1] >= image_h:
        y_start = -1
    else:
        y_start = random.randint(0, image_h - input_crop_size[1])

    if x_start < 0 and y_start < 0:
        if label is None:
            return image, None
        else:
            return image, label

    x_interval = [0, image_w] if x_start < 0 else [x_start, x_start + input_crop_size[0]]
    y_interval = [0, image_h] if y_start < 0 else [y_start, y_start + input_crop_size[1]]

    image = image[y_interval[0]:y_interval[1], x_interval[0]:x_interval[1]]

    if label is not None:
        label = label[y_interval[0]:y_interval[1], x_interval[0]:x_interval[1]]
        return image, label
    else:
        return image, None



def rand_valid_image_crop(image, input_crop_size, label=None):
    #input_crop_size (newW,newH)

    image_h, image_w = image.shape[0], image.shape[1]

    cur_scale_width = input_crop_size[0] / float(image_w)
    cur_scale_height = input_crop_size[1] / float(image_h)
    cur_scale = max(cur_scale_width, cur_scale_height)

    new_size = (int(cur_scale * image_w + 0.5), int(cur_scale * image_h + 0.5))
    resize_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    if label is not None:
        resize_label = cv2.resize(label, new_size, interpolation=cv2.INTER_NEAREST)
    else:
        resize_label = label

    crop_image, crop_label = rand_simple_crop(resize_image, input_crop_size, resize_label)
    return crop_image, crop_label


def merge_image(foreground, background, mask):
    alpha = np.expand_dims(mask.astype(np.float32), axis=2)
    alpha = alpha/255.
    results = foreground * alpha + background * (1 - alpha)
    return results


def get_latest_ckpt(path):
    try:
        list_of_files = glob.glob(os.path.join(path,'ckpt_*'))
        latest_file = max(list_of_files, key=os.path.getctime)
        return latest_file
    except ValueError:
        return None
    
def get_latest_D(path):
    try:
        list_of_files = glob.glob(os.path.join(path,'D_*'))
        latest_file = max(list_of_files, key=os.path.getctime)
        return latest_file
    except ValueError:
        return None
    

def save_params(state, params):
    state['model_params'] = params
    return state

def load_params(state):
    params = state['model_params']
    del state['model_params']
    return state, params


def resize(img, size=512, strict=False):
    short = min(img.shape[:2])
    scale = size/short
    if not strict:
        img = cv2.resize(img, (round(
            img.shape[1]*scale), round(img.shape[0]*scale)), interpolation=cv2.INTER_NEAREST)
    else:
        img = cv2.resize(img, (size,size), interpolation=cv2.INTER_NEAREST)
    return img

def crop(img, size=512):
    try:
        y, x = random.randint(
            0, img.shape[0]-size), random.randint(0, img.shape[1]-size)
    except Exception as e:
        y, x = 0, 0
    return img[y:y+size, x:x+size, :]


def load_image(filename, size=None, use_crop=False):
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if size:
        img = resize(img, size=size)
    if use_crop:
        img = crop(img, size)
    return img

def reduce_tensor_dict(tensor_dict, mode='mean'):
    """
    average tensor dict over different GPUs
    """
    for key, tensor in tensor_dict.items():
        if tensor is not None:
            tensor_dict[key] = reduce_tensor(tensor, mode)
    return tensor_dict


def reduce_tensor(tensor, mode='mean'):
    """
    average tensor over different GPUs
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if mode == 'mean':
        rt /= CONFIG.world_size
    elif mode == 'sum':
        pass
    else:
        raise NotImplementedError("reduce mode can only be 'mean' or 'sum'")
    return rt

class averageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_grid_image(X, max_count):
    X = X[:max_count]
    X = torchvision.utils.make_grid(X.detach().cpu(), nrow=X.shape[0])
    return X



def make_image(image_list, norm_fn, max_count=1):
    grid_list = [get_grid_image(x, max_count) for x in image_list]
    
    res = torch.cat(tuple(grid_list), dim=1)
    res = norm_fn(res)
    return res.numpy()


def make_image(image_list, flow_list, norm_fn, max_count=8):
    grid_list = [get_grid_image(x, max_count) for x in image_list]
    
    flows = []
    for flow in flow_list:
        flow_figure = flow.permute(0, 3, 1, 2)
        tmp = torch.zeros_like(flow_figure)
        
        flow_figure = torch.cat((flow_figure,tmp[:,:,:,:]), 1)
        flow_figure = flow_figure[:,:3,:,:]
        flow_figure = get_grid_image(flow_figure,max_count)
    
        flow_figure = flow_figure.permute(1,2,0)
     
        flow_figure = flow_figure.detach().cpu().numpy()

        flow_vis = visualize_flow(flow_figure)
        flow_vis = torch.from_numpy(flow_vis)
        flow_vis = flow_vis.permute(2,0,1)
        flow_vis = flow_vis.float()
        flows.append(flow_vis)
    flow_vis = torch.cat(tuple(flows), dim=1)
    

    res = torch.cat(tuple(grid_list), dim=1)
    res = norm_fn(res)
    res = torch.cat((res, flow_vis), dim=1)
    return res.numpy()

def enlarged_bbox(bbox, img_width, img_height, enlarge_ratio = 0.2):
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

    new_right = right + int(roi_width *enlarge_ratio)
    new_right = img_width if new_right > img_width else new_right

    new_bottom = bottom + int(roi_height * enlarge_ratio)
    new_bottom = img_height if new_bottom > img_height else new_bottom

    bbox = [new_left, new_top, new_right, new_bottom]

    bbox = [int(x) for x in bbox]
    
    return bbox


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

def to_tensor(img):
    '''convert numpy.ndarray to torch tensor. \n
        if the image is uint8 , it will be divided by 255;\n
        if the image is uint16 , it will be divided by 65535;\n
        if the image is float , it will not be divided, we suppose your image range should between [0~1] ;\n

    Arguments:
        img {numpy.ndarray} -- image to be converted to tensor.
    '''
    if not _is_numpy_image(img):
        raise TypeError('data should be numpy ndarray. but got {}'.format(type(img)))
    
    if img.ndim == 2:
        img = img[:, :, None]
    
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535
    elif img.dtype in [np.float32, np.float64]:
        img = img.astype(np.float32) / 1
    else:
        raise TypeError('{} is not support'.format(img.dtype))
 
    img = torch.from_numpy(img.transpose((2, 0, 1))).contiguous()
    
    return img


def warp(x, flow, mode='bilinear', padding_mode='zeros', coff=0.2):
    n, c, h, w = x.size()
    yv, xv = torch.meshgrid([torch.arange(h), torch.arange(w)])
    xv = xv.float() / (w - 1) * 2.0 - 1
    yv = yv.float() / (h - 1) * 2.0 - 1

    '''
    grid[0,:,:,0] =
    -1, .....1
    -1, .....1
    -1, .....1

    grid[0,:,:,1] =
    -1,  -1, -1
     ;        ;
     1,   1,  1


    image  -1 ~1       -128~128 pixel
    flow   -0.4~0.4     -51.2~51.2 pixel
    '''

    if torch.cuda.is_available():
        grid = torch.cat((xv.unsqueeze(-1), yv.unsqueeze(-1)), -1).unsqueeze(0).cuda()
    else:
        grid = torch.cat((xv.unsqueeze(-1), yv.unsqueeze(-1)), -1).unsqueeze(0)
    grid_x = grid + 2 * flow * coff
    warp_x = F.grid_sample(x, grid_x, mode=mode, padding_mode=padding_mode)
    return warp_x


def vflip(img):
    return cv2.flip(img, 0)

def hflip(img):
    return cv2.flip(img, 1)

def crop(img, top, left, height, width):
    '''crop image

    Arguments:
        img {ndarray} -- image to be croped
        top {int} -- top size
        left {int} -- left size
        height {int} -- croped height
        width {int} -- croped width
    '''
    if not _is_numpy_image(img):
        raise TypeError('the input image should be numpy ndarray with dimension 2 or 3.'
                        'but got {}'.format(type(img))
                        )
    
    if width < 0 or height < 0 or left < 0 or height < 0:
        raise ValueError('the input left, top, width, height should be greater than 0'
                         'but got left={}, top={} width={} height={}'.format(left, top, width, height)
                         )
    if img.ndim == 2:
        img_height, img_width = img.shape
    else:
        img_height, img_width, _ = img.shape
    if (left + width) > img_width or (top + height) > img_height:
        raise ValueError('the input crop width and height should be small or \
         equal to image width and height. ')
    
    if img.ndim == 2:
        return img[top:(top + height), left:(left + width)]
    elif img.ndim == 3:
        return img[top:(top + height), left:(left + width), :]


# P为线外一点，AB为线段两个端点
def getDist_Point2Line(pointP, pointA, pointB):
    #求直线方程
    A = pointA[1] - pointB[1]
    B = pointB[0] - pointA[0]
    C = pointA[0]*pointB[1] - pointA[1]*pointB[0]
    # 代入点到直线距离公式
    distance = (math.fabs(A*pointP[0] + B*pointP[1] + C)) / ( math.sqrt(A*A + B*B))
    return distance

def get_mask_bbox(mask):
    '''

    :param mask:
    :return: [x,y,w,h]
    '''

    ret, mask = cv2.threshold(mask, 127, 1, 0)

    if cv2.countNonZero(mask) == 0:
        return [None, None, None, None]

    top, bottom, left, right = None, None, None, None
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


def rotate_image(img, theta, borderMode =  cv2.BORDER_REPLICATE, rotM = None):
    # 逆时针旋转theta
    
    h, w = img.shape[:2]
    if rotM is None:
        rotM = cv2.getRotationMatrix2D((w // 2, h // 2), theta * 180.0 / math.pi, 1.0)
    rotated_img = cv2.warpAffine(img, rotM, (w, h), borderMode=borderMode)
    return rotated_img,rotM


def rotate_flow(flow, theta, rotM=None, ang_crack_mask = None):
    # 逆时针旋转theta
    # theta = 1*math.pi/180
    
    h, w = flow.shape[:2]
    mag, ang = cv2.cartToPolar(flow[..., 0] + 1e-8, flow[..., 1] + 1e-8)

    if rotM is None:
        rotM = cv2.getRotationMatrix2D((w // 2, h // 2), theta * 180.0 / math.pi, 1.0)
    mag = cv2.warpAffine(mag, rotM, (w, h))

    if ang_crack_mask is None:
        ang_crack_mask = np.zeros_like(ang)
        ang_crack_mask[ang < 2 * math.pi * 0.25] = 1.0
        ang_crack_mask[ang > 2 * math.pi * 0.75] = 1.0
        ang_crack_mask = cv2.warpAffine(ang_crack_mask, rotM, (w, h))
        ang_crack_mask = cv2.GaussianBlur(ang_crack_mask, (5, 5), 0)

    ang_a = ang.copy()
    ang_b = ang.copy()
    
    ang_b[ang_b < math.pi] += math.pi * 2
    
    ang_a = cv2.warpAffine(ang_a, rotM, (w, h))
    ang_b = cv2.warpAffine(ang_b, rotM, (w, h))
    ang_b = ang_b % (2 * math.pi)
    ang = ang_a * (1.0 - ang_crack_mask) + ang_b * ang_crack_mask
    
    ang -= theta
    
    x, y = cv2.polarToCart(mag, ang, angleInDegrees=False)
    return np.dstack((x, y)), rotM, ang_crack_mask


def normalize_field(x):
    mag, ang = cv2.cartToPolar(x[..., 0] + 1e-5, x[..., 1] + 1e-5)
    n_x = x * 1.0 / np.dstack((mag, mag))
    return n_x



def random_bright(im, delta):
    im = im + delta
    im = im.clip(min=0, max=255)
    return im

def random_contrast( im, alpha):
    im = im * alpha
    im = im.clip(min=0, max=255)
    return im

def random_gray(im):
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    return im


def cal_lpips_and_ssim(img1, img2):
    '''calculate LIPIS

     img1, img2: [0, 255] ,RGB
     '''
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    tensor1 = torch.from_numpy(img1)
    tensor2 = torch.from_numpy(img2)

    # image should be RGB, IMPORTANT: normalized to [-1,1]
    tensor1 = (tensor1 /255.0 -0.5) *2
    tensor2 = (tensor2 /255.0 -0.5) *2

    tensor1 = tensor1.permute((2,0,1)).unsqueeze(0).cuda()
    tensor2 = tensor2.permute((2, 0, 1)).unsqueeze(0).cuda()

    with torch.no_grad():
        lpips_value = loss_fn_alex(tensor1 ,tensor2)
        ssim_value = criterion_ssim((tensor1 +1.0)/2,(tensor2+1.0)/2)

    return lpips_value.cpu().item(), ssim_value.cpu().item()

def psnr(img1, img2):
    '''calculate PSNR

    img1, img2: [0, 255]
    '''
    _img1 = img1.astype(np.float32)
    _img2 = img2.astype(np.float32)
    mse = np.mean((_img1 - _img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

#
# def ssim(img1, img2):
#       C1 = (0.01 * 255)**2
#       C2 = (0.03 * 255)**2
#       img1 = img1.astype(np.float64)
#       img2 = img2.astype(np.float64)
#       kernel = cv2.getGaussianKernel(11, 1.5)
#       window = np.outer(kernel, kernel.transpose())
#       mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5] # valid
#       mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
#       mu1_sq = mu1**2
#       mu2_sq = mu2**2
#       mu1_mu2 = mu1 * mu2
#       sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
#       sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
#       sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
#       ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
#                                   (sigma1_sq + sigma2_sq + C2))
#       return ssim_map.mean()
#
# def calculate_ssim(img1, img2):
#   '''calculate SSIM
#   the same outputs as MATLAB's
#   img1, img2: [0, 255]
#   '''
#   if not img1.shape == img2.shape:
#     raise ValueError('Input images must have the same dimensions.')
#   if img1.ndim == 2:
#     return ssim(img1, img2)
#   elif img1.ndim == 3:
#     if img1.shape[2] == 3:
#       ssims = []
#       for i in range(3):
#         ssims.append(ssim(img1, img2))
#       return np.array(ssims).mean()
#     elif img1.shape[2] == 1:
#       return ssim(np.squeeze(img1), np.squeeze(img2))
#   else:
#     raise ValueError('Wrong input image dimensions.')


if __name__ == "__main__":
    image_path = "/Users/yaoyuan/Develop/AIedit/FlowBased-BodySlim/datasets/bg/00018903.jpg"
    image = cv2.imread(image_path)
    input_crop_size = (1500, 1001)
    crop_image, _ = rand_valid_image_crop(image, input_crop_size)
    cv2.imwrite('crop.jpg', crop_image)

    # fore = cv2.imread("/Users/yaoyuan/Develop/AIedit/FlowBased-BodySlim/f1.jpg")
    # alpha = cv2.imread("/Users/yaoyuan/Develop/AIedit/FlowBased-BodySlim/f1.png", -1)
    # results = merge_image(fore, crop_image, alpha)
    # cv2.imwrite("result.jpg", results)
