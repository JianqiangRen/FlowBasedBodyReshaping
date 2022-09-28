import pytorch_ssim
import lpips
import numpy as np
import torch
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

loss_fn_alex = lpips.LPIPS(net='alex').to(device) # best forward scores
# loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization
criterion_ssim = pytorch_ssim.SSIM().to(device)


def cal_lpips_and_ssim(img1, img2):
    '''calculate LIPIS

     img1, img2: [0, 255] ,RGB
     '''
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    tensor1 = torch.from_numpy(img1).cuda()
    tensor2 = torch.from_numpy(img2).cuda()

    # image should be RGB, IMPORTANT: normalized to [-1,1]
    tensor1 = (tensor1 /255.0 -0.5) *2
    tensor2 = (tensor2 /255.0 -0.5) *2

    tensor1 = tensor1.permute((2,0,1)).unsqueeze(0)
    tensor2 = tensor2.permute((2, 0, 1)).unsqueeze(0)

    with torch.no_grad():
        lpips_value = loss_fn_alex(tensor1 ,tensor2)
        ssim_value = criterion_ssim((tensor1 +1.0)/2,(tensor2+1.0)/2)

    return lpips_value.cpu().item(), ssim_value.cpu().item()

def avg_epe(flow1, flow2):
    _flow1 = flow1.astype(np.float32)
    _flow2 = flow2.astype(np.float32)
    # avg_epe = np.sqrt(np.sum((_flow1 - _flow2) ** 2,axis=2))
    assert  _flow2.shape ==_flow1.shape
    avg_epe = np.mean(np.sqrt(np.sum((_flow1 - _flow2) ** 2, axis=-1)))
    return  avg_epe


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