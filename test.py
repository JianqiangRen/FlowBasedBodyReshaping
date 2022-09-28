# coding=utf-8
# summary:
# author: Jianqiang Ren
# date:

from reshape_base_algos.body_retoucher import BodyRetoucher
import time
import cv2
import argparse
import numpy as np
import glob
import tqdm
import os
import json
import shutil
from utils.eval_util import cal_lpips_and_ssim, psnr
from config.test_config import TESTCONFIG, load_config
import toml


def recurve_search(root_path, all_paths, suffix=[]):
    for file in os.listdir(root_path):
        target_file = os.path.join(root_path, file)
        if os.path.isfile(target_file):
            (path, extension) = os.path.splitext(target_file)
            
            if extension in suffix:
                all_paths.append(target_file)
        else:
            recurve_search(target_file, all_paths, suffix)


if __name__ == "__main__":
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/test_cvpr_setting.toml', required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        load_config(toml.load(f))

    print('TEST CONFIG: \n{}'.format(TESTCONFIG))
    print("loading model:{}".format(TESTCONFIG.reshape_ckpt_path))

    ret = BodyRetoucher.init(reshape_ckpt_path=TESTCONFIG.reshape_ckpt_path,
                             pose_estimation_ckpt=TESTCONFIG.pose_estimation_ckpt,
                             device=0, log_level='error',
                             log_path='test_log.txt',
                             debug_level=0)
    if ret == 0:
        print('init done')
    else:
        print('init error:{}'.format(ret))
        exit(0)

    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())

    if os.path.exists(TESTCONFIG.save_dir):
        shutil.rmtree(TESTCONFIG.save_dir)

    os.makedirs(TESTCONFIG.save_dir, exist_ok=True)

    shutil.copy(args.config, os.path.join(TESTCONFIG.save_dir, os.path.basename(args.config )))

    if os.path.isfile(TESTCONFIG.src_dir):
        img_list = [TESTCONFIG.src_dir]
    elif os.path.exists(os.path.join(TESTCONFIG.src_dir, "src")):
        img_list = glob.glob("{}/*.*g".format(os.path.join(TESTCONFIG.src_dir, "src")))
    else:
        img_list = []
        recurve_search(TESTCONFIG.src_dir, img_list, suffix=['.png', '.jpg', '.jpeg','.JPG'])

    img_list = sorted(img_list)

    lpips_list = []
    ssim_list = []
    psnr_list = []
    epe_list = []

    src_lpips_list = []
    src_ssim_list = []
    src_psnr_list = []

    bad_sample = []

    pbar = tqdm.tqdm(img_list)
    for src_path in pbar:
        print('image_path: {}'.format(src_path))
        basename = os.path.basename(src_path)

        gt_path = os.path.join(TESTCONFIG.gt_dir, basename)

        if os.path.exists(gt_path):
            gt = cv2.imread(gt_path)
        else:
            gt = None

        base = os.path.splitext(basename)[0]

        src_img = cv2.imread(src_path)
        if src_img is None:
            print('Error: src_img is None')
            continue

        t1 = time.time()
        pred, flow = BodyRetoucher.reshape_body(src_img, degree=TESTCONFIG.degree)

        print('time of BodyRetoucher.run: {}ms/frame'.format(int((time.time() - t1) * 1000)))

        if flow is None:
            bad_sample.append(src_path)
            continue

        info = ""


        if gt is not None:
            if gt.shape[:2] != src_img.shape[:2]:
                gt = cv2.resize(gt,(src_img.shape[1], src_img.shape[0]),interpolation=cv2.INTER_LINEAR)

            src_rgb = src_img[:,:,::-1]
            gt_rgb = gt[:,:,::-1]

            pred_rgb = pred[:,:,::-1]

            t2 = time.time()
            psnr_value = psnr(pred_rgb, gt_rgb)
            src_psnr_value = psnr(src_rgb, gt_rgb)


            t2 = time.time()
            lpips_value, ssim_value = cal_lpips_and_ssim(pred_rgb,gt_rgb)
            src_lpips_value, src_ssim_value = cal_lpips_and_ssim(src_rgb, gt_rgb)

            lpips_list.append(lpips_value)
            ssim_list.append(ssim_value)
            psnr_list.append(psnr_value)

            src_lpips_list.append(src_lpips_value)
            src_ssim_list.append(src_ssim_value)
            src_psnr_list.append(src_psnr_value)

            pbar.set_description(info + "pred ssim:{:.4},psnr:{:.4},lpips:{:.4}".format(np.mean(ssim_list),np.mean(psnr_list),np.mean(lpips_list)) + "|src ssim:{:.4},psnr:{:.4},lpips:{:.4}".format(np.mean(src_ssim_list), np.mean(src_psnr_list),
                                                                                 np.mean(src_lpips_list)))

        cv2.imwrite(os.path.join(TESTCONFIG.save_dir,base + "_warp_{}.jpg".format(os.path.basename(TESTCONFIG.reshape_ckpt_path).split('.')[0])),pred)
        cv2.imwrite(os.path.join(TESTCONFIG.save_dir, base + ".jpg"), src_img)

        if gt is not None:
            cv2.imwrite(os.path.join(TESTCONFIG.save_dir, base + "_gt.jpg"), gt)

        if BodyRetoucher._debug_level > 0:
            for i in [0,1]:
                if os.path.exists('pred_{}.jpg'.format(i)):
                    os.rename('pred_{}.jpg'.format(i), base+'_pred_{}.jpg'.format(i))
                    shutil.move(base+'_pred_{}.jpg'.format(i), TESTCONFIG.save_dir)
                    
                if os.path.exists('flow_{}.jpg'.format(i)):
                    os.rename('flow_{}.jpg'.format(i), base + 'flow_{}.jpg'.format(i))
                    shutil.move(base + 'flow_{}.jpg'.format(i), TESTCONFIG.save_dir)
    
                if os.path.exists('flow_{}.jpg'.format(i)):
                    os.rename('flow_{}.jpg'.format(i), base + 'flow_{}.jpg'.format(i))
                    shutil.move(base + 'flow_{}.jpg'.format(i), TESTCONFIG.save_dir)
                
                if os.path.exists('x_fusion_map_{}.jpg'.format(i)):
                    os.rename('x_fusion_map_{}.jpg'.format(i), base + '_x_fusion_map_{}.jpg'.format(i))
                    shutil.move(base + '_x_fusion_map_{}.jpg'.format(i), TESTCONFIG.save_dir)
                
                if os.path.exists('y_fusion_map_{}.jpg'.format(i)):
                    os.rename('y_fusion_map_{}.jpg'.format(i), base + '_y_fusion_map_{}.jpg'.format(i))
                    shutil.move(base + '_y_fusion_map_{}.jpg'.format(i), TESTCONFIG.save_dir)

            os.rename('flow_all.jpg', base + '_flow_all.jpg')
            shutil.move(base + '_flow_all.jpg', TESTCONFIG.save_dir)

    if gt is not None:
        print(f"val count:{len(psnr_list)}")
        print(f"pred mean ssim:{np.mean(ssim_list)}")
        print(f"pred mean psnr:{np.mean(psnr_list)}")
        print(f"pred mean lpips:{np.mean(lpips_list)}")

        print(f"src mean ssim:{np.mean(src_ssim_list)}")
        print(f"src mean psnr:{np.mean(src_psnr_list)}")
        print(f"src mean lpips:{np.mean(src_lpips_list)}")
        print(f"bad sample count:{len(bad_sample)}")
        print(f"bad samples:{bad_sample}")

        cur_time =timestamp
        print(cur_time)
        with open("record_{}_weight_{}.txt".format(cur_time, os.path.basename(TESTCONFIG.reshape_ckpt_path).split('.')[0]),"w") as f:
                f.write("time:{}\n".format(timestamp))
                f.write("count :{}\n".format(len(ssim_list)))
                f.write("ssim of pred/gt :{}\n".format(np.mean(ssim_list)))
                f.write("psnr of pred/gt :{}\n".format(np.mean(psnr_list)))
                f.write("lpips of pred/gt :{}\n".format(np.mean(lpips_list)))

    BodyRetoucher.release()
    print('all done')

