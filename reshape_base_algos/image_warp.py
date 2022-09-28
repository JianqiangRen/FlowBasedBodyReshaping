# coding=utf-8
# summary:
# author: Jianqiang Ren
# date:
import numpy as np
import cv2
import math
import numba
import time
from math import isnan

@numba.jit(nopython=True,  parallel=False)
def bilinear_interp(x, y, v11, v12, v21, v22):
    result = (v11 * (1 - y) + v12 * y) * (1 - x) + (v21 * (1 - y) + v22 * y) * x
    return result


@numba.jit(nopython=True,  parallel=False)
def image_warp_grid1(rDx, rDy, oriImg, transRatio, width_expand, height_expand):

    srcW = oriImg.shape[1]
    srcH = oriImg.shape[0]

    newImg = oriImg.copy()
    
    for i in range(srcH):
        for j in range(srcW):
            _i = i
            _j = j

            deltaX = rDx[_i, _j]
            deltaY = rDy[_i, _j]


            nx = _j + deltaX * transRatio
            ny = _i + deltaY * transRatio

            if nx >= srcW - width_expand-1:
                if nx > srcW -1:
                    nx = srcW - 1
 
                # if _j < right_bound:
                #     right_bound = _j
                
            if ny >= srcH -height_expand-1:
                if ny > srcH -1:
                    ny = srcH -1
 
                # if _i < bottom_bound:
                #     bottom_bound = _i
            
            if nx < width_expand:
                if nx < 0:
                    nx = 0
 
                # if _j+1 > left_bound:
                #     left_bound = _j+1

            if ny < height_expand:
                if ny < 0:
                    ny = 0
 
                # if _i+1 > top_bound:
                #     top_bound = _i+1
                
            nxi = int(math.floor(nx))
            nyi = int(math.floor(ny))
            nxi1 = int(math.ceil(nx))
            nyi1 = int(math.ceil(ny))


            for ll in range(3):
                newImg[_i , _j , ll] = bilinear_interp(
                    ny - nyi, nx - nxi,
                    oriImg[nyi, nxi, ll],
                    oriImg[nyi, nxi1, ll],
                    oriImg[nyi1, nxi, ll],
                    oriImg[nyi1, nxi1, ll])
    return newImg