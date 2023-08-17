import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
import copy
a = []
height = []
width = []
for i in range(1, 10):
    a.append(cv.imread('0'+str(i)+'.jpg'))
    height.append(a[i-1].shape[0])
    width.append(a[i-1].shape[1])
maxheight, maxwidth = max(height), max(width)
b = []
for i in range(9):
    b.append(cv.resize(a[i], (maxwidth, maxheight)))

original_image = copy.deepcopy(b)
#######################################
#
#        放置处理代码
#        b[i]中存放着图片
#
#######################################

# 方法1 equalizeHist 直方图
for i in range(9):
    for j in range(3):
        b[i][:, :, j] = cv.equalizeHist(b[i][:, :, j])

# 方法2 点操作
# d = [6, 9, 7, 7, 7, 14, 7, 10, 13]  # 放大倍数
# for i in range(9):
#     b[i] = b[i].astype(np.uint16)
#     b[i] *= d[i]
#     b[i] = np.clip(b[i], 0, 255)
#     b[i] = b[i].astype(np.uint8)

# 方法3
# for i in range(9):
#     min_percentile_pixel = np.percentile(b[i], 0)
#     max_percentile_pixel = np.percentile(b[i], 90)
#     b[i][b[i] < min_percentile_pixel] = min_percentile_pixel
#     b[i][b[i] > max_percentile_pixel] = max_percentile_pixel
#     cv.normalize(b[i], b[i], 0, 255, cv.NORM_MINMAX)

# 方法4 gamma变换
# d=[1.55,1.8,1.7,1.65,1.6,2.2,1.65,1.9,2.0]
# for i in range(9):
#     b[i]=cv.cvtColor(b[i],cv.COLOR_BGR2HSV)
#     c=b[i][:,:,2].astype(np.uint32)
#     c=np.power(c,d[i])
#     b[i][:,:,2]=c.clip(0,255).astype(np.uint8)
#     b[i]=cv.cvtColor(b[i],cv.COLOR_HSV2BGR)

# 方法5 BGR->HSV equalizeHist HSV->BGR
# for i in range(9):
#     b[i] = cv.cvtColor(b[i], cv.COLOR_BGR2HSV)
#     b[i][:, :, 2] = cv.equalizeHist(b[i][:, :, 2])
#     b[i] = cv.cvtColor(b[i], cv.COLOR_HSV2BGR)

# 方法5变体 BGR->HLS equalizeHist HLS->BGR
# for i in range(9):
#     b[i] = cv.cvtColor(b[i], cv.COLOR_BGR2HLS)
#     b[i][:, :, 1] = cv.equalizeHist(b[i][:, :, 1])
#     b[i] = cv.cvtColor(b[i], cv.COLOR_HLS2BGR)

# for i in range(9):
#     b[i]=cv.cvtColor(b[i],cv.COLOR_BGR2HLS)

#######################################
#
#            图像拼接及显示
#
#######################################
z = np.ones(b[0].shape)*255
z = z.astype(np.uint8)
c1 = cv.hconcat([b[0], b[1], b[2], b[3], b[4]])
c2 = cv.hconcat([b[5], b[6], b[7], b[8], z])
c = cv.vconcat([c1, c2])
hist = []
for i in range(9):
    temp = []
    for j in range(3):
        temp.append(cv.calcHist(b[i], [j], None, [256], [0, 255]))
    hist.append(temp)
fig = plt.figure()
ax = fig.subplots(2, 5)

for i in range(2):
    for j in range(5):
        if i == 1 and j == 4:
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            ax[i, j].axis('off')
            continue
        for k, color in enumerate(['blue', 'green', 'red']):
            # if k==1 or k==0:
            #     continue
            ax[i, j].plot(range(256), hist[5*i+j][k], color=color)
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
fig.show()
fig.savefig('processed_hist.jpg', dpi=400)
cv.namedWindow('test', cv.WINDOW_NORMAL)
cv.imshow('test', c)
cv.waitKey(0)
cv.destroyAllWindows()
# PSNR
# for i in range(9):
#     MSE=np.mean((original_image[i]/1.0-b[i]/1.0)**2)
#     print(MSE)
#     PSNR=10*np.log10((255**2)/MSE)
#     print(PSNR)
# SSIM
# for i in range(9):
#     s = ssim(original_image[i], b[i],win_size=3)
#     print(s)
# 对比度（标准差）
# for i in range(9):
#     [mean,stddev]=cv.meanStdDev(b[i])
#     print(str(np.mean(stddev)))
# 平均梯度
# for i in range(9):
#     g=0
#     for j in range(maxheight-1):
#         for k in range(maxwidth-1):
#             g += np.sqrt(0.5*((b[i][j, k]-b[i][j+1, k])**2+(b[i][j, k]-b[i][j, k+1])**2))
#     g=np.mean(g/(maxheight*maxwidth))
#     print(g)
cv.imwrite('processed.jpg', c)
