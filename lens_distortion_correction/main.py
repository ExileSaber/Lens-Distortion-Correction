import cv2
import numpy as np
import glob
# 找棋盘格角点
# 阈值
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# print(cv2.TERM_CRITERIA_EPS,'',cv2.TERM_CRITERIA_MAX_ITER)
# w h分别是棋盘格模板长边和短边规格（角点个数）
w = 5
h = 5

dir = 'original_image\\1.jpg'  # 所有图片数据
re_im = 'original_image\\1.jpg'  # 要矫正的图像

sign_im = 'sign_image' + re_im[14:-4] + '_sign.jpg'
sv_im = 'adjust_image' + re_im[14:-4] + '_adjust.jpg'

# 世界坐标系中的棋盘格点,去掉Z坐标，记为二维矩阵，认为在棋盘格这个平面上Z=0
objp = np.zeros((w*h, 3), np.float32)  # 构造0矩阵，用于存放角点的世界坐标
objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)  # 三维网格坐标划分

# 储存棋盘格角点的世界坐标和图像坐标对
objpoints = []  # 在世界坐标系中的三维点
imgpoints = []  # 在图像平面的二维点

images = glob.glob(dir)

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # cv2.imshow('fin', gray)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # 粗略找到棋盘格角点 这里找到的是这张图片中角点的亚像素点位置，（w,h）为角点规模
    ret, corners = cv2.findChessboardCorners(gray, (w, h))

    # 如果找到足够点对，将其存储起来
    if ret is True:
        # 精确找到角点坐标
        corners = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), criteria)

        # 将正确的objp点放入objpoints中
        objpoints.append(objp)
        imgpoints.append(corners)
        # 将角点在图像上显示
        cv2.drawChessboardCorners(img, (w, h), corners, ret)
        cv2.imwrite(sign_im, img)  # 保存图片
        # cv2.imshow('findCorners', img)
        # cv2.waitKey()
# cv2.destroyAllWindows()

# 标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None, flags=cv2.CALIB_RATIONAL_MODEL)
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 去畸变
img2 = cv2.imread(re_im)
h, w = img2.shape[:2]

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))  # 自由比例参数

dst = cv2.undistort(img2, mtx, dist, None, newcameramtx)

cv2.imwrite(sv_im, dst)  # 保存图片
# cv2.imshow('fin', dst)
# cv2.waitKey()
# cv2.destroyAllWindows()

# 反投影误差
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    total_error += error
print("total error: ", total_error/len(objpoints))

