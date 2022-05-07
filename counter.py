# 导入库
import cv2
import numpy as np


def cv_show_image(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)  # 等待时间，单位是毫秒，0代表任意键终止
    cv2.destroyAllWindows()


# 读取图像
im = cv2.imread("images/blob.jpg", cv2.IMREAD_GRAYSCALE)

# 设置SimpleBlobDetector参数
params = cv2.SimpleBlobDetector_Params()

# 改变阈值
# 只会检测minThreshold 和 maxThreshold之间的
params.minThreshold = 0
params.maxThreshold = 300

# 按Color：首先，您需要设置filterByColor = 1。
# 设置blobColor = 0来选择较暗的Blobs，设置blobColor = 255来选择较亮的Blobs。
params.filterByColor = True
params.blobColor = 0

# 根据面积过滤
# 按大小:可以根据大小过滤Blobs，方法是设置参数filterByArea = 1，以及适当的minArea和maxArea值。
# 例如，设置minArea = 100将过滤掉所有像素个数小于100的Blobs。
params.filterByArea = True
params.minArea = 100
params.maxArea = 500

# 根据Circularity过滤，这个参数是(圆度)
# 这只是测量了这个blob与圆的距离。正六边形的圆度比正方形高。
# 要根据圆度进行过滤，设置filterByCircularity = 1。然后设置适当的minCircularity和maxCircularity值。
params.filterByCircularity = True
params.minCircularity = 0.1

# 根据Convexity过滤，这个参数是(凹凸性)
# 凸性定义为(Blob的面积/它的凸包的面积)。现在，凸包的形状是最紧的凸形状，完全包围了形状。
# 设置filterByConvexity = 1，然后设置0≤minConvexity≤1和maxConvexity(≤1)。
params.filterByConvexity = True
params.minConvexity = 0.14
params.maxConvexity = 1

# 根据Inertia过滤,惯性比
# 它衡量的是一个形状的伸长程度。例如，对于圆，这个值是1，对于椭圆，它在0和1之间，对于直线，它是0。
# 初步可以认为是外接矩形的长宽比，圆的外接矩形的长宽相等，椭圆是有长短轴，短轴长度除以长轴长度，介于0~1
# 直线可以认为没有宽度，因此是0
params.filterByInertia = True
params.minInertiaRatio = 0.01

# 创建一个带有参数的检测器
detector = cv2.SimpleBlobDetector_create(params)

# 检测blobs
keypoints = detector.detect(im)
keypoints_array = np.array(keypoints)
print(keypoints_array.shape)

# 用红色圆圈画出检测到的blobs
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS 确保圆的大小对应于blob的大小
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 结果显示
cv_show_image("Keypoints", im_with_keypoints)
