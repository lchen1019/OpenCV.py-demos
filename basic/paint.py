import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# img = cv.imwrite("../images/person2.jpg", 0)
# 创建一个空白图像
img = np.zeros((512, 512, 3), np.uint8)
# 画线，要绘制的图形，起点，终点，颜色，宽度
cv.line(img, (0, 0), (512, 512), (255, 0, 0), 10)
# 画圆
cv.circle(img, (256, 256), 60, (0, 0, 255), 4)
# 画矩形
cv.rectangle(img, (100, 100), (400, 400), (0, 0, 255), 4)
# 写字
cv.putText(img, 'hello', (100, 150), cv.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 3)
plt.imshow(img[:, :, ::-1])
plt.show()
