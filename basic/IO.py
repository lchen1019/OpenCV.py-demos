import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# 现实图像的读入，location, mode
img = cv.imread('../images/person2.jpg', 2)

# 使用CV的方式打开，不推荐使用
# cv.imshow("title", img)
# cv.waitKey(0)
# cv.destroyAllWindows()
# plt.show()

# 使用plt打开
plt.imshow(img, cmap=plt.cm.gray)
plt.show()

# 保存
cv.imwrite('../images/person2_gray.jpg', img)
