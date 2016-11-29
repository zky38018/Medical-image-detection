import cv2
import numpy as np
#RGB转化成HSV 提取黑色物体

def  yiliao_bg(*args, **kwargs):
    if kw.__contains__('filename'):
        print("filename = %s\n" % kwargs['filename'])
    else:
        raise ValueError("Missing filename parameter")

    img = cv2.imread(kw['filename'])
    pass

#膨胀
#二值化
#A*算法
img=cv2.imread("FX00000.JPG")

hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lower_blue = np.array([0 , 0, 0])
upper_blue = np.array([180 , 255 , 46])
mask = cv2.inRange(hsv, lower_blue, upper_blue)
# 对原图像和掩膜进行位运算
#res = cv2.bitwise_and(img, img, mask=mask)
cv2.imshow("img", img)
cv2.imshow("mask", mask)
#cv2.imshow("res", res)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
dilated = cv2.dilate(mask, kernel)
# 显示膨胀后的图像
#cv2.imshow("Dilated Image", dilated);

#二值化
max1=0 #max1是最亮的点
for i in range(dilated.shape[0]):
    for j in range(dilated.shape[1]):
        if dilated[i,j]>max1:
            max1=dilated[i,j]

print(max1)
for i in range(dilated.shape[0]):
    for j in range(dilated.shape[1]):
        if dilated[i,j]<(0.8*max1):
            dilated[i,j]=0
        else:
            dilated[i,j]=max1

cv2.imshow("Dilated Image", dilated);

#找1点
for i in range(dilated.shape[0]):
    for j in range(dilated.shape[1]):
        if dilated[i,j]==max1:
            if





cv2.waitKey()
cv2.destroyAllWindows()