import numpy as np
from dataloader import TrainPhotos,TestPhotos
from torchvision.utils import save_image
import random
import torch
from torchvision import transforms
import cv2

transform = transforms.Compose([ 
    #transforms.Resize((60,100)),                       #注意Totensor()必须在Resize()后，否则会报错
    transforms.ToTensor(),                            # 将图片转换为Tensor,归一化至[0,1]
    #transforms.ToPILImage(),
])

def sp_noise(image,prob):
    '''
    添加椒盐噪声
    prob:噪声比例 
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def gasuss_noise(image, mean=0, var=0.01):
    ''' 
        添加高斯噪声
        mean : 均值 
        var : 方差
    '''
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    #cv.imshow("gasuss", out)
    return out


if __name__ == "__main__":

    TestData=TestPhotos('./dc_img')
    for i in range(TestData.__len__()):
        srcImage = cv2.imread("./dc_img/{}.png".format(i)) 
        #cv2.namedWindow("Original image") 
        #cv2.imshow("Original image", srcImage) 
    
    
        gauss_noiseImage = gasuss_noise(srcImage) #添加高斯噪声 
        #cv2.imshow("Add_GaussianNoise Image",gauss_noiseImage) 
        cv2.imwrite("./aaa/{}.png ".format(i),gauss_noiseImage)
    
      
    
        #cv2.waitKey(0) 
        #cv2.destroyAllWindows()
    #TestData=TestPhotos('./dc_img')
    #for i in range(TestData.__len__()):
    #    numpy_test = TestData[i].numpy()
    #    print(numpy_test)
    #    #numpy_test = TestData[i]
    #    gauss_noiseImage = gasuss_noise(numpy_test)
    #    #gauss_noiseImage = torch.tensor(gauss_noiseImage)
    #    print(gauss_noiseImage)
    #    gauss_noiseImage = transform(gauss_noiseImage)
    #    print(gauss_noiseImage)
    #    save_image(gauss_noiseImage, './aaa/{}.png'.format(i))

