from dataload import Photos
from torchvision.utils import save_image
import numpy as np

grand_NPY = []

if __name__ == '__main__':
    Data=Photos('./aaa')
    
    for i in range(Data.__len__()):
    #    print(TestData[i])
    #    print(TestData[i].shape())
        k = Data[i]
        k = k.view(-1,720)
        k = k.numpy()
        grand_NPY.append(k)
    
    print(grand_NPY)
    np.save('./npy/noise_NPY.npy', grand_NPY)