# -*- coding: utf-8 -*-
import numpy as np
from numpy.fft import fft2, ifft2
import cv2


class AugmentImage():
  def __init__(self):
    self.lst_log = []         #list of log function
    
                              #0  = noise, 1 =  rotationAndNoise, 2 =  Wiener
    self.lst_augmented_func = [self.noise, self.rotationAndNoise, self.Wiener]
  
  def getAugmentFunc(self, idx_func):
    '''
       - idx_func: is index of augmented function
       #0 - noise
       #1 - rotationAndNoise
       #2 - Wiener
    '''
    ls_str_func = ['noise', 'rotationAndNoise', 'Wiener']
    if (idx_func < len(self.lst_augmented_func) and idx_func >= 0):
      self.lst_log.append(ls_str_func[idx_func])
      return self.lst_augmented_func[idx_func]
    else:
      self.lst_log.append(ls_str_func[0])
      return self.lst_augmented_func[0]
  
  #get a 2D array, disc of ones
  @staticmethod
  def disc(size):
    '''
       Function get an quadratic 2D array, representing a disc
       - size: size of array
    '''
    r = int(size / 2)                 #calculate radius
    y,x = np.ogrid[-r: r+1, -r: r+1]
    mask = x**2+y**2 > r**2
    np_disc = np.ones((size, size))
    np_disc[mask] = 0                 #create a disc
    np_disc = np_disc / (np_disc.sum() + 1) #
    return np_disc
  
  
  #get a 2D array, disc of random values
  @staticmethod
  def disc_rand(size):
    '''
       Function get an quadratic 2D array, representing a random disc
       - size: size of array
    '''
    if (size == 1):
      np_disc = np.array([np.random.uniform(low=-0.01, high=0.091, size=1)])
    else:
      r = int(size / 2)
      y,x = np.ogrid[-r: r+1, -r: r+1]
      mask = x**2+y**2 > r**2
                     #create a quadratic matrix with random value
      np_disc = np.random.uniform(low=0.01, high=0.091, size=(size, size))
      np_neg_idx_x, np_neg_idx_y  = np.random.randint(low=0, high=(size-1), size=2).flatten()
      np_disc[np_neg_idx_x, np_neg_idx_y] *= -1.
      np_disc[mask] = 0                 #create a disc
      np_disc = np_disc / (np.absolute(np_disc).sum() + 1)
    return np_disc
  
  #noise function
  def noise(self, np_array):
    '''
       Add a random noise at -10 to 10
       - np_array: raw image
       - tmp_data: noised image
    '''
    size_ = np.prod(np_array.shape)
    tmp_data = np_array.reshape(size_)
    #get noise
    np_noise = np.random.randint(low=-10, high=10, size=size_)
    #add noise
    tmp_data = np.array(tmp_data + np_noise)
    #standardization
    np_idx_pos = np.argwhere(tmp_data > 255)
    tmp_data[np_idx_pos] = 255
    
    np_idx_neg = np.argwhere(tmp_data < 0)
    tmp_data[np_idx_neg] = 0
    
    tmp_data = np.array(tmp_data, dtype=np.uint8).reshape(np_array.shape)
    return tmp_data



  #rotation function
  def rotationAndNoise(self, np_data):
    '''
       Rotation matrix
       - np_array: raw image
       - tmp_data: rotated image
       - theta: rotation angle
    '''
    theta = np.random.uniform(low=-180., high=180., size=1)[0]
    hight, width, channel = np_data.shape
    M = cv2.getRotationMatrix2D((width / 2, hight / 2), theta, 1)
    tmp_data = cv2.warpAffine(np_data, M, (width, hight))
    tmp_data = self.noise(tmp_data) #add noise
    return tmp_data

  #randomly flips numpy array
  @staticmethod
  def flipArray(np_array):
    '''
       Flip matrix
       - np_array: raw image
       - tmp_data: fliped image
    '''
    flip_arr = [1, -1, 1, -1, -1, 1, 1]
    np.random.shuffle(flip_arr)
    np_array = np_array[::flip_arr[0],::flip_arr[1]]
    return np_array

  #filtre wiener
  def Wiener(self, np_data):
    '''
       Wiener filter
       - np_array: raw image
       - out: filtered image
       - theta: rotation angle
       - size_kernel: size of kernel, size can be [1, 3, 5]
       - isDisc: True - disc function, False - disc_rand function
       - Lamda: signal to noise ratio
    '''
    #get random size of kernel
    size_kernel = [1, 3, 5]
    np.random.shuffle(size_kernel)
    size_kernel = size_kernel[0]
    self.lst_log.append('\t size_kernel:{}'.format(size_kernel))
    #get random distorsion function
    isDisc = ((np.random.randint(low=-10, high=10, size=1)[0]) > 0)
    self.lst_log.append('\t isDisc:{}'.format(isDisc))
    if (isDisc):
      tmp_kernel = self.disc(size_kernel)
      Lambda = np.random.uniform(low=0.21, high=0.9, size=1)[0]
    else:
      tmp_kernel = self.disc_rand(size_kernel)
      Lambda = np.random.uniform(low=0.25, high=0.5, size=1)[0]
    self.lst_log.append('\t tmp_kernel:{}'.format(tmp_kernel))
    self.lst_log.append('\t Lambda:{}'.format(Lambda))

    hight, width, channel = np_data.shape

    pad_hight, pad_width = hight - size_kernel, width - size_kernel
    tmp_kernel = np.pad(tmp_kernel,((0, pad_hight), (0, pad_width)))
    tmp_fft2_kernel = fft2(tmp_kernel)
    w = np.conj(tmp_fft2_kernel) / (np.conj(tmp_fft2_kernel) * tmp_fft2_kernel + Lambda)
    hight, width, channel = np_data.shape
    print(hight, width, channel)
    if (channel == 1):
      out = np.real(ifft2(w * fft2(np_data[:,:,0])))
    else:
      out = np.zeros((np_data.shape))
      for i in range(channel):
        out[:,:,i] = np.real(ifft2(w * fft2(np_data[:,:,i])))

    return out
