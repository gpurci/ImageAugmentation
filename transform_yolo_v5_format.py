import numpy as np
import cv2
from pathlib import Path # file management

from ImageAugmentation.yolo_v5_format_manipulation import *

#function that calculates the updated location of one coordinate
#after rotation
def rotated_coord(points, M):
  points = np.array(points)
  size = points.shape[0]
  ones = np.ones((size, 1))
  points_ones = np.concatenate((points, ones), axis=1)
  transformed_pt = np.array(M.dot(points_ones.T).T, dtype=np.int32)
  transformed_pt = np.array(list(zip(*transformed_pt)), dtype=np.int32)
  return transformed_pt


#function that calculates the updated locations of the coordinates
#after rotation
def rotationCartesianCoord(l_x0, l_y0, l_x1, l_y1, rotate_matrix):
  #create all 4 points from an rectangle
  l_x01, l_y01, l_x10, l_y10 = l_x0, l_y1, l_x1, l_y0
  #recalculate the coordinate by matrix rotation
  try:
    size = len(l_x0)
  except:
    size = 1
  points = list(zip(l_x0, l_y0))
  l_x0, l_y0 = rotated_coord(points, rotate_matrix)
  points = list(zip(l_x1, l_y1))
  l_x1, l_y1 = rotated_coord(points, rotate_matrix)
  points = list(zip(l_x10, l_y10))
  l_x10, l_y10 = rotated_coord(points, rotate_matrix)
  points = list(zip(l_x01, l_y01))
  l_x01, l_y01 = rotated_coord(points, rotate_matrix)

  l_x0 = np.array([l_x0, l_x1, l_x01, l_x10]).min(axis=0)
  l_x1 = np.array([l_x0, l_x1, l_x01, l_x10]).max(axis=0)
  l_y0 = np.array([l_y0, l_y1, l_y01, l_y10]).min(axis=0)
  l_y1 = np.array([l_y0, l_y1, l_y01, l_y10]).max(axis=0)

  return l_x0, l_y0, l_x1, l_y1

def validObjAfterRotation(lst_label, l_x0, l_y0, l_x1, l_y1, height, width):
  idxs_not_valid_x0 = np.argwhere(list(map(lambda x: (x < 0) or (x >= width), l_x0))).reshape(-1)
  idxs_not_valid_x1 = np.argwhere(list(map(lambda x: (x < 0) or (x >= width), l_x1))).reshape(-1)
  idxs_not_valid_y0 = np.argwhere(list(map(lambda y: (y < 0) or (y >= height), l_y0))).reshape(-1)
  idxs_not_valid_y1 = np.argwhere(list(map(lambda y: (y < 0) or (y >= height), l_y1))).reshape(-1)
  tmp_idx = np.concatenate([idxs_not_valid_x0, 
                            idxs_not_valid_x1,
                            idxs_not_valid_y0,
                            idxs_not_valid_y1], 
                          axis=0)
  unique_idx = np.unique(tmp_idx)
  print('unique_idx', unique_idx)

  idxs_valid = np.array([i for i in range(l_x0.shape[0]) if i not in unique_idx])
  
  return lst_label[idxs_valid], l_x0[idxs_valid], l_y0[idxs_valid], l_x1[idxs_valid], l_y1[idxs_valid]



def getPadingAfterAffineTransform(height, width, rotate_matrix):
  # calculate new coordinate after affine transformation
  n_x0, n_y0, n_x1, n_y1 = rotationCartesianCoord(np.array([0]), np.array([0]), 
                                                  np.array([width+1]), np.array([height+1]), rotate_matrix)
  n_x0, n_y0, n_x1, n_y1 = n_x0[0], n_y0[0], n_x1[0], n_y1[0]

  print('(n_x0 {}, n_y0 {}), (n_x1 {}, n_y1 {})'.format(n_x0, n_y0, n_x1, n_y1))
  # save only negative coordinate for left top point
  if (n_x0 > 0):
    n_x0 = 0
  if (n_y0 > 0):
    n_y0 = 0
  # save only negative coordinate for left top point
  if (n_x1 < 0):
    n_x1 = width - n_x1
  if (n_y1 < 0):
    n_y1 = height - n_y1
  if (n_x1 < width):
    n_x1 = width
  if (n_y1 < height):
    n_y1 = height

  (p_x0, p_y0), (p_x1, p_y1) = (0 - n_x0, 0 - n_y0), (n_x1 - width, n_y1 - height)

  return p_x0, p_y0, p_x1, p_y1

def getCutBorderAfterAffineTransform(height, width, rotate_matrix):
  n_x0, n_y0, n_x1, n_y1 = rotationCartesianCoord(np.array([0]), np.array([0]), 
                                                  np.array([width+1]), np.array([height+1]), rotate_matrix)
  n_x0, n_y0, n_x1, n_y1 = n_x0[0], n_y0[0], n_x1[0], n_y1[0]

  c_x0, c_y0 = 0, 0
  if (n_x0 > 0):
    c_x0 = n_x0
    n_x0 = 0
  if (n_y0 > 0):
    c_y0 = n_y0
    n_y0 = 0
  if (n_x1 < 0):
    n_x1 = 0
  if (n_y1 < 0):
    n_y1 = 0

  c_x1 = n_x1 - n_x0
  c_y1 = n_y1 - n_y0
  return c_x0, c_y0, c_x1, c_y1



def getRotPoints(height, width, nbr_example_per_image):
    center_s = tuple(zip(np.random.randint(low=0, high=height, size=nbr_example_per_image, dtype=int), 
                np.random.randint(low=0, high=width, size=nbr_example_per_image, dtype=int)))
    
    class_angle = np.random.randint(low=0, high=3, size=nbr_example_per_image)
    angle_s_0  = np.random.uniform(low=-11., high= 11., size=nbr_example_per_image)
    angle_s_1  = np.random.uniform(low= 85., high= 95., size=nbr_example_per_image)
    angle_s_2  = np.random.uniform(low=-85., high=-95., size=nbr_example_per_image)
    angle_s_m = zip(angle_s_0, angle_s_1, angle_s_2)

    map_fn = map(lambda inputs: inputs[0][inputs[1]], zip(angle_s_m, class_angle))
    angle_s = np.array(list(map_fn))

    lst_pts1 = []
    lst_pts2 = []
    for center, angle in zip(center_s, angle_s):
      x, y = center
      pts1 = np.float32([[x, y], [x+200, y], [x+200, y+200]])
      rotate_matrix = cv2.getRotationMatrix2D(center=(int(center[0]), int(center[1])), 
                                              angle=angle, 
                                              scale=1.)
      
      pts2 = rotated_coord(pts1, rotate_matrix).T.astype(np.float32)
      lst_pts1.append(pts1)
      lst_pts2.append(pts2)

    return lst_pts1, lst_pts2



def getTransformPoints(height, width, nbr_example_per_image):
  center_s = tuple(zip(np.random.randint(low=0, high=height, size=nbr_example_per_image, dtype=int), 
              np.random.randint(low=0, high=width, size=nbr_example_per_image, dtype=int)))
  
  lst_pts1 = []
  lst_pts2 = []
  for center in center_s:
    x, y = center
    pts1 = np.float32([[x, y], [x+200, y], [x+200, y+200]])
    pts2 = np.float32([[x, y], [x+200, y], [x+200, y+200]])
    pts2[0][0] = np.float32(np.random.randint(low=x-50 , high=x+50 , size=1)[0])
    pts2[0][1] = np.float32(np.random.randint(low=y-50 , high=y+50 , size=1)[0])
    pts2[1][0] = np.float32(np.random.randint(low=x+150, high=x+250, size=1)[0])
    pts2[1][1] = np.float32(np.random.randint(low=y-50 , high=y+50 , size=1)[0])
    pts2[2][0] = np.float32(np.random.randint(low=x+150, high=x+250, size=1)[0])
    pts2[2][1] = np.float32(np.random.randint(low=y+150, high=y+250, size=1)[0])
    lst_pts1.append(pts1)
    lst_pts2.append(pts2)

  return lst_pts1, lst_pts2



def translateYoloV5Format(src_path, dst_path, get_translate_point_fn, nbr_example_per_image):
  # src_path - source path
  # nbr_example_per_image - number of images per row image

  #get all filename of labels yolo v5 format
  path = Path(src_path)
  #get all filename of images
  lst_file_F = list(map(lambda x: str(x), path.glob('images/*')))
  for filename_F in lst_file_F:
    #get all filename of labels
    filename_T = path.joinpath('labels', Path(filename_F).stem)
    filename_T = filename_T.with_suffix(filename_T.suffix + '.txt')
    #get filename to generate new images and labels
    filename = Path(filename_F).stem
    #index for new generated images and labels
    idx_name = 0
    #read image
    img = cv2.imread(filename_F)
    height, width, channels = img.shape
    #read label
    lst_label, lst_center_x, lst_center_y, lst_w, lst_h = readLabelsYoloV5Format(filename_T)
    #transform yolo v5 format to cartesian format
    l_x0, l_y0, l_x1, l_y1 = transformCenter2Cartesian( lst_center_x, 
                                                        lst_center_y, 
                                                        lst_w, 
                                                        lst_h, 
                                                        height, width)
    # calculate points to perform affine transformation
    pts1_s, pts2_s = get_translate_point_fn(height, width, nbr_example_per_image)

    for pts1, pts2 in zip(pts1_s, pts2_s):
      # calculate transform matrix
      rotate_matrix = cv2.getAffineTransform(pts1, pts2)
      # calculate pading to put image in center
      p_x0, p_y0, p_x1, p_y1 = getPadingAfterAffineTransform(height, width, rotate_matrix)
      # calculate cut border to put image in center
      c_x0, c_y0, c_x1, c_y1 = getCutBorderAfterAffineTransform(height, width, rotate_matrix)
      # update transform points
      pts1[:, 0] += p_x0
      pts1[:, 1] += p_y0
      pts2[:, 0] += p_x0
      pts2[:, 1] += p_y0
      # calculate transform matrix
      rotate_matrix = cv2.getAffineTransform(pts1, pts2)
      # add pading to image
      tmp_img = np.pad(img, ((p_y0, p_y1), (p_x0, p_x1), (0, 0)), constant_values=205)
      height, width, channels = tmp_img.shape
      # transform image
      tmp_img = cv2.warpAffine( src=tmp_img,
                                M=rotate_matrix,
                                dsize=(width, height))
      # cut border of image after affine transform
      tmp_img = tmp_img[c_y0:c_y1, c_x0:c_x1]
      # calculate coordinate of object from image after affine tranformation
      new_x0, new_y0, new_x1, new_y1 = l_x0+p_x0, l_y0+p_y0, l_x1+p_x0, l_y1+p_y0
      new_x0, new_y0, new_x1, new_y1 = rotationCartesianCoord(new_x0, new_y0, new_x1, new_y1, rotate_matrix)
      new_x0, new_y0, new_x1, new_y1 = new_x0-c_x0, new_y0-c_y0, new_x1-c_x0, new_y1-c_y0
      # remove the labels that are not in image plan
      new_label, new_x0, new_y0, new_x1, new_y1 = validObjAfterRotation(lst_label, new_x0, new_y0, new_x1, new_y1, height, width)
      # tranform new coordinate in yolo v5 format
      center_x, center_y, w, h = transformCartesian2Center(new_x0, new_y0, 
                                                           new_x1, new_y1, 
                                                           height, width)
      #generate new stem for image and label 
      tmp_stem = '{}{}{}'.format(filename, '.affine_', idx_name)
      idx_name += 1
      #save new image
      name_F = Path(filename_F).with_stem(tmp_stem).name
      name_F = Path(dst_path).joinpath('images', 'name').with_name(name_F)
      cv2.imwrite(str(name_F), tmp_img)
      #save new label
      zip_coord = zip(new_label, center_x, center_y, w, h)
      name_T = Path(filename_T).with_stem(tmp_stem).name
      name_T = Path(dst_path).joinpath('labels', 'name').with_name(name_T)
      writeLabelsYoloV5Format(str(name_T), zip_coord)

