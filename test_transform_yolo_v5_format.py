import cv2
import numpy as np
from pathlib import Path

def test_rot_points(filename_F, angle, scale):
  #get all filename of labels
  root_data = list(Path(filename_F).parents)[1]
  filename_T = Path(root_data).joinpath('labels', Path(filename_F).stem)
  filename_T = filename_T.with_suffix(filename_T.suffix + '.txt')
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
  
  pts1_s, pts2_s = getTransformPoints(height, width, nbr_example_per_image=1)
  print('l_x0', l_x0)
  for pts1, pts2 in zip(pts1_s, pts2_s):
    print(pts1, pts2)
    rotate_matrix = cv2.getAffineTransform(pts1, pts2)

    l_x0, l_y0, l_x1, l_y1 = rotationCartesianCoord(l_x0, l_y0, l_x1, l_y1, rotate_matrix)

    img = cv2.warpAffine(src=img,
                        M=rotate_matrix,
                        dsize=(height, width))

    for x0, y0, x1, y1 in zip(l_x0, l_y0, l_x1, l_y1):
      # Start coordinate
      # represents the top left corner of rectangle
      start_point = (int(x0), int(y0))
        
      # Ending coordinate
      # represents the bottom right corner of rectangle
      end_point = (int(x1), int(y1))
      # Red color in BGR
      color = (0, 0, 255)
      # Line thickness of 2 px
      thickness = 2
      # Draw a rectangle with red line borders of thickness of 2 px
      print(start_point, end_point)
      img = cv2.rectangle(img, start_point, end_point, color, thickness)

  cv2_imshow(img)





def test_rotationCartesianCoordYoloV5Format(filename_F, angle, scale):
  #get all filename of labels
  root_data = list(Path(filename_F).parents)[1]
  filename_T = Path(root_data).joinpath('labels', Path(filename_F).stem)
  filename_T = filename_T.with_suffix(filename_T.suffix + '.txt')
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
  
  center = (int(np.random.randint(low=0, high=img.shape[0], size=1)[0]), 
            int(np.random.randint(low=0, high=img.shape[1], size=1)[0]))
  print('l_x0', l_x0)
  rotate_matrix = cv2.getRotationMatrix2D(center=center, 
                                          angle=angle, 
                                          scale=scale)
  
  l_x0, l_y0, l_x1, l_y1 = rotationCartesianCoord(l_x0, l_y0, l_x1, l_y1, rotate_matrix)

  img = cv2.warpAffine(src=img,
                      M=rotate_matrix,
                      dsize=(height, width))

  for x0, y0, x1, y1 in zip(l_x0, l_y0, l_x1, l_y1):
    # Start coordinate
    # represents the top left corner of rectangle
    start_point = (int(x0), int(y0))
      
    # Ending coordinate
    # represents the bottom right corner of rectangle
    end_point = (int(x1), int(y1))
    # Red color in BGR
    color = (0, 0, 255)
    # Line thickness of 2 px
    thickness = 2
    # Draw a rectangle with red line borders of thickness of 2 px
    print(start_point, end_point)
    img = cv2.rectangle(img, start_point, end_point, color, thickness)

  cv2_imshow(img)




def test_afinne_rotationCartesianCoordYoloV5Format(filename_F, pts1, pts2):
  #get all filename of labels
  root_data = list(Path(filename_F).parents)[1]
  filename_T = Path(root_data).joinpath('labels', Path(filename_F).stem)
  filename_T = filename_T.with_suffix(filename_T.suffix + '.txt')
  #read image
  img = cv2.imread(filename_F)
  height, width, channels = img.shape
  print('height {}, width {}, channels {}'.format(height, width, channels))
  #read label
  lst_label, lst_center_x, lst_center_y, lst_w, lst_h = readLabelsYoloV5Format(filename_T)
  #transform yolo v5 format to cartesian format
  l_x0, l_y0, l_x1, l_y1 = transformCenter2Cartesian( lst_center_x, 
                                                      lst_center_y, 
                                                      lst_w, 
                                                      lst_h, 
                                                      height, width)
  

  rotate_matrix = cv2.getAffineTransform(pts1, pts2)

  p_x0, p_y0, p_x1, p_y1 = getPadingAfterAffineTransform(height, width, rotate_matrix)
  c_x0, c_y0, c_x1, c_y1 = getCutBorderAfterAffineTransform(height, width, rotate_matrix)
  print('(p_x0 {}, p_y0 {}), (p_x1 {}, p_y1 {})'.format(p_x0, p_y0, p_x1, p_y1))
  print('(c_x0 {}, c_y0 {}), (c_x1 {}, c_y1 {})'.format(c_x0, c_y0, c_x1, c_y1))

  pts1[:, 0] += p_x0
  pts1[:, 1] += p_y0
  pts2[:, 0] += p_x0
  pts2[:, 1] += p_y0

  rotate_matrix = cv2.getAffineTransform(pts1, pts2)

  img = np.pad(img, ((p_y0, p_y1), (p_x0, p_x1), (0, 0)), constant_values=205)
  height, width, channels = img.shape
  img = cv2.warpAffine(src=img,
                      M=rotate_matrix,
                      dsize=(width, height))

  img = img[c_y0:c_y1, c_x0:c_x1]
  
  l_x0, l_y0, l_x1, l_y1 = l_x0+p_x0, l_y0+p_y0, l_x1+p_x0, l_y1+p_y0
  l_x0, l_y0, l_x1, l_y1 = rotationCartesianCoord(l_x0, l_y0, l_x1, l_y1, rotate_matrix)
  l_x0, l_y0, l_x1, l_y1 = l_x0-c_x0, l_y0-c_y0, l_x1-c_x0, l_y1-c_y0
  
  for x0, y0, x1, y1 in zip(l_x0, l_y0, l_x1, l_y1):
    # Start coordinate
    # represents the top left corner of rectangle
    start_point = (int(x0), int(y0))
      
    # Ending coordinate
    # represents the bottom right corner of rectangle
    end_point = (int(x1), int(y1))
    # Red color in BGR
    color = (0, 0, 255)
    # Line thickness of 2 px
    thickness = 2
    # Draw a rectangle with red line borders of thickness of 2 px
    print(start_point, end_point)
    img = cv2.rectangle(img, start_point, end_point, color, thickness)

  cv2_imshow(img)





def test_validObjAfterRotation(filename_F, test_filename, angle, scale):
  #get all filename of labels
  root_data = list(Path(filename_F).parents)[1]
  filename_T = Path(root_data).joinpath('labels', Path(filename_F).stem)
  filename_T = filename_T.with_suffix(filename_T.suffix + '.txt')
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
  
  center = (int(np.random.randint(low=0, high=img.shape[0], size=1)[0]), 
            int(np.random.randint(low=0, high=img.shape[1], size=1)[0]))
  print('l_x0', l_x0)
  rotate_matrix = cv2.getRotationMatrix2D(center=center, 
                                          angle=angle, 
                                          scale=scale)
  
  l_x0, l_y0, l_x1, l_y1 = rotationCartesianCoord(l_x0, l_y0, l_x1, l_y1, rotate_matrix)
  print('l_x0 shape', l_x0.shape)

  lst_label, l_x0, l_y0, l_x1, l_y1 = validObjAfterRotation(lst_label, l_x0, l_y0, l_x1, l_y1, height, width)
  print('l_x0 shape', l_x0.shape)
  print('l_x0', l_x0)

  img = cv2.warpAffine(src=img,
                      M=rotate_matrix,
                      dsize=(height, width))

  for x0, y0, x1, y1 in zip(l_x0, l_y0, l_x1, l_y1):
    # Start coordinate
    # represents the top left corner of rectangle
    start_point = (int(x0), int(y0))
      
    # Ending coordinate
    # represents the bottom right corner of rectangle
    end_point = (int(x1), int(y1))
    # Red color in BGR
    color = (0, 0, 255)
    # Line thickness of 2 px
    thickness = 2
    # Draw a rectangle with red line borders of thickness of 2 px
    print(start_point, end_point)
    img = cv2.rectangle(img, start_point, end_point, color, thickness)

  cv2_imshow(img)
  cv2.imwrite(test_filename, img)
