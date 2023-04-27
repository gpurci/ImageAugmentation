# -*- coding: utf-8 -*-
"""yolo_v5_format_manipulation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jRerl4xbz9O4V-ojKxUH7gIoeuzTmvzH
"""

import numpy as np
import cv2
import re #read yolo v5 format
from pathlib import Path # file management


#get a dictionary where keys are name of image and elements are a list with all classes from image
def getClassName(str_class_name):
  #create a regular expres for class reprezentation data
  # filename           | all classes for an image
  #20201104_175012.jpg | 4 0 0 4 1 0 3
  reObjScore = re.compile(r"(?P<filename>\S+)\.jpg(?P<score>(\s+(\S+))+)")
  dict_class_name = {}
  for line in str_class_name.strip().split('\n'):
    reScore = reObjScore.match(line)
    #make a dictionary, key is filename and valuare are a list with all classes from image
    dict_class_name[reScore.group('filename')] = reScore.group('score').strip().split(' ')
  return dict_class_name

#get all scores
def getUniqueScore(str_score):
  # str_score - string with all scores
  # filename           | all classes for an image
  #20201104_175012.jpg | 4 0 0 4 1 0 3

  lst_score = []
  dict_score = getClassName(str_score)
  for key in dict_score:
    lst_score.extend(dict_score[key])

  unique_score = np.unique(np.array(lst_score))
  return unique_score




#get a dictionary with all scores of an image for every image
def getLabels(path, dict_score):
  # path       - str: the path of labels yolo v5 format
  # dict_score - dict: all scores for every image, keys - filename, elements - scores

  #get all filename of labeled image
  lst_score_file = list(dict_score.keys())

  #get all filename of labels yolo v5 format
  lst_file = list(Path(path).glob('labels/*'))
  #cast to string
  name_original_labels = list(map(lambda x: str(x), lst_file))
  #truncate filename
  name_score_labels = list(map(lambda x: str(Path(x).stem).split('.')[0][:-4], lst_file))

  dict_obj_classified = {}
  for key, filename in zip(name_score_labels, name_original_labels):
    # check is key in score_file
    if (key in lst_score_file):
      filename_F = Path(filename).stem
      lst_label, lst_center_x, lst_center_y, lst_w, lst_h = readLabelsYoloV5Format(filename)
      dict_obj_classified[filename_F] = {'score': dict_score[key], 'labels':lst_label, 
                                        'center_x':lst_center_x, 'center_y':lst_center_y, 
                                        'w':lst_w, 'h':lst_h}
    else:
      pass
  return dict_obj_classified


#function to group the detected object by group of class 
def groupDetectObjYoloV5Format(str_class_name, src_path, dst_path, height, width):
  dict_score = getClassName(str_class_name)
  #print(dict_score)
  dict_obj_classified = getLabels(src_path, dict_score)
  #print(dict_obj_classified.keys())
  lst_filename_img = list(map(lambda x: str(x), Path(src_path).glob('images/*')))
  name_new_img = 0
  for filename in lst_filename_img:
    key_name = Path(filename).stem
    print(filename)
    try:
      img = cv2.imread(filename)
    except:
      pass
    try:
      lst_X = cutBoxObjImage(img, dict_obj_classified[key_name], height, width)
      lst_y = dict_obj_classified[key_name]['score']
    except:
      lst_X, lst_y = [], []

    for x, y in zip(lst_X, lst_y):
      filename_clasification = '{}/{}/{}.png'.format(dst_path, y, name_new_img)
      cv2.imwrite(filename_clasification, x)
      name_new_img += 1




#read a file Yolo v5 format and get array of labels, centers, width and height
def readLabelsYoloV5Format(filename):
  # filename - a txt file yolo v5 format
  # can use pandas: pd.read_csv(filename, sep=' ', names=['label', 'center_x', 'center_y', 'w', 'h'])

  #pathern of Yolo v5 format
  reObjLabel = re.compile(r"(?P<label>\S+) (?P<center_x>\S+) (?P<center_y>\S+) (?P<w>\S+) (?P<h>\S+)")

  lst_label = []
  lst_center_x = []
  lst_center_y = []
  lst_w = []
  lst_h = []
  try:
    f = open(filename, "r")
    for line in f:
      #interpretation of yolo v5 format
      reLabel = reObjLabel.match(line)

      label = int(reLabel.group('label'))
      center_x = float(reLabel.group('center_x'))
      center_y = float(reLabel.group('center_y'))
      w = float(reLabel.group('w'))
      h = float(reLabel.group('h'))

      #add to list all class
      lst_label.append(label)
      lst_center_x.append(center_x)
      lst_center_y.append(center_y)
      lst_w.append(w)
      lst_h.append(h)
  except OSError as e:
    print(e.errno)
  finally:
    f.close()

  #cast list to numpy
  np_label, np_center_x = np.array(lst_label), np.array(lst_center_x)
  np_center_y, np_w, np_h = np.array(lst_center_y), np.array(lst_w), np.array(lst_h)

  #get 
    # label - object class
    # center_x - percent of center of object X (width ) axis 
    # center_y - percent of center of object Y (height) axis 
    # w - percent of width  of located object
    # h - percent of height of located object
    # height - height of row image
    # width  - width  of row image
  return np_label, np_center_x, np_center_y, np_w, np_h

#write a file Yolo v5 format and put labels, centers, width and height
def writeLabelsYoloV5Format(filename, zip_elements):
  # filename     - a txt file yolo v5 format
  # zip_elements - labels, centers, width and height
    # label - object class
    # center_x - percent of center of object X (width ) axis 
    # center_y - percent of center of object Y (height) axis 
    # w - percent of width  of located object
    # h - percent of height of located object
  # if use pandas: df.to_string(filename, index=False, header=False)
  try:
    f = open(filename, "w")
    for l, c_x, c_y, w, h in zip_elements:
      data = '{} {} {} {} {}\n'.format(l, c_x, c_y, w, h)
      f.write(data)
  except OSError as e:
    print(e.errno)
  finally:
    f.close()

#transform yolo v5 format to cartesian coordinate
def transformCenter2Cartesian(center_x, center_y, w, h, height, width):
  # center_x - percent of center of object X (width ) axis 
  # center_y - percent of center of object Y (height) axis 
  # w - percent of width  of located object
  # h - percent of height of located object
  # height - height of row image
  # width  - width  of row image

  tmp_center_x = np.array(center_x * width)
  tmp_center_y = np.array(center_y * height)
  tmp_w = np.array(w * width)
  tmp_h = np.array(h * height)
  x0 = np.array((tmp_center_x - tmp_w/2.), dtype=np.uint32)
  y0 = np.array((tmp_center_y - tmp_h/2.), dtype=np.uint32)
  x1 = np.array((x0 + tmp_w), dtype=np.uint32)
  y1 = np.array((y0 + tmp_h), dtype=np.uint32)

  #get
    # x0, y0 - coordinate of leftmost X, topmost Y point of object
    # x1, y1 - coordinate of rightmost X, bottommost Y point of object
  return x0, y0, x1, y1

#transform cartesian coordinate to yolo v5 format
def transformCartesian2Center(x0, y0, x1, y1, height, width):
  # x0, y0 - coordinate of leftmost X, topmost Y point of object
  # x1, y1 - coordinate of rightmost X, bottommost Y point of object
  # height - height of row image
  # width  - width  of row image

  #calculate width and height of object 
  tmp_w = x1 - x0
  tmp_h = y1 - y0

  #calculate center of object 
  tmp_center_x = x0 + tmp_w/2.
  tmp_center_y = y0 + tmp_h/2.

  #calculate percent of center of object
  center_x = tmp_center_x / width
  center_y = tmp_center_y / height

  #calculate percent of width and height of object 
  w, h = tmp_w / width, tmp_h / height

  np_center_x, np_center_y = np.array(center_x), np.array(center_y)
  np_w, np_h = np.array(w), np.array(h)

  #get
    # center_x - percent of center of object X (width ) axis 
    # center_y - percent of center of object Y (height) axis 
    # w - percent of width  of located object
    # h - percent of height of located object
  return np_center_x, np_center_y, np_w, np_h

#
def cutBoxObjImage(img, dict_obj_IO, height, width):
  lst_img = []
  np_x0, np_y0, np_x1, np_y1 = transformCenter2Cartesian(
                                              dict_obj_IO['center_x'], 
                                              dict_obj_IO['center_y'], 
                                              dict_obj_IO['w'], 
                                              dict_obj_IO['h'], 
                                              img.shape[0], img.shape[1])
  
  for x0, y0, x1, y1 in zip(np_x0, np_y0, np_x1, np_y1):
    # Cropped image of above dimension
    tmp_img = img[y0:y1, x0:x1]
    tmp_img = cv2.resize(tmp_img, (width, height))
    lst_img.append(tmp_img)
  return lst_img

# crop a box from image of all classes
def cutBorderYoloV5Format(path, labels_name):
  # path        - (Path object), the path of (train or test) yolo v5 format,
  #               labels has a group of box that include all classes to have
  #               
  # labels_name - dict of labels name, keys name, elements index of name

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
    #read label
    lst_label, lst_center_x, lst_center_y, lst_w, lst_h = readLabelsYoloV5Format(filename_T)

    #group by border, 
    idxs_border = np.argwhere(np.array(lst_label) == labels_name['cut'])
    idxs_border = idxs_border.reshape(idxs_border.shape[0])
    #group by all classes
    idxs_score = np.argwhere(np.array(lst_label) != labels_name['cut'])
    idxs_score = idxs_score.reshape(idxs_score.shape[0])
    #remove group border from all labels
    zip_coord = zip(lst_label[idxs_score], 
                    lst_center_x[idxs_score], 
                    lst_center_y[idxs_score], 
                    lst_w[idxs_score], 
                    lst_h[idxs_score])
    writeLabelsYoloV5Format(filename_T, zip_coord)
    #
    l_x0, l_y0, l_x1, l_y1 = transformCenter2Cartesian( lst_center_x, 
                                                        lst_center_y, 
                                                        lst_w, 
                                                        lst_h, 
                                                        img.shape[0], img.shape[1])
    #select cartesian coordinate by class
    s_x0, s_y0, s_x1, s_y1 = l_x0[idxs_score],  l_y0[idxs_score],  l_x1[idxs_score],  l_y1[idxs_score]
    #select cartesian coordinate by border
    c_x0, c_y0, c_x1, c_y1 = l_x0[idxs_border], l_y0[idxs_border], l_x1[idxs_border], l_y1[idxs_border]
    #recalculate the coordinate by new coordinate
    for x0, y0, x1, y1 in zip(c_x0, c_y0, c_x1, c_y1):
      new_x0, new_y0, new_x1, new_y1 = s_x0 - x0, s_y0 - y0, s_x1 - x0, s_y1 - y0
      #crop by border
      tmp_img = img[y0:y1, x0:x1]
      #calculate new coordinate by yolo v5 format
      center_x, center_y, w, h = transformCartesian2Center(new_x0, new_y0, 
                                                           new_x1, new_y1, 
                                                           tmp_img.shape[0], tmp_img.shape[1])
      #generate new stem for image and label 
      tmp_stem = '{}{}{}'.format(filename, '.cut_border_', idx_name)
      idx_name += 1
      #save new image
      cv2.imwrite(str(Path(filename_F).with_stem(tmp_stem)), tmp_img)
      #save new label
      zip_coord = zip(lst_label[idxs_score], center_x, center_y, w, h)
      writeLabelsYoloV5Format(str(Path(filename_T).with_stem(tmp_stem)), zip_coord)



def splitBorderYoloV5Format(src_path, per_x0, per_y0, per_x1, per_y1, nbr_example_per_image):
  # src_path - source path
  # per_x0 - cut percent of left side of image from row image
  # per_y0 - cut percent of top side of image from row image
  # per_x1 - cut percent of right side of image from row image
  # per_y1 - cut percent of bpttom side of image from row image
  # nbr_example_per_image - number of images per row image

  assert (per_x0 < 1.) and (per_x0 > 0.), "per_x0 need to more that #0 and less than #1 but is {}".format(per_x0)
  assert (per_y0 < 1.) and (per_y0 > 0.), "per_y0 need to more that #0 and less than #1 but is {}".format(per_y0)
  assert (per_x1 < 1.) and (per_x1 > 0.), "per_x1 need to more that #0 and less than #1 but is {}".format(per_x1)
  assert (per_y1 < 1.) and (per_y1 > 0.), "per_y1 need to more that #0 and less than #1 but is {}".format(per_y1)

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
    #read label
    lst_label, lst_center_x, lst_center_y, lst_w, lst_h = readLabelsYoloV5Format(filename_T)
    #transform yolo v5 format to cartesian format
    l_x0, l_y0, l_x1, l_y1 = transformCenter2Cartesian( lst_center_x, 
                                                        lst_center_y, 
                                                        lst_w, 
                                                        lst_h, 
                                                        img.shape[0], img.shape[1])

    min_x0 = l_x0.min()
    min_y0 = l_y0.min()
    max_x1 = l_x1.max()
    max_y1 = l_y1.max()


    #recalculate the coordinate by new coordinate
    low_x1 = img.shape[1] - (img.shape[1] - max_x1) * per_x1
    low_y1 = img.shape[0] - (img.shape[0] - max_y1) * per_y1
    for x0, y0, x1, y1 in zip(np.random.randint(low=0, high=int(min_x0 * per_x0), size=nbr_example_per_image), 
                              np.random.randint(low=0, high=int(min_y0 * per_y0), size=nbr_example_per_image), 
                              np.random.randint(low=low_x1, high=img.shape[1], size=nbr_example_per_image), 
                              np.random.randint(low=low_y1, high=img.shape[0], size=nbr_example_per_image)):

      new_x0, new_y0, new_x1, new_y1 = l_x0 - x0, l_y0 - y0, l_x1 - x0, l_y1 - y0
      #crop by border
      tmp_img = img[y0:y1, x0:x1]
      #calculate new coordinate by yolo v5 format
      center_x, center_y, w, h = transformCartesian2Center(new_x0, new_y0, 
                                                           new_x1, new_y1, 
                                                           tmp_img.shape[0], tmp_img.shape[1])
      #generate new stem for image and label 
      tmp_stem = '{}{}{}'.format(filename, '.split_', idx_name)
      idx_name += 1
      #save new image
      cv2.imwrite(str(Path(filename_F).with_stem(tmp_stem)), tmp_img)
      #save new label
      zip_coord = zip(lst_label, center_x, center_y, w, h)
      writeLabelsYoloV5Format(str(Path(filename_T).with_stem(tmp_stem)), zip_coord)


