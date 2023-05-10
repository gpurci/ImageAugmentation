


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
    #remove group cut border from all labels
    new_label = lst_label[idxs_score]
    zip_coord = zip(new_label, 
                    lst_center_x[idxs_score], 
                    lst_center_y[idxs_score], 
                    lst_w[idxs_score], 
                    lst_h[idxs_score])
    writeLabelsYoloV5Format(filename_T, zip_coord)
    # transform from yolo v5 format to cartesian coordinate
    l_x0, l_y0, l_x1, l_y1 = transformCenter2Cartesian( lst_center_x, 
                                                        lst_center_y, 
                                                        lst_w, 
                                                        lst_h, 
                                                        img.shape[0], img.shape[1])
    #select cartesian coordinate by object from image
    obj_x0, obj_y0, obj_x1, obj_y1 = l_x0[idxs_score],  l_y0[idxs_score],  l_x1[idxs_score],  l_y1[idxs_score]
    #select cartesian coordinate by cut border class
    c_x0, c_y0, c_x1, c_y1 = l_x0[idxs_border], l_y0[idxs_border], l_x1[idxs_border], l_y1[idxs_border]
    #recalculate the coordinate by new coordinate
    for x0, y0, x1, y1 in zip(c_x0, c_y0, c_x1, c_y1):
      new_x0, new_y0, new_x1, new_y1 = obj_x0 - x0, obj_y0 - y0, obj_x1 - x0, obj_y1 - y0
      #crop by border
      tmp_img = img[y0:y1, x0:x1]
      #calculate new coordinate by yolo v5 format
      center_x, center_y, w, h = transformCartesian2Center(new_x0, new_y0, 
                                                           new_x1, new_y1, 
                                                           tmp_img.shape[0], tmp_img.shape[1])
      #generate new stem for image and label 
      tmp_stem = '{}{}{}'.format(filename, '.cut_', idx_name)
      idx_name += 1
      #save new image
      name_F = Path(filename_F).with_stem(tmp_stem).name
      name_F = Path(dst_path).joinpath('images').with_name(name_F)
      cv2.imwrite(str(name_F), tmp_img)
      #save new label
      zip_coord = zip(new_label, center_x, center_y, w, h)
      name_T = Path(filename_T).with_stem(tmp_stem).name
      name_T = Path(dst_path).joinpath('labels').with_name(name_T)
      writeLabelsYoloV5Format(str(name_T), zip_coord)



def imageDilatationYoloV5Format(src_path, dst_path, per_x0, per_y0, per_x1, per_y1, nbr_example_per_image):
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
      name_F = Path(filename_F).with_stem(tmp_stem).name
      name_F = Path(dst_path).joinpath('images').with_name(name_F)
      cv2.imwrite(str(name_F), tmp_img)
      #save new label
      zip_coord = zip(lst_label, center_x, center_y, w, h)
      name_T = Path(filename_T).with_stem(tmp_stem).name
      name_T = Path(dst_path).joinpath('labels').with_name(name_T)
      writeLabelsYoloV5Format(str(name_T), zip_coord)
