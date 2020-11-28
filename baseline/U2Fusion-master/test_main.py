from __future__ import print_function

import time
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.ndimage
# from Net import Generator, WeightNet
# from scipy.misc import imread, imsave
from imageio import imread, imsave
import imageio
from skimage import transform, data
from glob import glob
from model import Model
from os import listdir
import cv2
from PIL import Image
import matplotlib.image as mpimg


MODEL_SAVE_PATH = './model/model.ckpt'
##pet-ct
# output_path='./results/medical/pet-mr/fusion/'
# path = '/home/hww/fusion0820/MF_dataset/Code/rgb_data/case22'
# path1 = path + '/'
# path2 = path + '/'

##ct-mr
# output_path='./results/medical/'
# path = '/home/hww/fusion0820/data/valid'
# # path = '/home/hww/fusion0820/data/case6'
# path1 = path + '/'
# path2 = path + '/'

# output_path='./results/vis-ir/TNO/'
# path = './test_imgs/vis-ir/TNO/'
# path1 = path + 'vis/'
# path2 = path + 'ir/'

# output_path='./results/vis-ir/RoadScene/'
# path = './test_imgs/vis-ir/RoadScene/'
# path1 = path + 'vis/'
# path2 = path + 'ir/'

# output_path ='./results/medical/'
# path = './test_imgs/medical/'
# path1 = path + 'pet/'
# path2 = path + 'mri/'

# output_path='./results/multi-exposure/dataset1/'
# path = './test_imgs/multi-exposure/dataset1/'
# path1 = path + 'oe/'
# path2 = path + 'ue/'

# output_path='./results/multi-exposure/dataset2/'
# path = './test_imgs/multi-exposure/dataset2/'
# path1 = path + 'oe/'
# path2 = path + 'ue/'

output_path='./results/multi-focus/'
# path = '/home/hww/fusion0820/baseline/PMGI_AAAI2020-master/Code_PMGI/Multi-focus/'
path = '/home/hww/fusion0820/MF_dataset/Multi-focus-Image-Fusion-Dataset-master (2)/NaturalMultifocusImages/'
path1 = path + '/'
path2 = path + '/'




def main():
	print('\nBegin to generate pictures ...\n')
	Format='.png'


	# for i in range(1):
		##ct-mr
		# # file_name1 = path1 + str(i + 1) + Format
		# # file_name2 = path2 + str(i + 1) + Format
		# file_name1 = path1 + "c0"+str(i + 1) +'_1'+ '.tif'
		# file_name2 = path2 +"c0"+ str(i + 1) + '_2'+'.tif'
		# # file_name1 = path1 + "ct1_009" + '.gif'
		# # file_name2 = path2 +"mr2_009" + '.gif'
	# for lists in listdir(path1):##一个model
	# 	if lists.endswith('.png') and lists.find('dg1__en')!=-1:
	# 		pet_path = path1 + lists
	# 		mr_path = path1+"mr2_"+pet_path[-7:-4]+'.png'
	#
	# 		pet_image = imageio.imread(pet_path)
	# 		ycrcb_image = cv2.cvtColor(pet_image, cv2.COLOR_RGB2YCR_CB)
	# 		y_image = ycrcb_image[:, :, 0]
	# 		im = Image.fromarray(y_image)
	# 		y_path = './results/medical/pet-mr/y_path/' + pet_path[-7:-4] + '.png'
	# 		im.save(y_path)
	#
	# 		file_name1 = y_path
	# 		file_name2 = mr_path

	for lists in listdir(path1):##一个model
		if lists.endswith('.tif') and lists.find('a')!=-1 and lists.find('_1')!=-1:
			pet_path = path1 + lists
			mr_path = path2+ lists.replace('_1.', '_2.')

		elif lists.endswith('.bmp') and lists.find('1.') != -1:
			pet_path = path1 + lists
			mr_path = path2+ lists.replace('1.', '2.')

		elif lists.endswith('.jpg') and lists.find('1.') != -1 or lists.endswith('.tif') and lists.find('flower_1') != -1 or lists.endswith('.bmp') and lists.find('_A') != -1 :
			if lists.endswith('.jpg'):
				pet_path = path1 + lists
				mr_path = path2+ lists.replace('1.', '2.')
			elif lists.endswith('.tif'):
				pet_path = path1 + lists
				mr_path = path2+ lists.replace('_1', '_2')
			elif lists.endswith('.bmp'):
				pet_path = path1 + lists
				mr_path = path2+ lists.replace('_A', '_B')

			pet_image = imageio.imread(pet_path)
			ycrcb_image = cv2.cvtColor(pet_image, cv2.COLOR_RGB2YCR_CB)
			y_image = ycrcb_image[:, :, 0]
			im = Image.fromarray(y_image)
			y_path = './results/multi-focus/y_path/img1/' + lists[:-4] + '.png'
			im.save(y_path)

			mr_image = imageio.imread(mr_path)
			ycrcb_image2 = cv2.cvtColor(mr_image, cv2.COLOR_RGB2YCR_CB)
			y2_image = ycrcb_image2[:, :, 0]
			im2 = Image.fromarray(y2_image)
			y2_path = './results/multi-focus/y_path/img2/' + lists[:-4] + '.png'
			im2.save(y2_path)
			pet_path = y_path
			mr_path = y2_path
		else:
			continue


		# pet_image = imageio.imread(pet_path)
		# ycrcb_image = cv2.cvtColor(pet_image, cv2.COLOR_RGB2YCR_CB)
		# y_image = ycrcb_image[:, :, 0]
		# im = Image.fromarray(y_image)
		# y_path = './results/multi-focus/y_path/img1/' + lists[:-4] + '.png'
		# im.save(y_path)
		#
		# mr_image = imageio.imread(mr_path)
		# ycrcb_image2 = cv2.cvtColor(mr_image, cv2.COLOR_RGB2YCR_CB)
		# y2_image = ycrcb_image2[:, :, 0]
		# im2 = Image.fromarray(y2_image)
		# y2_path = './results/multi-focus/y_path/img2/' + lists[:-4] + '.png'
		# im2.save(y2_path)

		# file_name1 = y_path
		# file_name2 = y2_path

		file_name1 = pet_path
		file_name2 = mr_path

		img1 = imread(file_name1) / 255.0
		img2 = imread(file_name2) / 255.0
		print('file1:', file_name1)
		print('file2:', file_name2)

		Shape1 = img1.shape
		h1 = Shape1[0]
		w1 = Shape1[1]
		Shape2 = img2.shape
		h2 = Shape2[0]
		w2 = Shape2[1]
		assert (h1 == h2 and w1 == w2), 'Two images must have the same shape!'
		print('input shape:', img1.shape)
		img1 = img1.reshape([1, h1, w1, 1])
		img2 = img2.reshape([1, h1, w1, 1])

		with tf.Graph().as_default(), tf.Session() as sess:
			M = Model(BATCH_SIZE=1, INPUT_H=h1, INPUT_W=w1, is_training=False)
			# restore the trained model and run the style transferring
			t_list = tf.trainable_variables()
			saver = tf.train.Saver(var_list = t_list)
			model_save_path = MODEL_SAVE_PATH
			print(model_save_path)
			sess.run(tf.global_variables_initializer())
			saver.restore(sess, model_save_path)
			outputs = sess.run(M.generated_img, feed_dict = {M.SOURCE1: img1, M.SOURCE2: img2})
			output = outputs[0, :, :, 0] # 0-1

			# fig = plt.figure()
			# f1 = fig.add_subplot(311)
			# f2 = fig.add_subplot(312)
			# f3 = fig.add_subplot(313)
			# f1.imshow(img1, cmap = 'gray')
			# f2.imshow(img2, cmap = 'gray')
			# f3.imshow(output, cmap = 'gray')
			# plt.show()

			# if not os.path.exists(output_path):
			# 	os.makedirs(output_path)
			# imsave(output_path + pet_path[-7:-4] + Format, output)
			##multi-focus
			if not os.path.exists(output_path):
				os.makedirs(output_path)
			imsave(output_path + lists[:-4] + Format, output)



if __name__ == '__main__':
	main()
