# test phase
import torch
from torch.autograd import Variable
from net import DenseFuse_net,MedFuse_net
import utils
from args_fusion import args
import numpy as np
import time
import cv2
from os import listdir
import imageio

import os
from PIL import Image

def load_model(path, input_nc, output_nc):

	nest_model = DenseFuse_net(input_nc, output_nc)
	nest_model.load_state_dict(torch.load(path))

	para = sum([np.prod(list(p.size())) for p in nest_model.parameters()])
	type_size = 4
	print('Model {} : params: {:4f}M'.format(nest_model._get_name(), para * type_size / 1000 / 1000))

	nest_model.eval()
	nest_model.cuda()

	return nest_model


def _generate_fusion_image(model, strategy_type, img1, img2):
	# encoder
	# test = torch.unsqueeze(img_ir[:, i, :, :], 1)
	en_r = model.encoder(img1,img2)
	# vision_features(en_r, 'ir')
	# en_v = model.encoder(img2)
	# vision_features(en_v, 'vi')
	# fusion
	# f = model.fusion(en_r, en_v, strategy_type=strategy_type)
	# f = en_v
	# decoder
	# img_fusion = model.decoder(f)
	img_fusion = model.decoder(en_r)
	return img_fusion[0]


def run_demo(model, infrared_path, visible_path, output_path_root, index, fusion_type, network_type, strategy_type, ssim_weight_str, mode):

##multi-focus 大小为520*520
	# if mode == 'L':
	ir_img = utils.get_test_images(infrared_path, height=256, width=256, mode=mode)
	vis_img = utils.get_test_images(visible_path, height=256, width=256, mode=mode)
	# else:
	# 	img_ir = utils.tensor_load_rgbimage(infrared_path)
	# 	img_ir = img_ir.unsqueeze(0).float()
	# 	img_vi = utils.tensor_load_rgbimage(visible_path)
	# 	img_vi = img_vi.unsqueeze(0).float()

	# dim = img_ir.shape
	if args.cuda:
		ir_img = ir_img.cuda()
		vis_img = vis_img.cuda()
	ir_img = Variable(ir_img, requires_grad=False)
	vis_img = Variable(vis_img, requires_grad=False)
	dimension = ir_img.size()
	img_fusion = _generate_fusion_image(model, strategy_type, ir_img, vis_img)
	############################ multi outputs ##############################################
	print(infrared_path)

	##PET image
	file_name = 'fusion_frzz_' + 'pet-mr' + "_"+infrared_path[-7:-4] + '.png'
	file_name = 'fusion_nopre_' + 'pet-mr' + "_"+infrared_path[-7:-4] + '.png'

	##multi-focus
	# file_name = index+'.png'

	output_path = output_path_root + file_name

	if args.cuda:
		img = img_fusion.cpu().clamp(0, 255).data[0].numpy()
	else:
		img = img_fusion.clamp(0, 255).data[0].numpy()
	img = img.transpose(1, 2, 0).astype('uint8')
	utils.save_images(output_path, img)
	print(output_path)

def vision_features(feature_maps, img_type):
	count = 0
	for features in feature_maps:
		count += 1
		for index in range(features.size(1)):
			file_name = 'feature_maps_' + img_type + '_level_' + str(count) + '_channel_' + str(index) + '.png'
			output_path = 'outputs/feature_maps/' + file_name
			map = features[:, index, :, :].view(1,1,features.size(2),features.size(3))
			map = map*255
			# save images
			utils.save_image_test(map, output_path)


def main():
	# run demo
	# test_path = "images/test-RGB/"
	# test_path = "images/IV_images/"
	# test_path = "/home/hww/fusion0820/data/valid/"
	# test_path = "/home/hww/fusion0820/data/case6/"
	test_path = "/home/hww/fusion0820/MF_dataset/Code/rgb_data/case22/"

	network_type = 'densefuse'
	fusion_type = 'auto'  # auto, fusion_layer, fusion_all
	strategy_type_list = ['channel_fusion', 'saliency_mask']  # addition, attention_weight, attention_enhance, adain_fusion, channel_fusion, saliency_mask

	# output_path = './outputs/'
	# output_path = './outputs/pet-mr/'##pet-mr融合结果
	output_path = '/home/hww/fusion0820/densefuse-pytorch-master/models/'

	strategy_type = strategy_type_list[0]

	if os.path.exists(output_path) is False:
		os.mkdir(output_path)

	# in_c = 3 for RGB images; in_c = 1 for gray images
	in_c = 1
	if in_c == 1:
		out_c = in_c
		mode = 'L'
		model_path = args.model_path_gray
	else:
		out_c = in_c
		mode = 'RGB'
		model_path = args.model_path_rgb

	with torch.no_grad():
		print('SSIM weight ----- ' + args.ssim_path[2])
		ssim_weight_str = args.ssim_path[2]
		model = load_model(model_path, in_c, out_c)
		index = 2
	##pet
	for lists in listdir(test_path):##一个model
		if lists.endswith('.png') and lists.find('dg1__en')!=-1:
			pet_path = test_path + lists
			mr_path = test_path+"mr2_"+pet_path[-7:-4]+'.png'

			pet_image = imageio.imread(pet_path)
			ycrcb_image = cv2.cvtColor(pet_image, cv2.COLOR_RGB2YCR_CB)
			y_image = ycrcb_image[:, :, 0]
			im = Image.fromarray(y_image)
			y_path = 'images/IV_images/pet_y_channel/' + "y_" + pet_path[-7:-4] + '.png'
			im.save(y_path)

			pet_image = Image.open("/home/hww/fusion0820/MF_dataset/Code/060 (1).gif")
			infrared_path = y_path
			visible_path = mr_path
			run_demo(model, infrared_path, visible_path, output_path, index, fusion_type, network_type, strategy_type, ssim_weight_str, mode)


		# plt.imshow(y_image)
		# plt.show()
		# for i in range(1):##一个model
		# 	index = i + 1
		# 	im = Image.open(test_path + 'ct1_015' + '.gif')  # 打开gif格式的图片
		# 	ct_path= r'images/IV_images/ct1_015.png'
		# 	im.save(ct_path)
		#
		# 	im = Image.open(test_path + 'mr2_015' + '.gif')  # 打开gif格式的图片
		# 	mr_path = r'images/IV_images/mr2_015.png'
		# 	im.save(mr_path)
		#
		# 	infrared_path = ct_path
		# 	visible_path = mr_path
		# 	run_demo(model, infrared_path, visible_path, output_path, index, fusion_type, network_type, strategy_type, ssim_weight_str, mode)

		##medical_image(ct-mr)
		# for lists in listdir(test_path):##一个model
		# 	if lists.endswith('.jpg') and lists.find('_1.')!=-1:
		# 		infrared_path = test_path + lists
		# 		visible_path = test_path + lists.replace('_1.','_2.')
		# 		run_demo(model, infrared_path, visible_path, output_path, index, fusion_type, network_type, strategy_type, ssim_weight_str, mode)
		# 	if lists.endswith('.gif') and lists.find('ct1')!=-1:
		# 		infrared_path = test_path + lists
		# 		visible_path = test_path + lists.replace('ct1','mr2')
		# 		run_demo(model, infrared_path, visible_path, output_path, index, fusion_type, network_type, strategy_type, ssim_weight_str, mode)

		##multi-focus
		# MF_test = '/home/hww/fusion0820/MF_dataset/Multi-focus-Image-Fusion-Dataset-master (2)/LytroDataset/'
		# # MF_test ='/home/hww/fusion0820/MF_dataset/Multi-focus-Image-Fusion-Dataset-master (2)/NaturalMultifocusImages/'
		# MF_output_path = '/home/hww/fusion0820/densefuse-pytorch-master/outputs/multi-focus/'
		# # MF_output_path = '/home/hww/fusion0820/densefuse-pytorch-master/models/1/'
		# ##LytroDataset
		# for lists in listdir(MF_test):##一个model
		# 	if lists.endswith('.jpg') and lists.find('A.')!=-1:
		# 		infrared_path = MF_test + lists
		# 		visible_path = MF_test + lists.replace('A.','B.')
		# 		index = lists[:-4]
		# 		run_demo(model, infrared_path, visible_path, MF_output_path, index, fusion_type, network_type, strategy_type, ssim_weight_str, mode)
	##Natural
	# for lists in listdir(MF_test):  ##一个model
	# 	if lists.endswith('.tif') and lists.find('_1.') != -1:
	# 		infrared_path = MF_test + lists
	# 		visible_path = MF_test + lists.replace('_1.', '_2.')
	# 		index = lists[:-4]
	# 		run_demo(model, infrared_path, visible_path, MF_output_path, index, fusion_type, network_type,
	# 				 strategy_type, ssim_weight_str, mode)
	# 	elif lists.endswith('.bmp') and lists.find('1.') != -1:
	# 		infrared_path = MF_test + lists
	# 		visible_path = MF_test + lists.replace('1.', '2.')
	# 		index = lists[:-4]
	# 		run_demo(model, infrared_path, visible_path, MF_output_path, index, fusion_type, network_type,
	# 				 strategy_type, ssim_weight_str, mode)
	# 	elif lists.endswith('.jpg') and lists.find('1.') != -1:
	# 		infrared_path = MF_test + lists
	# 		visible_path = MF_test + lists.replace('1.', '2.')
	# 		index = lists[:-4]
	# 		run_demo(model, infrared_path, visible_path, MF_output_path, index, fusion_type, network_type,
	# 				 strategy_type, ssim_weight_str, mode)


		##vis-ir
		# ct_path = r'/home/hww/fusion0820/U2Fusion-master/test_imgs/vis-ir/TNO/ir/8.bmp'
		# mr_path = r'/home/hww/fusion0820/U2Fusion-master/test_imgs/vis-ir/TNO/vis/8.bmp'
		# infrared_path = ct_path	# output_path = './outputs/pet-mr/'##pet-mr融合结果
		# visible_path = mr_path
		# run_demo(model, infrared_path, visible_path, output_path, index, fusion_type, network_type, strategy_type,
		# 		 ssim_weight_str, mode)

	print('Done......')


if __name__ == '__main__':
	main()


