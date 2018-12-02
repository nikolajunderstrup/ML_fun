# import numpy as np
# import random
from keras.utils import np_utils
from matplotlib import pyplot as plt
import h5py
from PIL import ImageGrab, Image, ImageDraw
import numpy as np
import time
import pprint as pp
np.set_printoptions(threshold=np.nan)

def generator_colored_squares():
	while True:
		amount = 100
		height = 100
		width = 100
		rgb = 3
		offset_x = 0
		offset_y = 0
		square_size = 10

		image = np.random.rand(amount, height, width, rgb)
		result = np.zeros(amount)

		for i in range(amount):
			color = random.randint(0,2)
			result[i] = color
			offset_x = random.randint(0,width-square_size)
			offset_y = random.randint(0,height-square_size)

			for y in range(square_size):
				for u in range(square_size):
					image[i][offset_y:offset_y+square_size][y][offset_x:offset_x+square_size][u] = image[i][offset_y:offset_y+square_size][y][offset_x:offset_x+square_size][u] * 0.75
					image[i][offset_y:offset_y+square_size][y][offset_x:offset_x+square_size][u][color] = 1.0

		result = np_utils.to_categorical(result, 3)
		# print(result[10])
		# plt.imshow(image[10])
		# plt.show()
		# exit()
		yield image, result


def generator_wow_hp_images(test_or_validation="training"):
	f = h5py.File('wow-hp.hdf5', 'r')
	start = 0
	end = int( len(f)/2 )

	if test_or_validation == "validation":
		start = int(end*0.8)
	elif test_or_validation == "training": 
		end = int(end*0.8)

	while True:
		for i in range(start, end):
			images = f.get("images{}".format(i)).value
			results = np_utils.to_categorical(f.get("results{}".format(i)).value , 1215)
			# print(results[4])
			# plt.imshow(images[10])
			# plt.show()
			# exit()
			yield images, results

generator_wow_hp_images()

def predict_wow_hp():
	f = h5py.File('wow-hp.hdf5', 'r')
	return f.get("images0").value[:1]


# def predict_wow_hp():
# 	images = []
# 	for i in range(5):
# 		imagegrab = ImageGrab.grab(bbox=(90,70,215,100))
# 		pix = np.array(imagegrab)
# 		images.append(pix)
# 		time.sleep(0.1)
# 	images = np.array(images)
# 	# print(images.shape)
# 	return images

