import numpy as np
import random

def make_colored_squares():
	amount = 15000
	height = 100
	width = 100
	rgb = 3
	offset_x = 0
	offset_y = 0
	square_size = 5

	image = np.random.rand(amount, height, width, rgb)
	result = np.zeros(amount)

	for i in range(amount):
		color = random.randint(0,2)
		result[i] = color
		offset_x = random.randint(0,95)
		offset_y = random.randint(0,95)

		for y in range(5):
			for u in range(5):
				image[i][offset_y:offset_y+square_size][y][offset_x:offset_x+square_size][u] = image[i][offset_y:offset_y+square_size][y][offset_x:offset_x+square_size][u] * 0.6
				image[i][offset_y:offset_y+square_size][y][offset_x:offset_x+square_size][u][color] = 1.0

	return image[:int(amount*0.9)], result[:int(amount*0.9)], image[int(amount*0.9):], result[int(amount*0.9):]

