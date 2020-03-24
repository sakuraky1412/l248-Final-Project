import os

from imageio import imread
import pkg_resources


_image = {'baboon': 'baboon.png',
          'lenna': 'lenna.png'}

data_dir = pkg_resources.resource_filename('criticalnet', '/data/test/')
tanmay_dir = 'criticalnet/data/tanmay/'

def sample(name='baboon'):
    return imread('{}{}'.format(data_dir,_image[name]), pilmode='F')

def tanmay_images():
    images = []
    image_name_list = []
    image_names = os.listdir(tanmay_dir)
    image_names.sort()
    for img_id, image_name in enumerate(image_names):
        image_path = os.path.join(tanmay_dir, image_name)
        image = imread(image_path, pilmode='F')
        images.append(image)
        image_name_list.append(image_name)
        # print(image_name)
    return images, image_name_list
