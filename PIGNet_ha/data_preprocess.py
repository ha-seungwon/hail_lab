from pascal import VOCSegmentation
import random
from PIL import Image
import numpy as np
data_path= '/data/ADE/VOCdevkit'
save_path=data_path+'/VOC2012/preprocessing_data'

# VOCSegmentation 클래스로부터 데이터를 불러오는 함수
def load_datasets(root, train=True, crop_size=513):
    dataset = VOCSegmentation(root, train=train, crop_size=crop_size)
    return dataset
# 데이터셋을 불러옵니다.
dataset = load_datasets(data_path, train=True, crop_size=513)
valid_dataset = load_datasets(data_path, train=False, crop_size=513)

import os
# 데이터를 저장하는 함수
def save_data(image, mask, save_path,index):
    # 저장할 경로가 없으면 생성합니다.
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 이미지를 저장합니다.
    image.save(os.path.join(save_path+'/image', f'combined_image_{index}.png'))

    # 마스크를 저장합니다.
    mask.save(os.path.join(save_path+'/mask', f'combined_mask_{index}.png'))
#
#
#
# # 두 이미지를 겹치지 않게 조정하여 일정 비율만큼 겹치게 만드는 함수
# def combine_images(image1, image2, overlap_ratio=0.5):
#     width1, height1 = image1.size
#     width2, height2 = image2.size
#
#     # 두 이미지를 겹치지 않게 배치하기 위해 이미지 크기를 조정합니다.
#     overlap_width = int(width1 * overlap_ratio)
#     overlap_height = int(height1 * overlap_ratio)
#
#     # 두 번째 이미지를 첫 번째 이미지의 오른쪽 하단에 배치합니다.
#     new_width = width1 + width2 - overlap_width
#     new_height = height1 + height2 - overlap_height
#
#     top_left_x = width1 - overlap_width
#     top_left_y = height1 - overlap_height
#     bottom_right_x = width1
#     bottom_right_y = height1
#
#     new_image = Image.new('RGB', (new_width, new_height))
#
#     # 첫 번째 이미지를 새 이미지에 붙입니다.
#     new_image.paste(image1, (0, 0))
#
#
#     # 두 번째 이미지를 겹치는 영역을 고려하여 새 이미지에 붙입니다.
#     new_image.paste(image2, (width1 - overlap_width, height1 - overlap_height))
#
#     for x in range(top_left_x, bottom_right_x):
#         for y in range(top_left_y, bottom_right_y):
#             pixel1 = image1.getpixel((x, y))
#             pixel2 = image2.getpixel((x - (width1 - overlap_width), y - (height1 - overlap_height)))
#             # 픽셀 값이 정수인 경우에 대비하여 튜플로 변환합니다.
#             if isinstance(pixel1, int):
#                 pixel1 = (pixel1, pixel1, pixel1)
#             if isinstance(pixel2, int):
#                 pixel2 = (pixel2, pixel2, pixel2)
#             # 겹치는 부분의 픽셀 값을 더합니다.
#             new_pixel = tuple(p1 + p2 for p1, p2 in zip(pixel1, pixel2))
#             # 새로운 이미지에 픽셀 값을 적용합니다.
#             new_image.putpixel((x, y), new_pixel)
#
#     return new_image\
#
#
# def extract_image_from_mask(original_image, mask_image):
#     # 마스크 이미지를 NumPy 배열로 변환합니다.
#     mask_array = np.array(mask_image)
#
#     # 원본 이미지를 NumPy 배열로 변환합니다.
#     original_array = np.array(original_image)
#
#     # 마스크 이미지에서 픽셀 값이 0이 아닌 영역을 가져옵니다.
#     masked_region = np.where(mask_array != 0)
#
#     # 마스크된 영역의 최소/최대 x 및 y 좌표를 찾습니다.
#     min_y, max_y = min(masked_region[0]), max(masked_region[0])
#     min_x, max_x = min(masked_region[1]), max(masked_region[1])
#
#     # 마스크된 영역을 사용하여 원본 이미지에서 이미지를 추출합니다.
#     extracted_image_array = original_array[min_y:max_y, min_x:max_x, :]
#
#     # NumPy 배열을 이미지 객체로 변환합니다.
#     extracted_image = Image.fromarray(extracted_image_array)
#
#     return extracted_image
# # 데이터셋에서 두 이미지를 선택하고 겹쳐진 이미지를 생성하는 함수
# def generate_combined_image(dataset):
#     # 데이터셋에서 두 개의 이미지를 랜덤하게 선택합니다.
#     index1 = random.randint(0, len(dataset.images) - 1)
#     index2 = random.randint(0, len(dataset.images) - 1)
#
#     image1 = Image.open(dataset.images[index1]).convert('RGB')
#     image2 = Image.open(dataset.images[index2]).convert('RGB')
#
#     # 데이터셋에 있는 ground truth 데이터도 불러옵니다.
#     mask1 = Image.open(dataset.masks[index1]).convert('L')
#     mask2 = Image.open(dataset.masks[index2]).convert('L')
#
#     overlap_ratio = 0.5
#
#     extracted_image = extract_image_from_mask(image2, mask2)
#
#     extracted_image.show()
#
#     return 0
#
#     #combined_image = combine_images(image1, image2, overlap_ratio)
#
#     #combined_mask = combine_images(mask1, mask2, overlap_ratio)
#
#     #return combined_image, combined_mask
#
# overlap_save_path = save_path+'/overlap'
# for i in range(10):
#     print(f"overlap {i} image save ")
#     image = generate_combined_image(valid_dataset)
#
#     #image.save(f"image_{i}.png")
#     #combined_image, combined_mask = generate_combined_image(valid_dataset)
#     # 생성한 데이터를 저장합니다.
#     #save_data(combined_image, combined_mask, overlap_save_path,i)


# import cv2
# import numpy as np
#
# def find_contours(mask):
#     # 마스크에서 윤곽선을 찾습니다.
#     contours, _ = cv2.findContours(np.array(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     # 가장 바깥쪽 윤곽선을 반환합니다.
#     return max(contours, key=cv2.contourArea)
#
# def extract_inner_region(image, contour):
#     # 마스크에서 윤곽선을 다시 그립니다.
#     mask = np.zeros_like(image)
#     cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
#     # 원본 이미지와 마스크를 비트와이즈 AND 연산합니다.
#     result = cv2.bitwise_and(image, mask)
#     return result
# def merge_images(image1, image2, distance):
#     # 이미지1의 크기 가져오기
#     height1, width1, _ = image1.shape
#     # 이미지2의 크기 가져오기
#     height2, width2, _ = image2.shape
#
#     # 이미지1과 이미지2를 합칠 빈 이미지 생성
#     merged_image = np.zeros((max(height1, height2 + distance), width1 + width2 + distance, 3), dtype=np.uint8)
#
#     # 이미지1 복사
#     merged_image[:height1, :width1] = image1
#
#     # 이미지2를 오른쪽 아래로 이동하여 합침
#     merged_image[distance:distance + height2, width1 + distance:width1 + distance + width2] = image2
#
#     return merged_image
# def generate_combined_image(dataset, distance_factor=2):
#     index1 = random.randint(0, len(dataset.images) - 1)
#     index2 = random.randint(0, len(dataset.images) - 1)
#
#     image1 = cv2.imread(dataset.images[index1])
#     image2 = cv2.imread(dataset.images[index2])
#
#     mask1 = cv2.imread(dataset.masks[index1], cv2.IMREAD_GRAYSCALE)
#     mask2 = cv2.imread(dataset.masks[index2], cv2.IMREAD_GRAYSCALE)
#
#     contour2 = find_contours(mask2)
#     inner_region2 = extract_inner_region(image2, contour2)
#     mask_inv = cv2.bitwise_not(mask2)
#     inner_region2[mask_inv == 255] = 0
#
#
#     contour1 = find_contours(mask1)
#     inner_region1 = extract_inner_region(image1, contour1)
#     mask_inv = cv2.bitwise_not(mask1)
#     inner_region1[mask_inv == 255] = 0
#
#
#
#
#     # Resize inner_region2 to match the shape of inner_region1
#     inner_region2_resized = cv2.resize(inner_region2, (inner_region1.shape[1], inner_region1.shape[0]))
#
#     new_image = np.zeros((max(image1.shape[0], image2.shape[0])*2, max(image1.shape[1], image2.shape[1])*2, 3), dtype=np.uint8)
#
#
#     # Calculate the starting positions to paste the overlapping regions in the center of new_image
#     start_row = (new_image.shape[0] - inner_region1.shape[0]) // 2
#     start_col = (new_image.shape[1] - inner_region1.shape[1]) // 2
#
#     # Paste inner_region1 in the center of new_image
#     new_image[start_row:start_row + inner_region1.shape[0],
#     start_col:start_col + inner_region1.shape[1]] = inner_region1
#
#     # Calculate the starting positions to paste inner_region2_resized in the center of new_image
#     start_row_ = int((new_image.shape[0] - inner_region2_resized.shape[0]) // 2 + (distance_factor * 50))
#     start_col_ = int((new_image.shape[1] - inner_region2_resized.shape[1]) // 2 + (distance_factor * 50))
#
#
#     # Calculate the region where two inner_regions overlap
#     overlap_region = np.logical_and(inner_region1 != 0, inner_region2_resized != 0)
#
#     # Set the overlapping region in new_image to 0
#     new_image[start_row_:start_row_ + inner_region1.shape[0],
#     start_col_:start_col_ + inner_region1.shape[1]][overlap_region] = 0
#
#     # Paste inner_region2_resized in the center of new_image
#     new_image[start_row_:start_row_ + inner_region2_resized.shape[0],
#     start_col_:start_col_ + inner_region2_resized.shape[1]] += inner_region2_resized
#
#
#
#
#
#     mask2_resized=cv2.resize(mask2, (mask1.shape[1], mask1.shape[0]))
#
#     new_mask = np.zeros((max(image1.shape[0],image2.shape[0])*2, max(image1.shape[1],image2.shape[1])*2), dtype=np.uint8)
#
#
#     # Calculate the starting positions to paste the overlapping regions in the center of new_image
#     start_row = (new_mask.shape[0] - mask1.shape[0]) // 2
#     start_col = (new_mask.shape[1] - mask1.shape[1]) // 2
#
#     # Paste inner_region1 in the center of new_image
#     new_mask[start_row:start_row + mask1.shape[0],
#     start_col:start_col + mask1.shape[1]] = mask1
#
#     # Calculate the starting positions to paste inner_region2_resized in the center of new_image
#     start_row_ = int((new_mask.shape[0] - mask2.shape[0]) // 2 + (distance_factor * 50))
#     start_col_ = int((new_mask.shape[1] - mask2.shape[1]) // 2 + (distance_factor * 50))
#
#
#     # Calculate the region where two inner_regions overlap
#     overlap_region = np.logical_and(mask1 != 0, mask2_resized != 0)
#
#
#     #print(overlap_region.shape)
#     #print(overlap_region)
#     if np.any(overlap_region):
#         new_mask[start_row_:start_row_ + mask1.shape[0],start_col_:start_col_ + mask1.shape[1]][overlap_region] = 0
#
#     # Paste inner_region2_resized in the center of new_image
#     new_mask[start_row_:start_row_ + mask2_resized.shape[0],
#     start_col_:start_col_ + mask2_resized.shape[1]] += mask2_resized
#
#     # cv2.imshow('Image with Mask 1', overlap_region)
#     # cv2.imshow('Image with Mask 2', new_mask)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#
#     return new_image, inner_region1,inner_region2_resized,new_mask
#
#
#
#
#
# def cv2_to_pil(image):
#     """Convert OpenCV image (numpy array) to PIL image."""
#     return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# import matplotlib.pyplot as plt
#
# overlap_save_path = save_path+'/overlap'
# for i in range(100):
#     print(f"overlap {i} image save ")
#     combined_image, inner_region1,inner_region2,new_mask = generate_combined_image(valid_dataset,distance_factor=2)
#
#     save_data(cv2_to_pil(combined_image), cv2_to_pil(new_mask), overlap_save_path, i)
#
#






def select_random_image(dataset):
    index = random.randint(0, len(dataset.images) - 1)
    image = Image.open(dataset.images[index]).convert('RGB')
    mask = Image.open(dataset.masks[index]).convert('L')

    return image, mask

def zoom_object(image,mask,zoom_factor):
    original_width, original_height = image.size

    # 확대된 이미지의 크기를 계산합니다.
    new_width = int(original_width * zoom_factor)
    new_height = int(original_height * zoom_factor)

    # 이미지를 확대합니다.
    zoomed_image = image.resize((new_width, new_height))
    zoomed_mask = mask.resize((new_width, new_height))

    # 이미지 중심을 기준으로 크롭할 영역을 계산합니다.
    crop_left = (new_width - original_width) // 2
    crop_top = (new_height - original_height) // 2
    crop_right = crop_left + original_width
    crop_bottom = crop_top + original_height

    # 중심을 기준으로 이미지를 크롭합니다.
    zoomed_image = zoomed_image.crop((crop_left, crop_top, crop_right, crop_bottom))
    zoomed_mask = zoomed_mask.crop((crop_left, crop_top, crop_right, crop_bottom))

    return zoomed_image,zoomed_mask
#
#
#
#
# #
# zoom_save_path = save_path+'/zoom_out'
# for i in range(100):
#     print(f"zoom {i} image save ")
#     image, mask = select_random_image(valid_dataset)
#     zoomed_image, zoomed_mask = zoom_object(image, mask, zoom_factor=0.5)
#     save_data(zoomed_image, zoomed_mask, zoom_save_path,i)

zoom_save_path = save_path+'/zoom_in'
for i in range(100):
    print(f"zoom {i} image save ")
    image, mask = select_random_image(valid_dataset)
    zoomed_image, zoomed_mask = zoom_object(image, mask, zoom_factor=2)
    save_data(zoomed_image, zoomed_mask, zoom_save_path,i)


import os

# 특정 폴더 경로
folder_path = "/data/ADE/VOCdevkit/VOC2012/preprocessing_data/zoom_out/image"

# 마스크 폴더 경로
mask_folder_path = folder_path.replace("image", "mask")

# 상위 폴더 경로
parent_folder = "C:/Users/hail/Desktop/ha/model/ADE/VOCdevkit/VOC2012/"

# 폴더 내의 파일 목록 가져오기
file_list = os.listdir(folder_path)

# 파일 목록을 저장할 텍스트 파일 경로
output_file = "../zoom_out.txt"

# 파일 목록을 텍스트 파일에 저장
with open(output_file, 'w') as f:
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        mask_file_name = file_name.replace("image", "mask")
        mask_file_path = os.path.join(mask_folder_path, mask_file_name)
        # 파일 경로에서 상위 폴더 경로를 제거하여 저장
        file_path = file_path.replace(parent_folder, "/").replace("\\", "/")
        mask_file_path = mask_file_path.replace(parent_folder, "/").replace("\\", "/")
        f.write(file_path + " " + mask_file_path + '\n')

print("폴더 내의 파일 목록이 {}에 저장되었습니다.".format(output_file))





# import torch
# import torchvision
# import torchvision.transforms as transforms
# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# trainset = torchvision.datasets.CIFAR100(root='C:/Users/hail/Desktop/ha/model', train=True,
#                                         download=True, transform=transform)
# trainloader = torch.utils.model.DataLoader(trainset, batch_size=1,
#                                           shuffle=True, num_workers=2)
#
# testset = torchvision.datasets.CIFAR100(root='C:/Users/hail/Desktop/ha/model', train=False,
#                                        download=True, transform=transform)
#
# testloader = torch.utils.model.DataLoader(testset, batch_size=1,
#                                          shuffle=False, num_workers=2)
#
#
# def unpickle(file):
#     import pickle
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict
# metadata_path = 'C:/Users/hail/Desktop/ha/model/cifar-100-python/meta' # change this path`\
# metadata = unpickle(metadata_path)
# superclass_dict = dict(list(enumerate(metadata[b'coarse_label_names'])))
#
# print(trainset[0])