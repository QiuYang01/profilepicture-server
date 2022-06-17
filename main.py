# from pymatting import cutout
# cutout("img/lemur.png","img/lemur.png","img/lemur_cutout.png")

import torch
from torch.autograd import Variable
from torchvision import transforms#, utils
import numpy as np
from u_2_net.data_loader import RescaleT
from u_2_net.data_loader import ToTensorLab
from u_2_net.model import U2NET # full size version 173.6 MB
import os
from PIL import Image
from pymatting import *

colour_dict = {
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "blue": (67, 142, 219)
}

# 加载模型
def pre_net():
    # 采用n2net 模型数据
    model_name = 'u2net'
    path = os.path.dirname(__file__)
    print(path)
    model_dir = path+'/u_2_net/model/' + model_name + '.pth'
    print(model_dir)
    # print("...load U2NET---173.6 MB")
    net = U2NET(3,1)
    # 指定cpu
    net.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))
    if torch.cuda.is_available():
        print("use GPU")
        net.cuda()
    net.eval()
    return net


def preprocess(image):
    label_3 = np.zeros(image.shape)
    label = np.zeros(label_3.shape[0:2])

    if (3 == len(label_3.shape)):
        label = label_3[:, :, 0]
    elif (2 == len(label_3.shape)):
        label = label_3
    if (3 == len(image.shape) and 2 == len(label.shape)):
        label = label[:, :, np.newaxis]
    elif (2 == len(image.shape) and 2 == len(label.shape)):
        image = image[:, :, np.newaxis]
        label = label[:, :, np.newaxis]

    transform = transforms.Compose([RescaleT(320), ToTensorLab(flag=0)])
    sample = transform({
        'imidx': np.array([0]),
        'image': image,
        'label': label
    })

    return sample


def pre_test_data(img):
    torch.cuda.empty_cache()
    sample = preprocess(img)
    inputs_test = sample['image'].unsqueeze(0)
    inputs_test = inputs_test.type(torch.FloatTensor)
    if torch.cuda.is_available():
        inputs_test = Variable(inputs_test.cuda())
    else:
        inputs_test = Variable(inputs_test)
    return inputs_test

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn

# 将数据转换成图片
def get_im(pred):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    im = Image.fromarray(predict_np*255).convert('RGB')
    return im

def to_standard_trimap(alpha, trimap):
    #  Alpha图生成 trimap
    print(alpha)
    image = Image.open(alpha)
    print(image)
    # image = image.convert("P")
    # image_file.save('meinv_resize_trimap.png')
    sp = image.size
    width = sp[0]
    height = sp[1]

    for yh in range(height):
        for xw in range(width):
            dot = (xw, yh)
            color_d_arr = image.getpixel(dot)
            color_d=color_d_arr[0]

            if 0 < color_d <= 60:
                image.putpixel(dot, (0,0,0))
            if 60 < color_d <= 200:
                image.putpixel(dot, (128,128,128))
            if 200 < color_d <= 255:
                image.putpixel(dot, (255,255,255))

    image.save(trimap)

def test_seg_trimap(org,alpha,alpha_resize):
    # 将原始图片转换成 Alpha图
    # org：原始图片
    # org_trimap:
    # resize_trimap: 调整尺寸的trimap
    image = Image.open(org)
    print(image)
    img = np.array(image)
    net = pre_net()
    inputs_test = pre_test_data(img)
    d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)
    # normalization
    pred = d1[:, 0, :, :]
    pred = normPRED(pred)
    # 将数据转换成图片
    im = get_im(pred)
    im.save(alpha)
    sp = image.size
    # 根据原始图片调整尺寸
    imo = im.resize((sp[0], sp[1]), resample=Image.BILINEAR)
    imo.save(alpha_resize)

def to_background(org, resize_trimap, id_image, colour):
    """
        org：原始图片
        resize_trimap：trimap
        id_image：新图片
        colour: 背景颜色
    """
    scale = 1.0
    image = load_image(org, "RGB", scale, "box")
    trimap = load_image(resize_trimap, "GRAY", scale, "nearest")
    im = Image.open(org)
    # estimate alpha from image and trimap
    alpha = estimate_alpha_cf(image, trimap)

    new_background = Image.new('RGB', im.size, colour_dict[colour])
    new_background.save("bj.png")
    # load new background
    new_background = load_image("bj.png", "RGB", scale, "box")

    # estimate foreground from image and alpha
    foreground, background = estimate_foreground_ml(image, alpha, return_background=True)

    # blend foreground with background and alpha
    new_image = blend(foreground, new_background, alpha)
    save_image(id_image, new_image)

#########################################################################
# test_seg_trimap("img\\meinv.jpg","img\\meinv_alpha.png","img\\meinv_alpha_resize.png") #meinv
org_img = "img\\lemur.png" #
alpha_img = "img\\meinv_alpha.png" #生成alpha
trimap = "img\\meinv_trimap_resize.png" #生成trimap
alpha_resize_img = "img\\meinv_alpha_resize.png" #生成
id_image = "img\\meinv_id.png"

# 通过u_2_net 获取图像的 alpha
# test_seg_trimap(org_img, alpha_img, alpha_resize_img)
#
# # 通过alpha 获取 trimap
# to_standard_trimap(alpha_resize_img, trimap)
#
# # 证件照添加蓝底纯色背景
# to_background(org_img, trimap, id_image, "red")
pre_net()

