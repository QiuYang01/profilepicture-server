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

color_dict = {
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "blue": (67, 142, 219)
}

# 加载模型 返回网络
def pre_net():
    model_name = 'u2net'
    path = os.path.dirname(__file__)
    # print(path)
    model_dir = path+'/u_2_net/model/' + model_name + '.pth'
    # print(model_dir)
    net = U2NET(3,1)
    # 指定使用cpu 考虑到服务器没有GPU
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

def to_standard_trimap(alpha):
    #  Alpha图生成 trimap
    # image = Image.open(alpha)
    image = alpha;
    # image = image.convert("P")
    # image_file.save('resize_trimap.png')
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
    image.save("trimap1111.jpg")
    return image

# 从原始图片获得Alpha图  org：原始图的路径 返回得到的Alpha图
def seg_trimap(org):
    image = Image.open(org)
    img = np.array(image)
    net = pre_net()
    inputs_test = pre_test_data(img)
    d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)
    # normalization
    pred = d1[:, 0, :, :]
    pred = normPRED(pred)
    # 将数据转换成图片
    im = get_im(pred)
    # im.save(alpha)
    sp = image.size
    # 根据原始图片调整尺寸
    imo = im.resize((sp[0], sp[1]), resample=Image.BILINEAR)
    imo.save("alpha_resize111.jpg")
    return imo
    return imo

def to_background(org, resize_trimap, id_image, color):
    """
        org：原始图片
        resize_trimap：trimap
        id_image：新图片
        color: 背景颜色
    """
    scale = 1.0
    image = load_image(org, "RGB", scale, "box")
    trimap = load_image(resize_trimap, "GRAY", scale, "nearest")
    im = Image.open(org)
    # estimate alpha from image and trimap
    try:
        alpha = estimate_alpha_cf(image, trimap)
    except:
        print("异常")
        print(org)
        a = "-1"
        return a
    new_background = Image.new('RGB', im.size, color_dict[color])
    new_background.save("bj.png")
    # load new background
    new_background = load_image("bj.png", "RGB", scale, "box")
    # estimate foreground from image and alpha
    foreground, background = estimate_foreground_ml(image, alpha, return_background=True)
    # blend foreground with background and alpha
    new_image = blend(foreground, new_background, alpha)
    save_image(id_image, new_image)
    return id_image

# path = "img\\"
# orgImgName = path + "qy.jpg"
# trimapImgName = path + "qy_trimapImg.png"
# resultImgName = path + "qy_result.png"
# cropResultImgName = path + "qy_crop_result.png"

# 原始文件名 中间过程文件名 结果文件名 颜色
def cropImage(orgImgName,trimapImgName,resultImgName,color):
    AlphaImg =  seg_trimap(orgImgName) # 从原始图片获得Alpha图
    trimapImg = to_standard_trimap(AlphaImg) #将Alpha图转成trimap图
    trimapImg.save(trimapImgName)
    return to_background(orgImgName, trimapImgName, resultImgName, color)

# cropImage(orgImgName,trimapImgName,resultImgName,"red")
# test_landmarks("..//img//meinv.jpg","..//img//meinv_id_landmarks.png")


