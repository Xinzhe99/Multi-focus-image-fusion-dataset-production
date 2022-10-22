import os
from PIL import Image, ImageFilter,ImageChops

def GaussianBlur(input):
    img = input.filter(ImageFilter.GaussianBlur(2))
    return img

def mask1(input):
    img = input.convert('RGB')
    for x in range(img.width):
        for y in range(img.height):
            data = img.getpixel((x, y))
            if data[0]+data[1]+data[2]!=0:
                img.putpixel((x, y), (255,255,255))
    return img

def mask2(input):
    img = input.convert('RGB')
    for x in range(img.width):
        for y in range(img.height):
            data = img.getpixel((x, y))
            if data[0]+data[1]+data[2] == 0:
                img.putpixel((x, y), (255,255,255))
            else:
                img.putpixel((x, y), (0, 0, 0))

    return img

def trans(input_path1,input_path2):
    result_root1 = os.path.join(os.getcwd(),'SourceA\\')  # 输出结果保存路径A
    result_root2 = os.path.join(os.getcwd(),'SourceB\\')  # 输出结果保存路径B
    result_root3 = os.path.join(os.getcwd(),'decisionmap\\')  # 输出结果保存路径B

    Original_img = Image.open(input_path1)
    Ground_img = Image.open(input_path2)
    Blurred_img = GaussianBlur(Original_img)
    Mask1_img = mask1(Ground_img)
    Mask2_img = mask2(Ground_img)
    Part_imageA1 = ImageChops.multiply(Blurred_img, Mask1_img)
    Part_imageB1 = ImageChops.multiply(Original_img, Mask1_img)
    Part_imageA2 = ImageChops.multiply(Original_img, Mask2_img)
    Part_imageB2 = ImageChops.multiply(Blurred_img, Mask2_img)
    Synthesized_imgA = ImageChops.add(Part_imageA1, Part_imageA2)
    Synthesized_imgB = ImageChops.add(Part_imageB1, Part_imageB2)
    save_rootA = result_root1 + input_path1[-15:-4]+'.jpg'
    print(save_rootA)
    save_rootB = result_root2 + input_path1[-15:-4] + '.jpg'
    save_rootC = result_root3 + input_path1[-15:-4] + '.png'
    Synthesized_imgA.save(save_rootA)
    Synthesized_imgB.save(save_rootB)
    Mask1_img.save(save_rootC)

def main():
    data_root=os.path.join(os.getcwd(),'VOC2012')#数据集位置
    Ground_list_name = [i for i in os.listdir(os.path.join(data_root,'SegmentationObject'))if i.endswith('png')]
    Ground_list=[os.path.join(data_root,'SegmentationObject',i)for i in Ground_list_name]
    Original_list=[os.path.join(data_root,'JPEGImages',i.split('.')[0]+'.jpg')for i in Ground_list_name]
    print(Ground_list)
    for i in range(len(Ground_list)):
        trans(Original_list[i],Ground_list[i])
        print('finish no.',i+1,'对图片')

if __name__ == "__main__":
    main()