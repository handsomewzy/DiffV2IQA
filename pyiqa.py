import pyiqa

# list all available metrics
# pyiqa -ls

# test with default settings
# pyiqa ['musiq-koniq'] --target ['/home/vipsl416-10-wangzhaoyang/IQA_dataset/tid2013/diffusion_imgs/I01_01_1.bmp_0.png'] --ref [image_path or dir]

image_path= '/home/vipsl416-10-wangzhaoyang/IQA_dataset/tid2013/diffusion_imgs/I01_01_1.bmp_0.png'
for i in range(10):
    evaluator = pyiqa.create_metric('hyperiqa')
    value = evaluator(image_path)
    print(value)
    pass

python inference_iqa.py -m musiq -t /home/vipsl416-10-wangzhaoyang/IQA-pytorch/authentic_imgs

tres
maniqa-pipal
musiq
cnniqa
brisque