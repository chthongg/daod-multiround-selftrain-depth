from PIL import Image
import torch
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from matplotlib import colormaps
import glob
import tqdm
import os

def make_transform():
    to_tensor = v2.ToImage()
    resize = v2.Resize((1024, 2048), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])

REPO_DIR = "/home/bui/DAOD/DINO_Teacher/dinov3"
depther = torch.hub.load(REPO_DIR, 'dinov3_vit7b16_dd', source="local", 
            weights="/home/bui/DAOD/DINO_Teacher/weights/dinov3_vit7b16_synthmix_dpt_head-02040be1.pth", 
            backbone_weights="/home/bui/DAOD/DINO_Teacher/weights/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth")
depther = depther.cuda()

for split in ["train", "val"]:
    path_img = "/home/bui/DAOD/DINO_Teacher/datasets/cityscapes/leftImg8bit/" + split + "/"
    path_save = "/home/bui/DAOD/auxiliary_information/cityscapes/" + split + "/"
    os.makedirs(path_save, exist_ok=True)

    bsize = 1
    all_filenames = glob.glob(path_img + "/**/**.png")
    batch_all_pil_filenames = [all_filenames[i:i + bsize] for i in range(0, len(all_filenames), bsize)]

    all_pil_images = []
    for filename in tqdm.tqdm(all_filenames):
        image_pil = Image.open(filename).convert("RGB")
        all_pil_images.append(image_pil)

    batch_all_pil_images = [all_pil_images[i:i + bsize] for i in range(0, len(all_pil_images), bsize)]
    transform = make_transform()

    for filenames, images_pil in tqdm.tqdm(zip(batch_all_pil_filenames, batch_all_pil_images)):
        with torch.inference_mode():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                batch_img = transform(images_pil)
                depths = depther(torch.stack(batch_img).cuda())
                os.makedirs(path_save + filenames[0].split("/")[-2], exist_ok=True)
                path_save_img = path_save + filenames[0].split("/")[-2] + "/" + os.path.splitext(os.path.basename(filenames[0]))[0] + ".pth"
                torch.save(torch.cat([depths.squeeze(0)] * 3).cpu(), path_save_img)