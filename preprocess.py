import json
import torch
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop


data_path = "MovieNet/data/final_data.json"
dst_path = "MovieNet/data/embed"


class ImageDataset(Dataset):

    def __init__(self, img_paths, transform=None):

        self.transform = transform
        self.items = img_paths

    def __getitem__(self, index):

        img_path = self.items[index]
        
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, index

    def __len__(self):
        return len(self.items)


def _convert_to_rgb(image):
    return image.convert('RGB')

def _transform(n_px: int):
    normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    return Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        _convert_to_rgb,
        ToTensor(),
        normalize,
    ])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ### Building Model ###
    ckpt_path = "pretrained_models/clip/ViT-B-16.pt"

    model, _ =  clip.load(ckpt_path, jit=False)
    model = model.eval().cuda()
    print('Loaded pretrained model {} successfully!'.format(ckpt_path))


    ### Building Dataloader ###
    transform = _transform(224)
    segs = {}
    with open(data_path,'r') as f:
        total_data=json.load(f)
        for item in total_data:
            segs[item['id']]=item["image"]
    for seg,img_paths in tqdm(segs.items()):
        img_dataset = ImageDataset(img_paths, transform)
        img_dataloader = DataLoader(img_dataset,
                                    batch_size=args.batch_size, 
                                    num_workers=args.workers,
                                    pin_memory=True)

        img_feat = torch.zeros(len(img_paths), 512, device=device)
        with torch.no_grad():
            for j, (imgs, imgs_id) in enumerate(img_dataloader):
                imgs = imgs.to(device, non_blocking=True)
                img_feat[j*args.batch_size: min((j+1)*args.batch_size, len(img_paths))] = model.encode_image(image=imgs)

        torch.save(img_feat, f"{dst_path}/{seg}.pt")