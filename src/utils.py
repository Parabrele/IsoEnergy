from datetime import datetime
import os, requests, zipfile, hashlib, random
from io import BytesIO
from webdataset import TarWriter
import base64
from copy import deepcopy

from pathlib import Path
import json
import urllib.request

import aiohttp
import asyncio
import nest_asyncio

from functools import partial

import numpy as np
import torch
from tqdm import tqdm

import clip
from pycocotools.coco import COCO
import pandas as pd

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap, to_hex
from matplotlib.path import Path as mplPath
from matplotlib.patches import PathPatch
import colorsys
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from skimage.color import rgb2lab, lab2rgb

from timm.data.dataset import ImageDataset
from transformers import CLIPModel, CLIPProcessor, SiglipModel, SiglipProcessor, AutoProcessor, AutoModel, AutoImageProcessor, Dinov2Model
from typing import Callable
from dataclasses import dataclass, field

import overcomplete

import umap
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from src import losses, train, metrics

##########
# HuggingFace Setup
##########

from datasets import load_dataset
from huggingface_hub import login

# Read token
with open("hf_token.txt") as f:
    token = f.read().strip()

login(token=token)

##########
# Get Datasets
##########

clip_name = "openai/clip-vit-base-patch32"
clip_L_name = "openai/clip-vit-large-patch14"
openclip_name = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
openclip_L_name = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
siglip_name = "google/siglip-base-patch16-224"
siglip2_name = "google/siglip2-base-patch16-224"
dinov2_name = "facebook/dinov2-base"

def _get_model(model_name, device="cpu"):
        if model_name in [clip_name, clip_L_name, openclip_name, openclip_L_name]:
            model = CLIPModel.from_pretrained(model_name).to(device)
            processor = CLIPProcessor.from_pretrained(model_name)
        if model_name in [siglip_name, siglip2_name]:
            model = SiglipModel.from_pretrained(model_name).to(device)
            processor = SiglipProcessor.from_pretrained(model_name)
        if model_name in [dinov2_name]:
            model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
            processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        model.eval()
        try:
            max_length=model.config.max_position_embeddings
        except:
            max_length=None
        return model, processor, max_length

@torch.no_grad()
def _preprocess_img(img, model, processor, max_length):
    # if model is dinov2, custom code :
    if isinstance(model, Dinov2Model):
        image_tensor = processor(images=img, return_tensors="pt")
        img_feat = model(**image_tensor.to(model.device))[0][:, 0, :].cpu() # CLS token
        return img_feat.squeeze()
    image_tensor = processor(images=img, return_tensors="pt")['pixel_values'].to(model.device)
    img_feat = model.get_image_features(pixel_values=image_tensor)
    img_feat /= img_feat.norm(dim=-1, keepdim=True)
    return img_feat.squeeze()

@torch.no_grad()
def _preprocess_txt(txt, model, processor, max_length):
    image_tensor = processor(text=txt, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length).to(model.device)
    img_feat = model.get_text_features(**image_tensor)
    img_feat /= img_feat.norm(dim=-1, keepdim=True)
    return img_feat.squeeze()

def get_coco(batch_size=512, device="cpu", model_name="openai/clip-vit-base-patch32"):
    DEEL_coco_embeddings_path = "./coco/embeddings/"
    BROWN_coco_embeddings_path = "/media/data_cifs/projects/prj_multimodal/coco/embeddings/"
    base_path = BROWN_coco_embeddings_path + model_name
    img_path = os.path.join(base_path, "image_features.pt")
    txt_path = os.path.join(base_path, "text_features.pt")
    
    DEEL_annFile = '/datasets/shared_datasets/coco/annotations/captions_train2017.json'
    BROWN_annFile = '/media/data_cifs/projects/prj_multimodal/coco/annotations/captions_train2017.json'
    annFile = BROWN_annFile
    
    DEEL_image_dir = '/datasets/shared_datasets/coco/train2017'
    BROWN_image_dir = '/media/data_cifs/projects/prj_multimodal/coco/train2017'
    image_dir = BROWN_image_dir
    
    coco = COCO(annFile)

    # Prepare image-caption pairs
    data = []
    for img_id in coco.getImgIds():
        img_info = coco.loadImgs(img_id)[0]
        file_path = os.path.join(image_dir, img_info['file_name'])
        captions = [ann['caption'] for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id))]
        for caption in captions:
            data.append((file_path, caption))
        
    try:
        image_features = torch.load(img_path, map_location="cpu").to(device)
        text_features = torch.load(txt_path, map_location="cpu").to(device)    
    except FileNotFoundError:
        model, processor, max_length = _get_model(model_name, device=device)
    
        class CocoCaptionDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                img_path, caption = self.data[idx]
                I = Image.open(img_path)
                if I.mode != 'RGB':
                    I = I.convert('RGB')
                img_feat = _preprocess_img(I, model, processor, max_length)
                txt_feat = _preprocess_txt([caption], model, processor, max_length)
                return img_feat, txt_feat

        coco_dataset = CocoCaptionDataset(data)

        temp_loader = torch.utils.data.DataLoader(coco_dataset, batch_size=1, shuffle=False)
        image_features, text_features = [], []
        i = 0
        for img_feat, txt_feat in tqdm(temp_loader):
            image_features.append(img_feat.squeeze(0))
            text_features.append(txt_feat.squeeze(0))
            i += 1
        image_features = torch.stack(image_features).to(device)
        text_features = torch.stack(text_features).to(device)
        print(f"Image features shape: {image_features.shape}, memory: {image_features.element_size() * image_features.nelement() / 1024**2:.2f} MB")
        print(f"Text features shape: {text_features.shape}, memory: {text_features.element_size() * text_features.nelement() / 1024**2:.2f} MB")
        os.makedirs(base_path, exist_ok=True)
        torch.save(image_features.cpu(), img_path)
        torch.save(text_features.cpu(), txt_path)

    CENTER_DATASET = False
    if CENTER_DATASET:
        image_features = image_features - image_features.mean(dim=0, keepdim=True)
        text_features = text_features - text_features.mean(dim=0, keepdim=True)
    
    train_dataset = torch.utils.data.TensorDataset(image_features, text_features)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Display a few images with captions
    def plot_images_with_captions(data, num_images=5):

        # Select random images
        for i in range(num_images):
            img_path, caption = random.choice(data)
            img = Image.open(img_path)
            plt.imshow(img)
            plt.axis('off')
            plt.show()
            print(f"Caption: {caption}")

        plt.tight_layout()
        plt.show()

    # Plot random images with captions
    plot_images_with_captions(data)

    return train_loader

def _download_laion(data_dir="/datasets/shared_datasets/LAION400M/"):
    # Load dataset with token
    max_items=100_000
    ds_streamed = load_dataset("laion/laion400m", split="train", streaming=True)

    caption_dict = {}
    
    async def download_image(session, url, pbar, sem, caption, idx):
        async with sem:
            try:
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        img = Image.open(BytesIO(await resp.read()))
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        if 1 in img.size:
                            raise ValueError("Image size is too small.")

                        os.makedirs(os.path.join(data_dir, "images/"), exist_ok=True)
                        save_path = os.path.join(data_dir, "images/", f"{idx}.png")
                        img.save(save_path, format="PNG")
                        
                        caption_dict[idx] = caption
                        
            except Exception:
                pass
            finally:
                pbar.update(1)

    async def main(dataset, max_items=max_items, concurrency=40):
        sem = asyncio.Semaphore(concurrency)
        pbar = tqdm(total=max_items, desc="Processing images")
        async with aiohttp.ClientSession() as session:
            tasks = []
            for i, item in enumerate(dataset):
                if i >= max_items:
                    break
                url = item.get("URL") or item.get("url")
                caption = item.get("caption")
                if url:
                    tasks.append(download_image(session, url, pbar, sem, caption, i))
            await asyncio.gather(*tasks)
        pbar.close()

    nest_asyncio.apply()
    async def runner():
        await main(ds_streamed, max_items=max_items)
    
    asyncio.run(runner())
    
    # Save captions to a JSON file
    os.makedirs(os.path.join(data_dir, "captions/"), exist_ok=True)
    captions_path = os.path.join(data_dir, "captions/captions.json")
    with open(captions_path, 'w') as f:
        json.dump(caption_dict, f, indent=4)

    print(f"Downloaded {len(caption_dict)} images out of {max_items} requested.")
    
DEEL_laion_path = "/datasets/shared_datasets/LAION400M/"
BROWN_laion_path = "/media/data_cifs/projects/prj_multimodal/LAION400M/"

@torch.no_grad()
def get_laion(batch_size=512, device="cpu", model_name="openai/clip-vit-base-patch32", data_dir=BROWN_laion_path, vision_only=False, max_items=1_000_000):
    # Load dataset with token
    ds_streamed = load_dataset("laion/laion400m", split="train", streaming=True)
    
    base_path = data_dir + "embeddings/" + model_name
    img_path = os.path.join(base_path, "image_features.pt")
    txt_path = os.path.join(base_path, "text_features.pt")
    
    try:
        image_features = torch.load(img_path, map_location="cpu").to(device)
        if not vision_only:
            text_features = torch.load(txt_path, map_location="cpu").to(device)
    except FileNotFoundError:
        model, processor, max_length = _get_model(model_name, device=device)
        
        image_features = [None] * max_items
        text_features = [None] * max_items
        
        queue = asyncio.Queue(maxsize=4*batch_size)

        async def producer(session, dataset, sem, pbar, max_items):
            tasks = []

            async def fetch_one(i, url, caption):
                async with sem:
                    try:
                        async with session.get(url, timeout=10) as resp:
                            if resp.status == 200:
                                img = Image.open(BytesIO(await resp.read()))
                                if img.mode != 'RGB':
                                    img = img.convert('RGB')
                                if 1 in img.size:
                                    raise ValueError("Image too small")
                                await queue.put([img, caption, i])
                    except Exception:
                        pass
                    finally:
                        pbar.update(1)

            for i, item in enumerate(dataset):
                if i >= max_items:
                    break
                url = item.get("URL") or item.get("url")
                caption = item.get("caption")
                if not url:
                    pbar.update(1)
                    continue
                tasks.append(asyncio.create_task(fetch_one(i, url, caption)))

            # wait for all downloads to finish
            await asyncio.gather(*tasks)
            await queue.put(None)

        async def consumer(model, processor, max_length, image_features, text_features, batch_size=512):
            batch = []
            while True:
                item = await queue.get()
                if item is None:  # producer finished
                    if batch:
                        await asyncio.to_thread(
                            process_batch, batch, model, processor, max_length, image_features, text_features
                        )
                    break
                batch.append(item)
                if len(batch) >= batch_size:
                    await asyncio.to_thread(
                        process_batch, batch, model, processor, max_length, image_features, text_features
                    )
                    batch.clear()
                queue.task_done()

        def process_batch(batch, model, processor, max_length, image_features, text_features):
            imgs, captions, indices = zip(*batch)
            imgs, captions, indices = list(imgs), list(captions), list(indices)
            try:
                img_feats = _preprocess_img(imgs, model, processor, max_length)
                if not vision_only:
                    txt_feats = _preprocess_txt(list(captions), model, processor, max_length)
                    for idx, img_f, txt_f in zip(indices, img_feats, txt_feats):
                        image_features[idx] = img_f
                        text_features[idx] = txt_f
                else:
                    for idx, img_f in zip(indices, img_feats):
                        image_features[idx] = img_f
            except Exception:
                print("Warning: code 663431 ! I repeat: code 663431 !")
            
        async def main(dataset, model, processor, max_length, max_items, concurrency):
            sem = asyncio.Semaphore(concurrency)
            pbar = tqdm(total=max_items, desc="Downloading images")
            async with aiohttp.ClientSession() as session:
                producer_task = asyncio.create_task(producer(session, dataset, sem, pbar, max_items))
                consumer_task = asyncio.create_task(consumer(model, processor, max_length, image_features, text_features))
                await asyncio.gather(producer_task, consumer_task)
            pbar.close()
        
        nest_asyncio.apply()

        async def runner():
            await main(
                ds_streamed,
                model=model,
                processor=processor,
                max_length=max_length,
                max_items=max_items,
                concurrency=512,  # or whatever you want
            )

        asyncio.run(runner())
        
        """
        async def download_image(session, url, pbar, sem, caption, idx):
            async with sem:
                try:
                    async with session.get(url, timeout=10) as resp:
                        if resp.status == 200:
                            # with open(path, "wb") as f:
                            #     f.write(await resp.read())
                            img = Image.open(BytesIO(await resp.read()))
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            if 1 in img.size:
                                raise ValueError("Image size is too small.")
                            txt = [caption]
                            img_feat = _preprocess_img(img, model, processor, max_length)
                            txt_feat = _preprocess_txt(txt, model, processor, max_length)
                            image_features[idx] = img_feat
                            text_features[idx] = txt_feat
                except Exception:
                    pass
                finally:
                    pbar.update(1)

        async def main(dataset, max_items=max_items, concurrency=40):
            sem = asyncio.Semaphore(concurrency)
            pbar = tqdm(total=max_items, desc="Processing images")
            async with aiohttp.ClientSession() as session:
                tasks = []
                for i, item in enumerate(dataset):
                    if i >= max_items:
                        break
                    url = item.get("URL") or item.get("url")
                    caption = item.get("caption")
                    if url:
                        tasks.append(download_image(session, url, pbar, sem, caption, i))
                await asyncio.gather(*tasks)
            pbar.close()
        
        async def runner():
            await main(ds_streamed, max_items=max_items)
     
        nest_asyncio.apply()
        asyncio.run(runner())
        """
        
        not_none_indices = [i for i, img in enumerate(image_features) if img is not None]
        
        image_features = [image_features[i] for i in not_none_indices]
        image_features = torch.stack(image_features).to(device)
        
        if not vision_only:
            text_features = [text_features[i] for i in not_none_indices]
            text_features = torch.stack(text_features).to(device)
        
        os.makedirs(base_path, exist_ok=True)
        torch.save(image_features.cpu(), img_path)
        if not vision_only:
            torch.save(text_features.cpu(), txt_path)
    
    dataset = torch.utils.data.TensorDataset(image_features, text_features) if not vision_only else torch.utils.data.TensorDataset(image_features)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    """
    max_items=100
    ds_streamed = load_dataset("laion/laion400m", split="train", streaming=True)
    data = []
    for i, item in enumerate(ds_streamed):
        if i >= max_items:
            break
        url = item.get("URL") or item.get("url")
        caption = item.get("caption")
        if url and caption:
            # download the image
            try:
                with requests.get(url, stream=True, timeout=10) as r:
                    if r.status_code == 200:
                        img = Image.open(BytesIO(r.content))
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        if 1 in img.size:
                            continue
                        data.append((img, caption))
            except requests.exceptions.RequestException:
                continue

    # Display a few images with captions
    def plot_images_with_captions(data, num_images=5):

        # Select random images
        for i in range(num_images):
            img, caption = random.choice(data)
            plt.imshow(img)
            plt.axis('off')
            plt.show()
            print(f"Caption: {caption}")

        plt.tight_layout()
        plt.show()
        
    # Plot random images with captions
    plot_images_with_captions(data)
    """

    return train_loader

def get_laion_raw(target_nb_samples=int(10_000_000*4/3), samples_per_shard=10_000, data_dir=BROWN_laion_path, concurrency=512):
    # Download and save LAION images.
    # - Save metadata to keep track of (id, shard, cap, url), where
    #     - id : unique identifier for each image-caption pair
    #     - shard : shard index where the actual image is stored
    #     - cap : caption text
    #     - url : image URL
    # - Save sharded images to avoid overloading filesystem
    ds_streamed = load_dataset("laion/laion400m", split="train", streaming=True)
    
    queue = asyncio.Queue(maxsize=4096) # store a maximum of 4096 in the producer queue. If more than that are produced, the producer waits for the consumer to catch up.
    metadata = []
    
    async def producer(session, dataset, sem, pbar, max_items):
        tasks = []

        async def fetch_one(i, url, caption):
            async with sem:
                try:
                    async with session.get(url, timeout=10) as resp:
                        if resp.status != 200:
                            raise ValueError("Non-200 response")
                        data = await resp.read()
                        img = Image.open(BytesIO(data))
                        img.load()
                        
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        w, h = img.size
                        if w < 2 or h < 2:
                            raise ValueError("Image too small")
                        
                        await queue.put([img, caption, i, url])
                except Exception:
                    pass
                finally:
                    pbar.update(1)

        for i, item in enumerate(dataset):
            if i >= max_items:
                break
            url = item.get("URL") or item.get("url")
            caption = item.get("caption")
            if not isinstance(caption, str) or len(caption.strip()) == 0:
                pbar.update(1)
                continue
            if not url:
                pbar.update(1)
                continue
            tasks.append(asyncio.create_task(fetch_one(i, url, caption)))

        # wait for all downloads to finish
        await asyncio.gather(*tasks)
        await queue.put(None)
        
    async def consumer():
        batch = []
        shard_idx = 0
        shard_dir = Path(data_dir) / "images_sharded/"
        shard_dir.mkdir(parents=True, exist_ok=True)
        
        def write_shard(shard_idx, batch):
            shard_path = shard_dir / f"{shard_idx:06d}.tar"
            with TarWriter(str(shard_path)) as tw:
                for img, caption, i, url in batch:
                    sample_key = f"{i:09d}"
                    buf = BytesIO()
                    img.save(buf, format="JPEG", quality=95)
                    tw.write({
                        "__key__": sample_key,
                        "jpg": buf.getvalue(),
                    })
        
        while True:
            item = await queue.get()
            if item is None:  # producer finished
                if batch:
                    # shard the remaining images
                    write_shard(shard_idx, batch)
                break
            batch.append(item)
            metadata.append((item[2], shard_idx, item[1], item[3]))  # (id, shard, cap, url)
            if len(batch) >= samples_per_shard:
                # shard the images in the batch
                print(f"Writing shard {shard_idx} with {len(batch)} samples. Id of last sample: {batch[-1][2]}")
                write_shard(shard_idx, batch)
                shard_idx += 1
                batch.clear()
            queue.task_done()
    
    async def main(dataset, concurrency):
        sem = asyncio.Semaphore(concurrency)
        pbar = tqdm(total=target_nb_samples, desc="Downloading images")
        async with aiohttp.ClientSession() as session:
            producer_task = asyncio.create_task(producer(session, dataset, sem, pbar, target_nb_samples))
            consumer_task = asyncio.create_task(consumer())
            await asyncio.gather(producer_task, consumer_task)
        pbar.close()
    
    nest_asyncio.apply()

    async def runner():
        await main(
            ds_streamed,
            concurrency=concurrency,
        )

    asyncio.run(runner())
    
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    meta_path = Path(data_dir) / "metadata"
    meta_path.mkdir(exist_ok=True, parents=True)

    df = pd.DataFrame(metadata, columns=["id", "shard", "caption", "url"])
    table = pa.Table.from_pandas(df)
    pq.write_table(table, meta_path / "laion_metadata.parquet")

def _load_laion_dataset(data_dir):
    img_emb_dir = os.path.join(data_dir, "img_emb")
    text_emb_dir = os.path.join(data_dir, "text_emb")
    meta_dir = os.path.join(data_dir, "metadata")

    all_img_emb = []
    all_text_emb = []
    all_metadata = []

    for fname in sorted(os.listdir(img_emb_dir)):
        n = fname.split("_")[-1].split(".")[0]
        img_emb = np.load(os.path.join(img_emb_dir, f"img_emb_{n}.npy"))
        text_emb = np.load(os.path.join(text_emb_dir, f"text_emb_{n}.npy"))
        # metadata = pd.read_parquet(os.path.join(meta_dir, f"metadata_{n}.parquet"))

        all_img_emb.append(img_emb)
        all_text_emb.append(text_emb)
        # all_metadata.append(metadata[["url", "caption", "similarity"]])

    # Concatenate all
    img_emb_all = np.concatenate(all_img_emb, axis=0).astype(np.float32)
    text_emb_all = np.concatenate(all_text_emb, axis=0).astype(np.float32)
    # meta_all = pd.concat(all_metadata, ignore_index=True)

    img_emb_all = torch.from_numpy(img_emb_all)
    text_emb_all = torch.from_numpy(text_emb_all)

    dataset = torch.utils.data.TensorDataset(img_emb_all, text_emb_all)

    return dataset

def _compute_laion_embeddings(model_name, data_dir, device="cpu"):
    # get metadata :
    metadata = pd.read_parquet(data_dir + "/CLIP/metadata/metadata_0.parquet")

def __old_get_laion(batch_size=512, device="cpu", model_name="CLIP"):
    assert model_name in ["CLIP", "SigLIP", "SigLIP2"], f"Unsupported model name: {model_name}. Supported: ['CLIP', 'SigLIP', 'SigLIP2']"
    laion_dir = "/datasets/shared_datasets/LAION400M/embeddings" + f"/{model_name}"
    train_dataset = _load_laion_dataset(laion_dir)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )
    return train_loader

class DataGenerator:
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        self.shapes = ['circle', 'square', 'triangle', 'rectangle']
        self.shape_colors = ['blue', 'darkblue', 'lightblue', 'purple', 'indigo', 'green', 'gold']
        self.sizes = ['small', 'medium', 'large']
        self.backgrounds = [
            'lightgrey', 'grey', 'dimgray', 'black', 'white', 'red', 'darkred', 
        ]
    
        self.data = []
        if num_samples > 0:
            for _ in range(self.num_samples):
                self.data.append(self.generate_sample())
        elif num_samples == -1:
            self.generate_all()

    def _add_shape_to_image(self, image, shape, color, size):
        draw = ImageDraw.Draw(image)
        s = image.size[0]
        
        center_x = s // 2
        center_y = s // 2

        size_map = {'small': s // 10, 'medium': s // 7, 'large': s // 5}
        r = size_map[size]

        bbox = [center_x - r, center_y - r, center_x + r, center_y + r]

        match shape:
            case 'circle':
                draw.ellipse(bbox, fill=color)
            case 'square':
                draw.rectangle(bbox, fill=color)
            case 'rectangle':
                draw.rectangle([center_x - r, center_y - r//2, center_x + r, center_y + r//2], fill=color)
            case 'triangle':
                points = [
                    (center_x, center_y - r),
                    (center_x - r, center_y + r),
                    (center_x + r, center_y + r)
                ]
                draw.polygon(points, fill=color)

    def generate_sample(self):
        # Select random attributes and generate the corresponding text and image
        background = random.choice(self.backgrounds)

        n_shapes = 1 # random.randint(1, 3)
        shape_list = []
        for _ in range(n_shapes):
            shape = random.choice(self.shapes)
            color = random.choice(self.shape_colors)
            size = random.choice(self.sizes)
            shape_list.append((shape, color, size))
        
        # Generate the text description
        background_str = f" on a {background} background"
        link_between_shapes_str = f" along with "

        for i, (shape, color, size) in enumerate(shape_list):
            if i == 0:
                text = f"a {size} {color} {shape}"
            else:
                raise ValueError("Should not have more than one shape in a sample in the current implementation.")
                text += link_between_shapes_str + f"a {size} {color} {shape} at the {self._pos2str(position)} position"
            
        text += background_str

        # Generate the image
        image = Image.new('RGB', (224, 224), background)
        for shape, color, size in shape_list:
            self._add_shape_to_image(image, shape, color, size)

        # Raw contains the original information, before they are converted to text and image
        raw = {
            'background': background,
            'shapes': [{'shape': shape, 'color': color, 'size': size} for shape, color, size in shape_list]
        }
        
        return text, image, raw
    
    def generate_all(self):
        a = len(self.shapes)
        b = len(self.shape_colors)
        c = len(self.sizes)
        d = len(self.backgrounds)
        for l, background in enumerate(self.backgrounds):
            for i, shape in enumerate(self.shapes):
                for j, color in enumerate(self.shape_colors):
                    for k, size in enumerate(self.sizes):
                        text = f"A {size} {color} {shape} on a {background} background."
                        image = Image.new('RGB', (224, 224), background)
                        self._add_shape_to_image(image, shape, color, size)
                        raw = {
                            'background': background,
                            'shapes': [{'shape': shape, 'color': color, 'size': size}]
                        }
                        label = torch.zeros(a + b + c + d)
                        label[i] = 1
                        label[a + j] = 1
                        label[a + b + k] = 1
                        label[a + b + c + l] = 1
                        self.data.append((text, image, label, raw))

def get_toy(batch_size=512, num_samples=10000, device="cpu"):
    try:
        image_features = torch.load(os.path.join("./toy_cache", "image_features.pt"), map_location='cpu').squeeze(1)
        text_features = torch.load(os.path.join("./toy_cache", "text_features.pt"), map_location='cpu').squeeze(1)
    except FileNotFoundError:
        model, preprocess = clip.load("ViT-B/32", device=device)
        model = model.to(dtype=torch.float32)

        data_gen = DataGenerator(num_samples=num_samples)
        
        @torch.no_grad()
        def extract_activations(model, image_tensor, text_tokens):
            image_features = model.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features = model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            return image_features, text_features
        
        class ToyDataset(torch.utils.data.Dataset):
            def __init__(self, data, preprocess, tokenizer):
                self.data = data
                self.preprocess = preprocess
                self.tokenizer = tokenizer
                self.cache_dir = "./toy_cache"
                os.makedirs(self.cache_dir, exist_ok=True)

            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                text, image, _ = self.data[idx]
                # Preprocess image
                image_tensor = self.preprocess(image).unsqueeze(0).to(device)
                # Tokenize text
                text_tokens = self.tokenizer([text])[0].to(device).unsqueeze(0)

                image_features, text_features = extract_activations(model, image_tensor, text_tokens)

                return image_features.squeeze(0), text_features.squeeze(0)
            
        tokenizer = clip.tokenize
        toy_dataset = ToyDataset(data_gen.data, preprocess, tokenizer)
        raws = []
        for _, _, raw in tqdm(data_gen.data, desc="Extracting raw data"):
            raws.append(raw)
        torch.save(raws, os.path.join(toy_dataset.cache_dir, "raws.pt"))
        temp_loader = torch.utils.data.DataLoader(toy_dataset, batch_size=1, shuffle=True)
        image_features, text_features = [], []
        i = 0
        for img_feats, txt_feats in tqdm(temp_loader, desc="Extracting features"):
            image_features.append(img_feats)
            text_features.append(txt_feats)
            i += 1
        
        image_features = torch.stack(image_features).to('cpu').squeeze(1)
        text_features = torch.stack(text_features).to('cpu').squeeze(1)
        torch.save(image_features, os.path.join(toy_dataset.cache_dir, "image_features.pt"))
        torch.save(text_features, os.path.join(toy_dataset.cache_dir, "text_features.pt"))

    train_dataset = torch.utils.data.TensorDataset(image_features, text_features)
    print(train_dataset.tensors[0].shape, train_dataset.tensors[1].shape)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader

imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

def get_imagenet(batch_size=512, model_name=clip_name, split="test", device="cpu", shuffle=False, vision_only=False):
    DEEL_imagenet_embeddings_path = "./imagenet/embeddings/"
    BROWN_imagenet_embeddings_path = "/media/data_cifs/projects/multimodal/imagenet/embeddings/"
    base_path = BROWN_imagenet_embeddings_path + model_name
    train_path = 'train.pt'
    test_path = 'test.pt'
    if split == "train":
        file_path = os.path.join(base_path, train_path)
    elif split == "test" or split == "val":
        file_path = os.path.join(base_path, test_path)
    
    if not vision_only:
        classes_path = os.path.join(base_path, "classes.pt")
    
    class ImageNetDataset(ImageDataset):
        def __init__(self, path: str, transforms: Callable, **kwargs):
            print(f"Loading ImageNet dataset from {path}...")
            super(ImageNetDataset, self).__init__(path, **kwargs)
            self.transforms = transforms

        def __getitem__(self, item):
            img, label = super(ImageNetDataset, self).__getitem__(item)
            # img = self.transforms(img)

            return img, label
    
    found_img_file = False
    found_classes_file = False
    try:
        image_features, labels = torch.load(file_path, map_location='cpu')
        found_img_file = True
        if not vision_only:
            classes_features = torch.load(classes_path, map_location='cpu').to(device)
        found_classes_file = True
    except FileNotFoundError:
        print(base_path)
        os.makedirs(base_path, exist_ok=True)

        model, processor, max_length = _get_model(model_name, device=device)

        if not found_img_file:
            DEEL_imagenet_data_path = '/datasets/shared_datasets/imagenet/ILSVRC/Data/CLS-LOC/' + ('train' if split == "train" else "val")
            BROWN_imagenet_data_path = '/media/data_cifs/projects/prj_video_imagenet/imagenet/ILSVRC/Data/CLS-LOC/' + ('train' if split == "train" else "val2")
            _dataset = ImageNetDataset(
                BROWN_imagenet_data_path,
                partial(_preprocess_img, model=model, processor=processor, max_length=max_length),
            )
            loader = torch.utils.data.DataLoader(
                _dataset,
                batch_size=None,          # single samples
                shuffle=False,
                num_workers=os.cpu_count() // 2,  # parallel I/O
                pin_memory=True
            )

            queue = asyncio.Queue(maxsize=32)
            
            async def producer(pbar):
                for i, (img, label) in enumerate(loader):
                    await queue.put((i, img, label))
                    pbar.update(1)
                await queue.put((None, None, None))
            
            async def consumer():
                my_big_tensor, all_labels = [], []
                batch_imgs, batch_labels = [], []
                preprocess_fct = partial(_preprocess_img, model=model, processor=processor, max_length=max_length)
                while True:
                    i, img, label = await queue.get()
                    if i is None:
                        if batch_imgs:
                            imgs_tensor = preprocess_fct(batch_imgs)
                            my_big_tensor.append(imgs_tensor)
                            all_labels.append(torch.tensor(batch_labels))
                        break
                    batch_imgs.append(img)
                    batch_labels.append(label)
                    if len(batch_imgs) >= batch_size:
                        imgs_tensor = preprocess_fct(batch_imgs)
                        my_big_tensor.append(imgs_tensor)
                        all_labels.append(torch.tensor(batch_labels))
                        batch_imgs, batch_labels = [], []
                    queue.task_done()
                image_features = torch.cat(my_big_tensor, dim=0)
                labels = torch.cat(all_labels, dim=0)
                return image_features, labels
            
            with torch.no_grad():
                nest_asyncio.apply()
                async def runner():
                    pbar = tqdm(total=len(_dataset), desc="Processing ImageNet images")
                    prod_task = asyncio.create_task(producer(pbar))
                    cons_task = asyncio.create_task(consumer())
                    await asyncio.gather(prod_task, cons_task)
                    pbar.close()
                    return cons_task.result()
                image_features, labels = asyncio.run(runner())
                print(image_features.shape)
                print("File path :", file_path)
                torch.save((image_features, labels), file_path)
        
        if (not found_classes_file) and (not vision_only):

            def get_class_to_idx(path):
                subdirs = sorted([p.name for p in Path(path).iterdir() if p.is_dir()])
                return {name: idx for idx, name in enumerate(subdirs)}

            DEEL_class_to_idx_path = '/datasets/shared_datasets/imagenet/ILSVRC/Data/CLS-LOC/val'
            BROWN_class_to_idx_path = '/media/data_cifs/projects/prj_video_imagenet/imagenet/ILSVRC/Data/CLS-LOC/val2'
            class_to_idx = get_class_to_idx(BROWN_class_to_idx_path)
            idx_to_class = {v: k for k, v in class_to_idx.items()}
            
            url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
            with urllib.request.urlopen(url) as f:
                class_index = json.load(f)

            wnid_to_label = {v[0]: v[1] for v in class_index.values()}
            idx_to_label = {k: wnid_to_label[v] for k, v in idx_to_class.items()}
            
            class_features = []
            for k, v in idx_to_label.items():
                formatted = v.replace("_", " ").strip().lower()
                this_class_prompts = []
                for template in imagenet_templates:
                    prompt = template.format(formatted)
                    this_class_prompts.append(prompt)
                this_class_features = _preprocess_txt(this_class_prompts, model, processor, max_length)
                # mean & normalize :
                class_ensemble = this_class_features.mean(dim=0)
                class_ensemble /= class_ensemble.norm()
                class_features.append(class_ensemble)
            classes_features = torch.stack(class_features, dim=0)
            print(classes_features.shape)
            torch.save(classes_features, classes_path)

    train_dataset = torch.utils.data.TensorDataset(image_features.to(device), labels.to(device))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    
    """
    # data = []
    # imagenetdataset = ImageNetDataset('/datasets/shared_datasets/imagenet/ILSVRC/Data/CLS-LOC/' + ('train' if split == "train" else "val"),
    #                                   lambda x: x,)
    # for i in range(100):
    #     idx = random.randint(0, len(imagenetdataset) - 1)
    #     img, label = imagenetdataset[idx]
    #     data.append((img, label))
    
    # def get_class_to_idx(path):
    #     subdirs = sorted([p.name for p in Path(path).iterdir() if p.is_dir()])
    #     return {name: idx for idx, name in enumerate(subdirs)}

    # class_to_idx = get_class_to_idx('/datasets/shared_datasets/imagenet/ILSVRC/Data/CLS-LOC/val')
    # idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    # with urllib.request.urlopen(url) as f:
    #     class_index = json.load(f)

    # wnid_to_label = {v[0]: v[1] for v in class_index.values()}
    # idx_to_label = {k: wnid_to_label[v] for k, v in idx_to_class.items()}
            
    # # Display a few images with labels
    # def plot_images_with_labels(data, num_images=5):
    #     for i in range(num_images):
    #         img, label = random.choice(data)
    #         plt.imshow(img)
    #         plt.axis('off')
    #         print(f"Label: {label} ({idx_to_label[label]})")
    #         plt.show()
            
    # # Plot random images with labels
    # plot_images_with_labels(data, num_images=5)
    """
    
    if vision_only:
        return train_loader
    return train_loader, classes_features

def get_fashionIQ(batch_size=512, model_name=clip_name, device="cpu", split="train"):
    DEEL_fashionIQ_path = "/datasets/shared_datasets/fashion_iq/"
    BROWN_fashionIQ_path = "/media/data_cifs/projects/prj_multimodal/fashion_iq/"
    base_path = BROWN_fashionIQ_path + "embeddings/" + model_name
    cdd_path = os.path.join(base_path, "candidate_features.pt")
    trg_path = os.path.join(base_path, "target_features.pt")
    txt_path = os.path.join(base_path, "text_features.pt")
    
    try:
        candidate_features = torch.load(cdd_path, map_location="cpu").to(device)
        target_features = torch.load(trg_path, map_location="cpu").to(device)
        text_features = torch.load(txt_path, map_location="cpu").to(device)
    except FileNotFoundError:
        # prepare model
        model, processor, max_length = _get_model(model_name, device=device)

        image_path = BROWN_fashionIQ_path + "images/"
        captions_path = BROWN_fashionIQ_path + "captions/" # cap.{itemtype}.{split}.json
        
        # get metadata, a list of dicts with keys: target->id, candidate->id, captions->list of captions
        meta = []
        for itemtype in ["dress", "shirt", "toptee"]:
            with open(os.path.join(captions_path, f"cap.{itemtype}.{split}.json"), 'r') as f:
                meta.extend(json.load(f))
        
        #####
        # Images :
        #####
        
        # extract the list of unique ids used in this split.
        unique_trg = list(set([cap['target'] for cap in meta]))
        unique_cdd = list(set([cap['candidate'] for cap in meta]))
        unique_ids = list(set(unique_trg + unique_cdd))

        # find out their urls
        id2url = {}
        for itemtype in ["dress", "shirt", "toptee"]:
            with open(
                os.path.join(
                    BROWN_fashionIQ_path, "image_url", f"asin2url.{itemtype}.txt"
                ), 'r'
            ) as f:
                # each line in the file is id\tthen the url
                for line in f:
                    id, url = line.strip().split(' \t ')
                    if id in unique_ids:
                        id2url[id] = url
        
        # build a big tensor containing all image features, of size (N, d_model), with N = len(unique_ids)
        # save a dict idx -> id mapping
        id2idx = {id_: idx for idx, id_ in enumerate(unique_ids)}
        idx2id = {idx: id_ for idx, id_ in enumerate(unique_ids)}
        
        all_image_features = [None] * len(unique_ids)
        
        async def download_image(session, pbar, sem, idx):
            async with sem:
                try:
                    url = id2url[idx2id[idx]]
                    async with session.get(url, timeout=10) as resp:
                        if resp.status == 200:
                            img = Image.open(BytesIO(await resp.read()))
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            if 1 in img.size:
                                raise ValueError("Image size is too small.")
                            img_feat = _preprocess_img(img, model, processor, max_length)
                            all_image_features[idx] = img_feat
                except Exception:
                    pass
                finally:
                    pbar.update(1)
        
        async def main(concurrency=40):
            sem = asyncio.Semaphore(concurrency)
            pbar = tqdm(total=len(unique_ids), desc="Processing images")
            async with aiohttp.ClientSession() as session:
                tasks = []
                for i in range(len(unique_ids)):
                    tasks.append(download_image(session, pbar, sem, i))
                await asyncio.gather(*tasks)
            pbar.close()

        nest_asyncio.apply()
        async def runner():
            await main()
        
        asyncio.run(runner())
        
        # replace None with tensors of zeros
        # find out d :
        d_model = None
        for img_feat in all_image_features:
            if img_feat is not None:
                d_model = img_feat.shape[0]
                break

        # replace None with tensors of zeros
        for i in range(len(all_image_features)):
            if all_image_features[i] is None:
                all_image_features[i] = torch.zeros(d_model, device=device)
                
        all_image_features = torch.stack(all_image_features, dim=0).to(device)
        
        candidate_idx = [id2idx[cap['candidate']] for cap in meta]
        target_idx = [id2idx[cap['target']] for cap in meta]

        candidate_features = all_image_features[candidate_idx]
        target_features = all_image_features[target_idx]
        
        #####
        # Captions :
        #####
        
        text_features = [None] * len(meta)
        
        for i in tqdm(range(len(meta)), desc="Processing captions"):
            c = 0
            n = 0
            for cap in meta[i]['captions']:
                n += 1
                txt_feat = _preprocess_txt([cap], model, processor, max_length)
                c += txt_feat
            text_features[i] = c / n if n > 0 else None
        
        # if there are None, raise an error, this should not happen
        if any(txt_feat is None for txt_feat in text_features):
            raise ValueError("Some text features are None, this should not happen.")
        text_features = torch.stack(text_features, dim=0).to(device)
        # save the features
        os.makedirs(base_path, exist_ok=True)
        torch.save(candidate_features.cpu(), cdd_path)
        torch.save(target_features.cpu(), trg_path)
        torch.save(text_features.cpu(), txt_path)
        
    print(f"Candidate features shape: {candidate_features.shape}, memory: {candidate_features.element_size() * candidate_features.nelement() / 1024**2:.2f} MB")
    print(f"Target features shape: {target_features.shape}, memory: {target_features.element_size() * target_features.nelement() / 1024**2:.2f} MB")
    print(f"Text features shape: {text_features.shape}, memory: {text_features.element_size() * text_features.nelement() / 1024**2:.2f} MB")
    
    """
    captions_path = "/datasets/shared_datasets/fashion_iq/captions/" # cap.{itemtype}.{split}.json
    meta = []
    for itemtype in ["dress", "shirt", "toptee"]:
        with open(os.path.join(captions_path, f"cap.{itemtype}.{split}.json"), 'r') as f:
            meta.extend(json.load(f))
    
    # extract the list of unique ids used in this split.
    unique_trg = list(set([cap['target'] for cap in meta]))
    unique_cdd = list(set([cap['candidate'] for cap in meta]))
    unique_ids = list(set(unique_trg + unique_cdd))

    # find out their urls
    id2url = {}
    for itemtype in ["dress", "shirt", "toptee"]:
        with open(
            os.path.join(
                "/datasets/shared_datasets/fashion_iq", "image_url", f"asin2url.{itemtype}.txt"
            ), 'r'
        ) as f:
            # each line in the file is id\tthen the url
            for line in f:
                id, url = line.strip().split(' \t ')
                if id in unique_ids:
                    id2url[id] = url
    
    for i in range(10):
        j = random.randint(0, len(meta) - 1)
        print(f"Caption {j}: {meta[j]['captions']}")
        # plot the candidate image :
        cdd_id = meta[j]['candidate']
        cdd_url = id2url[cdd_id]
        
        try:
            with requests.get(cdd_url, stream=True, timeout=10) as r:
                if r.status_code == 200:
                    img = Image.open(BytesIO(r.content))
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    if 1 in img.size:
                        continue
        except requests.exceptions.RequestException:
            continue
        print(f"Candidate Image URL: {cdd_url}")
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        
        trg_id = meta[j]['target']
        trg_url = id2url[trg_id]
        try:
            with requests.get(trg_url, stream=True, timeout=10) as r:
                if r.status_code == 200:
                    img = Image.open(BytesIO(r.content))
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    if 1 in img.size:
                        continue
        except requests.exceptions.RequestException:
            continue
        print(f"Target Image URL: {trg_url}")
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    """
    
    return candidate_features, target_features, text_features

##########
# Miscellaneous
##########

supported_archi = ["ReLU", "relu", "JumpReLU", "jumprelu", "TopK", "topk", "BatchTopK", "batchtopk", "MP", "mp"]

def initialize_dictionary(archi, top_k, d_model, nb_concepts, maxialpha, batch_size, device="cpu"):
    normalisation = 'identity'
    if archi in ["ReLU", "relu"]:
        sae = overcomplete.sae.SAE(d_model, nb_concepts=nb_concepts, device=device, dictionary_params={'normalization': normalisation})
        alpha = 0.0
        desired_sparsity = top_k / nb_concepts
        update_rate = 0.001  # how aggressively alpha changes

        def update_alpha(alpha, current_sparsity, target_sparsity):
            current_sparsity = max(current_sparsity, 1e-12)
            target_sparsity = max(target_sparsity, 1e-12)
            
            # Adjust (on log scale, since sparsity starts very large and ends very small, when it goes below the target, the update steps in non log scale are too small)
            ratio = current_sparsity / target_sparsity
            alpha *= np.exp(update_rate * np.log(ratio))
            return max(alpha, 1e-12)

        def _criterion(x, x_hat, pre_codes, codes, dictionary, *args, **kwargs):
            if isinstance(codes, tuple):
                codes, _ = codes
                x, _ = x
                x_hat, _ = x_hat
                pre_codes, _ = pre_codes
                n = codes.shape[0]
            elif isinstance(codes, torch.Tensor):
                assert codes.shape[0] % 2 == 0, "batch size must be even"
                n = codes.shape[0] // 2
                codes = codes[:n] # shape: (n, d)
                x = x[:n]
                x_hat = x_hat[:n]
                pre_codes = pre_codes[:n]
            else:
                raise ValueError("codes must be either a tuple of two tensors or a single tensor")
    
            loss = (x - x_hat).square().mean()
            current_sparsity = (codes > 0).float().mean().item()

            sparsity_loss = codes.abs().mean()
            
            is_dead = ((codes > 0).sum(dim=0) == 0).float().detach()
            reanim_loss = (pre_codes * is_dead[None, :]).mean()

            loss += sparsity_loss * maxialpha.alpha - reanim_loss * 1e-3

            maxialpha.alpha = update_alpha(maxialpha.alpha, current_sparsity, desired_sparsity)
            return loss
        criterion_1 = lambda *args, **kwargs: _criterion(*args, **kwargs)
    elif archi in ["JumpReLU", "jumprelu"]:
        sae = overcomplete.sae.JumpSAE(d_model, nb_concepts=nb_concepts, device=device, dictionary_params={'normalization': normalisation})
        alpha = 0.0
        desired_sparsity = top_k / nb_concepts
        def _criterion(x, x_hat, pre_codes, codes, dictionary, *args, **kwargs):
            # here we directly use the thresholds of the model to control the sparsity
            if isinstance(codes, tuple):
                codes, _ = codes
                x, _ = x
                x_hat, _ = x_hat
                pre_codes, _ = pre_codes
                n = codes.shape[0]
            elif isinstance(codes, torch.Tensor):
                assert codes.shape[0] % 2 == 0, "batch size must be even"
                n = codes.shape[0] // 2
                codes = codes[:n] # shape: (n, d)
                x = x[:n]
                x_hat = x_hat[:n]
                pre_codes = pre_codes[:n]
            else:
                raise ValueError("codes must be either a tuple of two tensors or a single tensor")
    
            loss = (x - x_hat).square().mean()

            sparsity = (codes > 0).float().mean().detach()
            if sparsity > desired_sparsity:
                # if we are not sparse enough, increase the thresholds levels
                loss -= sae.thresholds.sum()
            
            is_dead = ((codes > 0).sum(dim=0) == 0).float().detach()
            # we push the pre_codes (before relu) towards the positive orthant
            reanim_loss = (pre_codes * is_dead[None, :]).mean()

            loss -= reanim_loss * 1e-3

            return loss
        criterion_1 = lambda *args, **kwargs: _criterion(*args, **kwargs)
    elif archi in ["TopK", "topk"]: 
        sae = overcomplete.sae.TopKSAE(d_model, nb_concepts=nb_concepts, top_k=top_k, device=device, dictionary_params={'normalization': normalisation})
        alpha = 1e-3  # Dead features loss penalty
        def _criterion(x, x_hat, pre_codes, codes, dictionary, *args, **kwargs):
            # here we directly use the thresholds of the model to control the sparsity
            if isinstance(codes, tuple):
                codes, _ = codes
                x, _ = x
                x_hat, _ = x_hat
                pre_codes, _ = pre_codes
                n = codes.shape[0]
            elif isinstance(codes, torch.Tensor):
                assert codes.shape[0] % 2 == 0, "batch size must be even"
                n = codes.shape[0] // 2
                codes = codes[:n] # shape: (n, d)
                x = x[:n]
                x_hat = x_hat[:n]
                pre_codes = pre_codes[:n]
            else:
                raise ValueError("codes must be either a tuple of two tensors or a single tensor")
    
            loss = (x - x_hat).square().mean()

            is_dead = ((codes > 0).sum(dim=0) == 0).float().detach()
            # we push the pre_codes (before relu) towards the positive orthant
            reanim_loss = (pre_codes * is_dead[None, :]).mean()

            loss -= reanim_loss * alpha

            return loss
        criterion_1 = _criterion
    elif archi in ["BatchTopK", "batchtopk"]:
        top_k_individual = top_k
        sae = overcomplete.sae.BatchTopKSAE(d_model, nb_concepts=nb_concepts, top_k=top_k_individual*batch_size, device=device, dictionary_params={'normalization': normalisation})
        alpha = 1e-3  # Dead features loss penalty
        def _criterion(x, x_hat, pre_codes, codes, dictionary, *args, **kwargs):
            if isinstance(codes, tuple):
                codes, _ = codes
                x, _ = x
                x_hat, _ = x_hat
                pre_codes, _ = pre_codes
                n = codes.shape[0]
            elif isinstance(codes, torch.Tensor):
                assert codes.shape[0] % 2 == 0, "batch size must be even"
                n = codes.shape[0] // 2
                codes = codes[:n] # shape: (n, d)
                x = x[:n]
                x_hat = x_hat[:n]
                pre_codes = pre_codes[:n]
            else:
                raise ValueError("codes must be either a tuple of two tensors or a single tensor")
    
            # here we directly use the thresholds of the model to control the sparsity
            loss = (x - x_hat).square().mean() / (x.square().mean() + 1e-8)

            is_dead = ((codes > 0).sum(dim=0) == 0).float().detach()
            # we push the pre_codes (before relu) towards the positive orthant
            reanim_loss = (pre_codes * is_dead[None, :]).mean()

            loss -= reanim_loss * alpha
            
            return loss
        criterion_1 = _criterion
    elif archi in ["MP", "mp"]:
        sae = overcomplete.sae.MpSAE(d_model, nb_concepts=nb_concepts, k=top_k, device=device, dictionary_params={'normalization': normalisation})
        alpha = 0.0 # L1 penalty
        criterion_1 = lambda *args, **kwargs: losses.mse_l1(*args, **kwargs, penalty=alpha)
    else:
        raise ValueError(f"Unsupported architecture: {archi}. Supported architectures: {supported_archi}")
    return sae, criterion_1, alpha

class SAE_loss:
    def __init__(self,
                 reconstruction_loss=None, # used for the demi-cycle consistency loss. If None, will use mse_loss.
                 alpha_tr=0., # Primary loss : translation reconstruction loss - requires matching pairs - supervised
                 alpha_cont=0., # Primary loss : contrastive loss - requires matching pairs - supervised
                 alpha_IsoE=0., # Secondary loss : IsoE loss - does not require matching pairs - self supervised
                 alpha_dcy=1., # Secondary loss : demi-cycle consistency loss (= reconstruction)
                               #                  does not require matching pairs - self supervised
                               # This will be the reconstruction_loss.
                 alpha_cy=0., # Secondary loss : cycle consistency loss - does not require matching pairs - self supervised
                 ):
        self.reconstruction_loss = reconstruction_loss if reconstruction_loss is not None else lambda *args, **kwargs: losses.mse_loss(*args, **kwargs, penalty=1e-3)
        self.alpha_tr = alpha_tr
        self.alpha_cont = alpha_cont
        self.alpha_IsoE = alpha_IsoE
        self.alpha_dcy = alpha_dcy
        self.alpha_cy = alpha_cy
    
    def __call__(self, *args, **kwargs):
        # don't compute the losses that will be multiplied by 0.
        l_tr = losses.tr_loss(*args, **kwargs) if self.alpha_tr > 0 else 0.
        l_cont = losses.cont_loss(*args, **kwargs) if self.alpha_cont > 0 else 0.
        l_dcy = self.reconstruction_loss(*args, **kwargs) if self.alpha_dcy > 0 else 0.
        l_cy = losses.cy_loss(*args, **kwargs) if self.alpha_cy > 0 else 0.
        l_IsoE = losses.IsoE_loss(*args, **kwargs) if self.alpha_IsoE > 0 else 0.

        return {
            "tr_loss": l_tr,
            "cont_loss": l_cont,
            "dcy_loss": l_dcy,
            "cy_loss": l_cy,
            "IsoE_loss": l_IsoE,
            "total_loss": (
            l_tr   * self.alpha_tr +
            l_cont * self.alpha_cont +
            l_dcy  * self.alpha_dcy +
            l_cy   * self.alpha_cy +
            l_IsoE * self.alpha_IsoE
        )}

def load_sae(sae_name, archi, top_k, d_model, expansion_factor, device="cpu"):
    nb_concepts = d_model * expansion_factor
    sae, _, _ = initialize_dictionary(
        archi, top_k, d_model, nb_concepts, None, 1, 
        device=device
    )
    sae.to(device)
    sae.eval()
    something = torch.load(f"./checkpoints/{sae_name}.pt", weights_only=False)
    sae_state_dict = something['model_state_dict']
    sae.load_state_dict(sae_state_dict)
    sae = sae.to(device)
    sae.train().eval()
    return sae

def _train(beta, train_loader, model_name, d_model=512, expansion_factor=8, top_k=20, lr=5e-4, epochs=5, archi="MP", dataset_name="laion", device="cpu", force_retrain=False, save_quand_meme=False, criterion_2_fct=None):
    nb_concepts = d_model * expansion_factor
    
    batch_size = train_loader.batch_size

    if archi not in supported_archi:
        raise ValueError(f"Unsupported architecture: {archi}. Supported architectures: {supported_archi}")
    
    class mouahahaIvenoideawhyglobaldoesntworksoletsdoanobject():
        def __init__(self, alpha=1e-3):
            self.alpha = alpha
    maxialpha = mouahahaIvenoideawhyglobaldoesntworksoletsdoanobject(alpha=1e-3)

    sae, criterion_1, alpha = initialize_dictionary(
        archi, top_k, d_model, nb_concepts, maxialpha, batch_size, device=device
    )

    sae.to(device)
    sae.train()

    optimizer = torch.optim.Adam(sae.parameters(), lr=5e-4)

    if criterion_2_fct is None:
        criterion_2 = lambda *args, **kwargs: losses.alignment_penalty(*args, **kwargs, penalty=beta, alignment_metric='cosim')
    else:
        criterion_2 = lambda *args, **kwargs: criterion_2_fct(*args, **kwargs, penalty=beta)
    criterion = lambda *args, **kwargs: criterion_1(*args, **kwargs) + criterion_2(*args, **kwargs)

    scheduler=None
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=epochs * steps_per_epoch,
    )

    alpha_name = str(alpha).replace('.', '')
    beta_name = str(beta).replace('.', '')

    CENTER_DATASET = False if 'CENTER_DATASET' not in locals() else CENTER_DATASET
    sae_name = model_name + "/"
    sae_name += f"{dataset_name}_{archi}_centered_{CENTER_DATASET}" + f"_{expansion_factor}_L0_{top_k}_alpha" + alpha_name + "beta" + beta_name + ""
    print(f"Model name: {sae_name}")

    found_one = False
    try:
        _sae = None
        something = torch.load(f"./checkpoints/{sae_name}.pt", weights_only=False)
        if force_retrain:
            found_one = True
            raise FileNotFoundError("Forcing retraining of the model.")
        
        _sae, _, _ = initialize_dictionary(
            archi, top_k, d_model, nb_concepts, maxialpha, batch_size, device=device
        )
        
        sae_state_dict = something['model_state_dict']
        _sae.load_state_dict(sae_state_dict)
        _sae = _sae.to(device)
        logs = None # something['logs']
    except FileNotFoundError:
        checkpoint_path = "./checkpoints" if (save_quand_meme or (not found_one)) else None
        logs = train.train_multimodal_sae(
            sae, train_loader, criterion, optimizer, scheduler=scheduler, nb_epochs=epochs, device=device,
            monitoring=1, verbose=True,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=5, checkpoint_name=sae_name,
        )
    if _sae is not None:
        sae = _sae
    sae.eval()
    return sae, sae_name, logs

def _train_cycle_isoE(loss, train_loaders,
                      model_name, d_model=512,
                      expansion_factor=8, top_k=20, archi="MP",
                      lr=5e-4, epochs=5, N_tokens=None, batch_size=None, dataset_name="laion",
                      device="cpu", run=None,
                      shared_init=False,
                      shared_weights=False,
                     ):
    # train a familly of SAEs with cycle consistency on several train loaders. When loss includes primary losses, the train loaders must provide matching pairs.
    
    nb_concepts = d_model * expansion_factor
    
    if batch_size is None:
        batch_size = train_loaders[0].batch_size

    if archi not in supported_archi:
        raise ValueError(f"Unsupported architecture: {archi}. Supported architectures: {supported_archi}")
    
    class mouahahaIvenoideawhyglobaldoesntworksoletsdoanobject():
        def __init__(self, alpha=1e-3):
            self.alpha = alpha
    maxialpha = mouahahaIvenoideawhyglobaldoesntworksoletsdoanobject(alpha=1e-3)

    saes = []
    losses = []
    
    for _ in range(len(train_loaders)):
        sae, criterion_1, alpha = initialize_dictionary(
            archi, top_k, d_model, nb_concepts, maxialpha, batch_size, device=device
        )
        sae.to(device).train()
        saes.append(sae)
        loss_ = SAE_loss(
            reconstruction_loss=criterion_1,
            alpha_tr=loss.alpha_tr,
            alpha_cont=loss.alpha_cont,
            alpha_IsoE=loss.alpha_IsoE,
            alpha_dcy=loss.alpha_dcy,
            alpha_cy=loss.alpha_cy,
        )
        losses.append(loss_)

    if shared_weights:
        for i in range(1, len(saes)):
            saes[i] = saes[0]
    elif shared_init:
        # share the same initialization for all saes but not the same weights during training
        base_state_dict = saes[0].state_dict()
        for i in range(1, len(saes)):
            saes[i].load_state_dict(base_state_dict)

    optimizers = [torch.optim.Adam(sae.parameters(), lr=lr) for sae in saes]

    schedulers = [None for _ in saes]
    # for i in range(len(saes)):
    #     steps_per_epoch = len(train_loaders[i])
    #     schedulers[i] = torch.optim.lr_scheduler.OneCycleLR(
    #         optimizers[i], max_lr=lr, total_steps=epochs * steps_per_epoch,
    #     )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sae_name = model_name + "/"
    sae_name += f"{dataset_name}_{archi}" + f"_{expansion_factor}_L0_{top_k}" + f"_cycleisoE_" + timestamp
    print(f"Model name: {sae_name}")

    checkpoint_path = "./checkpoints"
    logs = train.train_multiple_saes(
        saes, train_loaders, losses, optimizers, schedulers=schedulers, nb_epochs=epochs, N_tokens=N_tokens, device=device,
        monitoring=1, verbose=True,
        checkpoint_path=checkpoint_path,
        checkpoint_interval=5, checkpoint_name=sae_name,
        run=run,
    )
    
    for sae in saes:
        sae.eval()
    return saes, sae_name, logs

@dataclass
class Metrics:
    E: torch.Tensor  # Energy
    E_Img: torch.Tensor
    E_Txt: torch.Tensor
    f: torch.Tensor # Frequency per concept
    mu: torch.Tensor # Modality score
    to_save: list = field(default_factory=list)

    def _add_more(self, metric_dict, save=False):
        for key, value in metric_dict.items():
            setattr(self, key, value)
            if save and key not in self.to_save:
                self.to_save.append(key)

    # str function : print only the metrics that are in self.to_save
    def __str__(self):
        # Format : "{" then key : value, then ", " for each key-value pair, then "}"
        metrics_str = "{"
        for key in self.to_save:
            value = getattr(self, key)
            if isinstance(value, torch.Tensor):
                value = value.cpu().numpy()
            metrics_str += f"{key}: {value}, "
        metrics_str = metrics_str[:-2] + "}"  # Remove the last ", "
        return metrics_str

@torch.no_grad()
def measure_everything(sae, data_loader, device, return_sqr=False, modality_threshold=0.05):
    """
    - $E^{(\\textmd{d})} = \\mathbb{E}_{(\\textmd{d})}[z^2]$ : Energy
    - $f = \\mathbb{E}[\\mathbb{1}_{\\{z > 0\\}}]$ : Frequency
    - $\\mu = \\frac{E^{(\\mathrm{img})}}{E^{(\\mathrm{img})} + E^{(\\mathrm{text})}}$ : Modality Score
    """
    sae.eval()
    count_per_concept = torch.zeros(sae.nb_concepts).to(device)
    E_i = torch.zeros(sae.nb_concepts).to(device)
    E_i_sqr = torch.zeros(sae.nb_concepts).to(device)
    E_t = torch.zeros(sae.nb_concepts).to(device)
    E_t_sqr = torch.zeros(sae.nb_concepts).to(device)

    n_images = 0
    n_texts = 0

    for image_features, text_features in tqdm(data_loader):
        n_images += image_features.shape[0]
        n_texts += text_features.shape[0]

        image_features = image_features.to(device)
        text_features = text_features.to(device)

        # Forward pass through the model
        _, z_image, _ = sae(image_features)
        _, z_text, _ = sae(text_features)

        E_i += z_image.sum(dim=0)
        E_i_sqr += (z_image ** 2).sum(dim=0)
        E_t += z_text.sum(dim=0)
        E_t_sqr += (z_text ** 2).sum(dim=0)
        count_per_concept += (z_image != 0).sum(dim=0) + (z_text != 0).sum(dim=0)

    n_inputs = n_images + n_texts
    E = E_i + E_t
    E /= n_inputs
    E_sqr = E_i_sqr + E_t_sqr
    E_sqr /= n_inputs
    E_i /= n_images
    E_i_sqr /= n_images
    E_t /= n_texts
    E_t_sqr /= n_texts
    frequency_per_concept = count_per_concept / n_inputs
    modality_score = E_i / (E * 2)
    modality_sqr = E_i_sqr / (E_sqr * 2)

    dead_idx = (E < 1e-9).nonzero(as_tuple=True)[0]
    E_i[dead_idx] = 1000
    E_i_sqr[dead_idx] = 1000
    E_t[dead_idx] = 1000
    E_t_sqr[dead_idx] = 1000
    modality_score[dead_idx] = -1
    modality_sqr[dead_idx] = -1

    dead_mask = (E < 1e-9)
    mu = modality_sqr[~(dead_mask)] if return_sqr else modality_score[~(dead_mask)]
    energy = E_sqr[~(dead_mask)] if return_sqr else E[~(dead_mask)]
    mu_std_arround_05 = torch.sqrt(((mu - 0.5) ** 2 * energy).sum() / energy.sum())
    

    res = {
        "E": E_sqr if return_sqr else E,
        "E_Img": E_i_sqr if return_sqr else E_i,
        "E_Txt": E_t_sqr if return_sqr else E_t,
        "f": frequency_per_concept,
        "mu": modality_sqr if return_sqr else modality_score,
        # "mu_std_arround_05": mu_std_arround_05,
    }
    res = Metrics(**res)
    E = res.E
    mu = res.mu
    E_in_T = E[mu < modality_threshold].sum().item() / E.sum().item()
    E_in_I = E[mu > 1 - modality_threshold].sum().item() / E.sum().item()
    E_in_B = E[(mu >= modality_threshold) & (mu <= 1 - modality_threshold)].sum().item() / E.sum().item()
    res._add_more({"mu_std_arround_05": mu_std_arround_05}, save=True)
    res._add_more({"E_in_T": E_in_T, "E_in_I": E_in_I, "E_in_B": E_in_B}, save=True)
    return res

@torch.no_grad()
def _get_most_activating_samples(sae, data_loader, device, feature_idxs=None):
    """
    Select the top 10 activating image samples and text samples for each concept. # TODO : attribution on these.
    Then match the indices of these to images. Do it on COCO, to avoid downloading LAION.
    
    feature_idxs : None, int, list or tensor of ints. If not None, plot the visualisation for these. If None, just get all top 10 activating samples. and return that.
    """
    if isinstance(feature_idxs, int):
        feature_idxs = [feature_idxs]
    
    sae.eval()
    k = 10
    nb_c = sae.nb_concepts if feature_idxs is None else len(feature_idxs)
    
    top_vals_img = torch.full((nb_c, k), -float("inf"), device=device)
    top_idx_img  = torch.full((nb_c, k), -1, dtype=torch.long, device=device)
    top_vals_txt = torch.full((nb_c, k), -float("inf"), device=device)
    top_idx_txt  = torch.full((nb_c, k), -1, dtype=torch.long, device=device)
    
    offset = 0
    
    for img_feats, txt_feats in tqdm(data_loader):
        bsz = img_feats.size(0)
        img_feats, txt_feats = img_feats.to(device), txt_feats.to(device)

        _, z_img, _ = sae(img_feats)
        _, z_txt, _ = sae(txt_feats)
        
        if feature_idxs is not None:
            z_img = z_img[:, feature_idxs]
            z_txt = z_txt[:, feature_idxs]

        batch_idx = torch.arange(bsz, device=device) + offset
        offset += bsz
        
        batch_idx_exp = batch_idx.unsqueeze(1).expand(-1, nb_c)
        vals_cat = torch.cat([top_vals_img, z_img.T], dim=1)
        idx_cat  = torch.cat([top_idx_img, batch_idx_exp.T], dim=1)
        v, ix = torch.topk(vals_cat, k, dim=1)
        top_vals_img = v
        top_idx_img  = torch.gather(idx_cat, 1, ix)
        
        vals_cat = torch.cat([top_vals_txt, z_txt.T], dim=1)
        idx_cat  = torch.cat([top_idx_txt, batch_idx_exp.T], dim=1)
        v, ix = torch.topk(vals_cat, k, dim=1)
        top_vals_txt = v
        top_idx_txt  = torch.gather(idx_cat, 1, ix)

    return (top_vals_img.cpu(), top_idx_img.cpu(),
            top_vals_txt.cpu(), top_idx_txt.cpu())

@torch.no_grad()
def feature_visualisation(sae, model_name, device, feature_idxs=None, return_visualisation=False):
    print("Warning: TODO : add attribution maps to higlight the important bits of the images and texts.")
    if isinstance(feature_idxs, int):
        feature_idxs = [feature_idxs]
        
    base_path = "/datasets/shared_datasets/LAION400M/embeddings/" + model_name
    img_path = os.path.join(base_path, "image_features.pt")
    txt_path = os.path.join(base_path, "text_features.pt")

    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, return_img=False):
            self.image_path = "/datasets/shared_datasets/LAION400M/images/"
            self.captions_path = "/datasets/shared_datasets/LAION400M/captions/captions.json" # dict idx -> caption
            # find out the subset of images available from the captions file, from the idxs in the dict
            with open(self.captions_path, 'r') as f:
                self.captions = json.load(f)
            idxs = list(self.captions.keys())
            self.id2idx = {id: idx for idx, id in enumerate(idxs)}
            self.idx2id = {idx: id for idx, id in enumerate(idxs)}
            self.return_img = return_img

        def __len__(self):
            return len(self.captions)

        def __getitem__(self, idx):
            id = self.idx2id[idx]
            caption = self.captions[id]
            img_path = os.path.join(self.image_path, f"{id}.png")
            try:
                I = Image.open(img_path)
                if I.mode != 'RGB':
                    I = I.convert('RGB')
            except:
                # full black image of size 224x224
                I = Image.new('RGB', (224, 224), (0, 0, 0))
                caption = "Image not found"
            if self.return_img:
                return I, caption
            if caption is None or caption == "":
                caption = "No caption available"
            img_feat = _preprocess_img(I, model, processor, max_length)
            txt_feat = _preprocess_txt([caption], model, processor, max_length)
            return img_feat, txt_feat
    
    try:
        image_features = torch.load(img_path, map_location="cpu").to(device)
        text_features = torch.load(txt_path, map_location="cpu").to(device)    
    except FileNotFoundError:
        model, processor, max_length = _get_model(model_name, device=device)
        
        data_loader = torch.utils.data.DataLoader(CustomDataset(), batch_size=64, shuffle=False)
        image_features, text_features = [], []
        i=0
        for img_feat, txt_feat in tqdm(data_loader):
            image_features.append(img_feat.squeeze(0))
            text_features.append(txt_feat.squeeze(0))
            i += 1
        image_features = torch.cat(image_features, dim=0).to(device)
        text_features = torch.cat(text_features, dim=0).to(device)
        os.makedirs(base_path, exist_ok=True)
        torch.save(image_features.cpu(), img_path)
        torch.save(text_features.cpu(), txt_path)
    
    feature_dataset = torch.utils.data.TensorDataset(image_features, text_features)
    data_loader = torch.utils.data.DataLoader(feature_dataset, batch_size=64, shuffle=False)   
    I_val, I_idx, T_val, T_idx = _get_most_activating_samples(sae, data_loader, device, feature_idxs=feature_idxs)
    
    raw_dataset = CustomDataset(return_img=True)

    if feature_idxs is None:
        return I_val, I_idx, T_val, T_idx
    else:
        if return_visualisation:
            results = {}
        for i, fix in enumerate(feature_idxs):
            if isinstance(fix, torch.Tensor):
                fix = fix.item()
            if return_visualisation:
                results[fix] = []
            else:
                print(f"Feature {fix}:")
                print("Top activating images:")
            n_good = (I_val[i] > 1e-4).sum().item()
            if n_good == 0:
                print("No images found for this feature.")
                continue
            
            # make a plt plot of size n_good to plot the images. Add captions with activation values.
            if not return_visualisation:
                fig, axs = plt.subplots(1, n_good, figsize=(n_good * 3, 3))
            if n_good == 1:
                axs = [axs]

            for idx, j in enumerate((I_val[i] > 1e-4).nonzero(as_tuple=True)[0]):
                iix = I_idx[i, j].item()
                img, txt = raw_dataset[iix]
                if return_visualisation:
                    results[fix].append({"image": img, "activation": I_val[i, j].item()})
                else:
                    axs[idx].imshow(img)
                    axs[idx].axis('off')
                    axs[idx].set_title(f"{I_val[i, j].item():.2f}\nIdx:{iix}\n{txt}")
            if not return_visualisation:
                plt.tight_layout()
                plt.show()
            
                print("Top activating texts:")
                for idx, j in enumerate((T_val[i] > 1e-4).nonzero(as_tuple=True)[0]):
                    tix = T_idx[i, j].item()
                    img, txt = raw_dataset[tix]
                    print(f"Text {j}: {T_val[i, j].item()} - Index: {tix}, caption:\n\t{txt}")
        if return_visualisation:
            return results

@torch.no_grad()
def image_dissection(sae, model_name, device, img):
    model, processor, max_length = _get_model(model_name, device=device)
    img_feat = _preprocess_img(img, model, processor, max_length).squeeze().unsqueeze(0).to(device)
    _, z, _ = sae(img_feat)
    z = z.squeeze()
    idxs = torch.argsort(z, descending=True)
    idxs = idxs[z[idxs] > 0]
    z = z[idxs]
    mu = sae.metrics.mu[idxs]

    visu = feature_visualisation(sae, model_name, device, feature_idxs=idxs, return_visualisation=True)

    fig = go.FigureWidget(make_subplots(
        rows=1, cols=2,
        column_widths=[0.4, 0.6],
        subplot_titles=("Original Image", "Feature Activations")
    ))

    # Left: Original image
    buf = BytesIO()
    img.save(buf, format="PNG")
    encoded_img = base64.b64encode(buf.getvalue()).decode()
    fig.add_layout_image(
        dict(source=f"data:image/png;base64,{encoded_img}",
             xref="x domain", yref="y domain",
             x=0, y=1, sizex=1, sizey=1,
             xanchor="left", yanchor="top", layer="below"),
        row=1, col=1
    )
    fig.update_xaxes(visible=False, row=1, col=1)
    fig.update_yaxes(visible=False, row=1, col=1)

    # Right: Bar chart with customdata
    customdata = []
    for idx in idxs:
        imgs_b64 = []
        for top in visu[idx.item()][:3]:
            buf = BytesIO()
            top['image'].convert("RGB").save(buf, format="PNG")
            imgs_b64.append(base64.b64encode(buf.getvalue()).decode())
        customdata.append(imgs_b64)

    fig.add_trace(
        go.Bar(
            y=(len(idxs) - torch.arange(len(idxs))-1).cpu().numpy(),
            x=z.cpu().numpy(),
            orientation='h',
            text=[f"Idx {idxs[i]}<br>Strength: {z[i]:.2e}<br>Modality: {mu[i]:.2f}" for i in range(len(idxs))],
            textposition='inside',
            marker=dict(color='rgba(0, 200, 255, 0.5)'),
            customdata=customdata,
            hoverinfo='none'
        ),
        row=1, col=2
    )

    fig.update_layout(
        title="Image Dissection",
        margin=dict(l=40, r=40, t=40, b=40),
        height=600,
        width=1000,
        template='plotly_white'
    )

    fig.update_traces(marker_line_color='rgba(0, 200, 255, 0.8)',
                      marker_line_width=1.5, opacity=0.8)

    # Panel for clicked images
    out_fig = go.FigureWidget()
    out_fig.update_layout(height=300, width=300,
                          xaxis=dict(visible=False), yaxis=dict(visible=False))

    def on_click(trace, points, selector):
        out_fig.data = []
        if points.point_inds:
            imgs_b64 = trace.customdata[points.point_inds[0]]
            for i, b64_img in enumerate(imgs_b64):
                out_fig.add_layout_image(
                    dict(source=f"data:image/png;base64,{b64_img}",
                         x=0, y=1-i, sizex=1, sizey=1,
                         xanchor="left", yanchor="top", layer="above")
                )

    fig.data[0].on_click(on_click)

    from IPython.display import display
    display(fig, out_fig)

@torch.no_grad()
def image_dissection_export_html(sae, model_name, device, img, output_html="./figures/image_dissection.html"):
    model, processor, max_length = _get_model(model_name, device=device)
    img_feat = _preprocess_img(img, model, processor, max_length).squeeze().unsqueeze(0).to(device)
    _, z, _ = sae(img_feat)
    z = z.squeeze()
    idxs = torch.argsort(z, descending=True)
    idxs = idxs[z[idxs] > 0]
    z = z[idxs]
    mu = sae.metrics.mu[idxs]

    visu = feature_visualisation(sae, model_name, device, feature_idxs=idxs, return_visualisation=True)

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.4, 0.6],
        subplot_titles=("Original Image", "Feature Activations")
    )
    # Left: Original image
    # If image width > 1920, resize by factor 1920/width
    if img.width > 1920:
        factor = 1920 / img.width
        new_size = (int(img.width * factor), int(img.height * factor))
        img = img.resize(new_size, Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG")
    encoded_img = base64.b64encode(buf.getvalue()).decode()
    fig.add_layout_image(
        dict(source=f"data:image/png;base64,{encoded_img}",
             xref="x domain", yref="y domain",
             x=0, y=1, sizex=1, sizey=1,
             xanchor="left", yanchor="top", layer="below"),
        row=1, col=1
    )
    fig.update_xaxes(visible=False, row=1, col=1)
    fig.update_yaxes(visible=False, row=1, col=1)

    # Right: Bar chart
    customdata = []
    labels = []
    for i, idx in enumerate(idxs):
        imgs_b64 = []
        for top in visu[idx.item()][:3]:
            buf = BytesIO()
            top['image'].convert("RGB").save(buf, format="PNG")
            imgs_b64.append(base64.b64encode(buf.getvalue()).decode())
        customdata.append(imgs_b64)
        labels.append(f"Idx {idx.item()}<br>Strength: {z[i]:.2e}<br>Modality: {mu[i]:.2f}")

    fig.add_trace(
        go.Bar(
            y=(len(idxs) - torch.arange(len(idxs))-1).cpu().numpy(),
            x=z.cpu().numpy(),
            orientation='h',
            text=labels,
            textposition='inside',
            marker=dict(color='rgba(0, 200, 255, 0.5)'),
            customdata=customdata,
            hoverinfo='text'
        ),
        row=1, col=2
    )

    fig.update_layout(
        title="Image Dissection",
        margin=dict(l=40, r=40, t=40, b=40),
        height=600,
        width=1000,
        template='plotly_white'
    )

    fig.update_traces(marker_line_color='rgba(0, 200, 255, 0.8)',
                      marker_line_width=1.5, opacity=0.8)

    # Give the figure an ID
    html_fig = pio.to_html(fig, include_plotlyjs='cdn', full_html=False, div_id="myFig")

    # JavaScript click handler
    js_callback = """
    <script>
    document.addEventListener("DOMContentLoaded", function(){
        var figDiv = document.getElementById("myFig");
        figDiv.on('plotly_click', function(event){
            var cd = event.points[0].customdata;
            var panel = document.getElementById('image-panel');
            panel.innerHTML = '';
            cd.forEach(function(b64){
                var img = document.createElement('img');
                img.src = 'data:image/png;base64,' + b64;
                img.style.width = '150px';
                img.style.margin = '5px';
                panel.appendChild(img);
            });
        });
    });
    </script>
    """

    # Panel div
    html_div = '<div id="image-panel" style="margin-top:20px; display:flex; flex-wrap:wrap;"></div>'

    with open(output_html, "w") as f:
        f.write(html_fig + html_div + js_callback)

    print(f"Interactive HTML saved to {output_html}")

@torch.no_grad()
def measure_bridge(sae, data_loader, total_energy, image_energy, text_energy, device, center=True, normalize=True, null_C=False, null_D=False, weight_type="cov"):
    """
    - $B = W \\odot (D \\cdot D^\\top)$
        - $W$ is a weight matrix that is used to emphasize the importance of certain pairs of atoms in the bridge score calculation.
        - It is any similarity measure you like, e.g. functional similarity ($\\Sigma$), distributional similarity ($\\Gamma$), or any other measure.
        - $D$ is the dictionary matrix, which contains the atoms (features) learned by the model.
        - $D \\cdot D^\\top$ is the Gram matrix of the dictionary atoms, representing geometric alignment in latent space.
    - $B_{\\Sigma} = \\Sigma \\odot (D \\cdot D^\\top)$
        - Bridge score is a measure of functional alignment (functional similarity + geometric alignment) between two domains $I$ and $T$
            - Higher is better : means both domains are semantically aligned.
        - Bridge values link functionally aligned atoms
        - $\\Sigma$ is the cross covariance matrix of the dictionary atoms across domains $I$ and $T$
    - $B_{\\Gamma} = \\Gamma \\odot (D \\cdot D^\\top)$
        - Bridge score is a measure of distributional alignment, or how similar both distributions are. This is just 1 - OT cost.
            - Higher is better : means both domains are distributionally aligned.
        - Bridge values link distributionally aligned atoms
        - $\\Gamma$ is the OT plan corresponding to the OT of feature distributions across modalities $I$ and $T$
    """
    assert weight_type in ["cov", "OT"], f"Unknown weight type: {weight_type}. Must be one of ['cov', 'OT']."
    sae.eval().to(device)
    bridge = torch.zeros(sae.nb_concepts, sae.nb_concepts).to(device)
    image_var = torch.zeros_like(image_energy).to(device)
    text_var = torch.zeros_like(text_energy).to(device)
    
    dead_mask = (total_energy < 1e-9).cpu().numpy()
    
    if null_D:
        # D = torch.randn_like(sae.dictionary._fused_dictionary)
        # D /= D.norm(dim=-1, keepdim=True)
        D = sae.dictionary._fused_dictionary[torch.randperm(sae.dictionary._fused_dictionary.shape[0], device=device)]
    else:
        D = sae.dictionary._fused_dictionary
    D_align = D @ D.T
    threshold = torch.kthvalue(D_align.view(-1), int(0.9 * D_align.numel())).values
    D_align[D_align.abs() < threshold] = 0
    
    if weight_type == "OT":
        X = sae.dictionary._fused_dictionary.cpu()
        a = image_energy.cpu().numpy()
        a[dead_mask] = 0
        a /= a.sum()
        b = text_energy.cpu().numpy()
        b[dead_mask] = 0
        b /= b.sum()
        if null_C:
            a = a[np.random.permutation(len(a))]
            b = b[np.random.permutation(len(b))]
        c, gamma = metrics.Wasserstein(X, X, a, b, return_plan=True, metric='cosim')
        bridge = torch.tensor(gamma, device=device)  # gamma is the optimal transport plan
        print(f"Optimal transport cost: {c:.2e}")
    elif weight_type == "cov":
        n_images = 0
        n_texts = 0

        for image_features, text_features in tqdm(data_loader):
            n_images += image_features.shape[0]
            n_texts += text_features.shape[0]

            image_features = image_features.to(device)
            text_features = text_features.to(device)

            # Forward pass through the model
            _, z_image, _ = sae(image_features)
            _, z_text, _ = sae(text_features)

            if center:
                z_image = z_image - image_energy
                z_text = z_text - text_energy
            if null_C:
                # Randomly permute the features to nullify the covariance
                z_image = z_image[torch.randperm(z_image.shape[0], device=device)]
                z_text = z_text[torch.randperm(z_text.shape[0], device=device)]
            bridge += torch.einsum('ij,ik->jk', z_image, z_text)
            if normalize:
                image_var += (z_image ** 2).sum(dim=0)
                text_var += (z_text ** 2).sum(dim=0)
        
        bridge /= n_images # We now have the cross covariance matrix of features between images and text

        dead_img_mask = (image_energy == 0)
        dead_txt_mask = (text_energy == 0)

        if normalize:
            image_var /= n_images
            text_var /= n_texts
            bridge[~ dead_img_mask, :][:, ~ dead_txt_mask] /= torch.sqrt(image_var[~ dead_img_mask][:, None] * text_var[None, ~ dead_txt_mask])
            bridge[dead_img_mask, :] = 0
            bridge[:, dead_txt_mask] = 0

        threshold = torch.kthvalue(bridge.view(-1), int(0.999 * bridge.numel())).values
        bridge[bridge.abs() < threshold] = 0
        
    bridge *= D_align
    bridge = bridge.to("cpu")
    
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # print(f"Number of NaN in bridge: {torch.isnan(bridge).sum()}")
    # print(f"mean bridge: {bridge.mean():.2e}, std: {bridge.std():.2e}")
    # print(f"min bridge: {bridge.min():.2e}, max: {bridge.max():.2e}")
    return bridge

def get_bridge_norms(bridge, sae, device="cpu"):
    """
    - $E_{B}^{(\\mathrm{img})} = B \\cdot \\mathbb{1}$ : Energy of feature $i$ in the Image domain that is spread in the Text domain - when $i$ is active in an image, is there a $j$ aligned and active in the corresponding caption?
    - $E_{B}^{(\\mathrm{txt})} = \\mathbb{1}^\\top \\cdot B$
    - $E_{B} = \\frac{1}{2}(E_{B}^{(\\mathrm{img})} + E_{B}^{(\\mathrm{text})})$ : Measure feature's usefullness for cross-modal alignment
    - $\\mu_B = \\frac{E_B^{(\\mathrm{img})}}{E_B^{(\\mathrm{img})} + E_B^{(\\mathrm{text})}}$ : Modality Score
    """
    norms_T = bridge.abs().sum(dim=0).to(device)
    norms_I = bridge.abs().sum(dim=1).to(device) 
    norms = norms_I + norms_T
    # E = sae.metrics.E
    # top_2_idxs = torch.topk(E, k=2, largest=True, sorted=False).indices
    # norms[top_2_idxs] = norms[top_2_idxs] * 0.1

    mu_B = norms_I / (norms_T + norms_I + 1e-12)
    sae.metrics._add_more({"E_B": norms/2, "E_B_I": norms_I, "E_B_T": norms_T, "mu_B": mu_B}, save=False)
    
    E = norms[norms > 1e-9]
    mu_B = mu_B[norms > 1e-9]
    mu_std_arround_05 = torch.sqrt(((mu_B - 0.5) ** 2 * E).sum() / E.sum())
    sae.metrics._add_more({"mu_B_std_arround_05": mu_std_arround_05}, save=True)

def get_rho(sae, bridge, normalize=True, modality_threshold=0.05):
    ii = bridge['I', 'I']
    it = bridge['I', 'T']
    ib = bridge['I', 'B']
    ti = bridge['T', 'I']
    tt = bridge['T', 'T']
    tb = bridge['T', 'B']
    bi = bridge['B', 'I']
    bt = bridge['B', 'T']
    bb = bridge['B', 'B']
    
    alpha = ib + tb + bi + bt + bb
    oneminusalpha = ii + it + ti + tt
    
    if normalize:
        E = sae.metrics.E / sae.metrics.E.sum()
        mu = sae.metrics.mu
        e = E[(mu > modality_threshold) & (mu < 1-modality_threshold)].sum().item()
    else:
        e = 0.5
    
    rho = (alpha / e) * ((1-e) / oneminusalpha)
    sae.metrics._add_more({
        "rho": rho,
    }, save=True)
    print(f"rho: {rho:.4f}")

@torch.no_grad()
def get_modality_gap(I, T, sae, data_name="default"):
    # Measure the modality gap :
    # - norm of difference in mean image and text embeddings
    # - wasserstein distance between the image and text embeddings distributions.
    N_subsample = int(1e4) # dataset has 1e6 images and texts, subsample to 1e4 for speed.
    # raise
    # # TODO : try modality gap wasserstein without 
    p1 = torch.randperm(I.shape[0])[:N_subsample]
    p2 = torch.randperm(T.shape[0])[:N_subsample]
    T = I[p2].cpu()
    I = I[p1].cpu() # shape (N_subsample, D)
    I_mean = I.mean(axis=0) # shape (D,)
    T_mean = T.mean(axis=0)
    DiM = np.linalg.norm(I_mean - T_mean)
    c = metrics.Wasserstein(I, T, metric='cosim')
    sae.metrics._add_more({
        f"modality_gap_{data_name}": {
            "DiM": DiM,
            "Wasserstein": c,
        }
    }, save=True)

# remove unimodal concepts, then plot PCA similar to above : D_bimodal, I_bimodal, T_bimodal
@torch.no_grad()
def _intervene(sae, train_loader, modality_score, device, complement=False, use_E_B=False, eps=0.05, tau=0.5, return_D=True):
    I_bimodal = torch.zeros_like(train_loader.dataset.tensors[0])
    T_bimodal = torch.zeros_like(train_loader.dataset.tensors[1])

    with torch.no_grad():
        sae.eval()

        n_images = 0
        n_texts = 0
        
        if not use_E_B:
            mask = (modality_score >= eps) & (modality_score <= 1 - eps)
        else:
            E_B = sae.metrics.E_B
            E = sae.metrics.E
            ratio = E_B / (E + 1e-8)
            mask = (ratio >= tau)
        if complement:
            mask = ~mask
        idx = mask.cpu()

        for image_features, text_features in tqdm(train_loader):
            n_i = image_features.shape[0]
            n_t = text_features.shape[0]

            image_features = image_features.to(device)
            _, image_codes = sae.encode(image_features)
            truncated_image_codes = image_codes.clone()
            truncated_image_codes[:, ~idx] = 0
            x_hat_image_truncated = sae.decode(truncated_image_codes)

            I_bimodal[n_images:n_images+n_i] = x_hat_image_truncated
            
            if text_features.dtype == image_features.dtype:
                text_features = text_features.to(device)
                _, text_codes = sae.encode(text_features)
                truncated_text_codes = text_codes.clone()
                truncated_text_codes[:, ~idx] = 0
                x_hat_text_truncated = sae.decode(truncated_text_codes)
                
                T_bimodal[n_images:n_images+n_i] = x_hat_text_truncated

            n_images += n_i
            n_texts += n_t

        n_inputs = n_images + n_texts

    if text_features.dtype == image_features.dtype and return_D:
        D_bimodal = torch.cat([I_bimodal, T_bimodal], dim=0)
    else:
        D_bimodal = None
    return I_bimodal, T_bimodal, D_bimodal

##########
# Plots
##########

# Plot energy per concept, concepts sorted by energy
def plot_energy(E, f, sae_name):
    plt.rcParams['font.size'] = 14
    
    sorted_indices = torch.argsort(E, descending=True)

    sorted_energy = E[sorted_indices]

    sorted_frequency = f[sorted_indices]
    sorted_frequency_smoothed = torch.zeros_like(sorted_frequency)
    sorted_frequency_smoothed[0] = sorted_frequency[0]
    for i in range(1, len(sorted_frequency)):
        sorted_frequency_smoothed[i] = (0.9 * sorted_frequency_smoothed[i - 1] + 0.1 * sorted_frequency[i]) if sorted_frequency[i] > 0 else 0

    sorted_ratio = sorted_energy / (sorted_frequency + 1e-8)
    sorted_ratio_smoothed = torch.zeros_like(sorted_ratio)
    sorted_ratio_smoothed[0] = sorted_ratio[0]
    for i in range(1, len(sorted_ratio)):
        sorted_ratio_smoothed[i] = (0.9 * sorted_ratio_smoothed[i - 1] + 0.1 * sorted_ratio[i]) if sorted_ratio[i] > 0 else 0

    idx = torch.arange(len(sorted_energy)) + 1

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Concept Index')
    ax1.set_ylabel('Energy', color=color)
    ax1.loglog(idx, sorted_energy.cpu(), color=color, label='E')
    ax1.loglog(idx, sorted_ratio.cpu(), color='teal', alpha=0.5)
    ax1.loglog(idx, sorted_ratio_smoothed.cpu(), color='teal', label='E / f')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Frequency', color=color)
    ax2.loglog(idx, sorted_frequency.cpu(), color=color, alpha=0.5)
    ax2.loglog(idx, sorted_frequency_smoothed.cpu(), color=color, label='f')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title(f"Concepts sorted by Energy")
    fig.legend(loc='lower left', bbox_to_anchor=(0.1, 0.1), bbox_transform=ax1.transAxes)

    os.makedirs(f"./figures/{sae_name.replace('.pt', '')}/energy", exist_ok=True)
    fig.savefig(f"./figures/{sae_name.replace('.pt', '')}/energy/energy.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"./figures/{sae_name.replace('.pt', '')}/energy/energy.pdf")

    plt.show()
    plt.rcdefaults()

# Energy cumulative distribution function
def plot_energy_cdf(E, sae_name):
    energy_perm = torch.argsort(E, descending=True)
    energy_perm_inv = torch.argsort(energy_perm)

    thresholds = torch.linspace(E.min() + 1e-10, 1, steps=100)
    # thresholds = torch.logspace(-6, 0, steps=60)

    # plot energy vs index, cumulative sum vs index :
    sorted_energy = E[energy_perm]
    sorted_energy = sorted_energy / sorted_energy.sum()
    cumsum_energy = 1 - sorted_energy.cumsum(dim=0)

    # find thresholds index in cumsum_energy
    thresholds_idx = []
    for threshold in thresholds:
        idx = (cumsum_energy < threshold).nonzero(as_tuple=True)[0][0]
        thresholds_idx.append(idx.item())
    thresholds_idx = torch.tensor(thresholds_idx)

    plt.figure(figsize=(10, 6))

    id_perm = torch.arange(energy_perm.shape[0])
    plt.plot(cumsum_energy.cpu().numpy(), label='Cumulative Energy', color='blue')
    # for i, threshold in enumerate(thresholds):
    #     plt.axvline(thresholds_idx[i], color='red', linestyle='--')
    plt.xlabel('Concepts (sorted by energy)')
    # plt.yscale('log')
    plt.ylabel('Cumulative Energy')
    plt.title('Cumulative Energy vs Index')
    plt.legend()
    os.makedirs(f"./figures/{sae_name.replace('.pt', '')}/energy", exist_ok=True)
    plt.savefig(f"./figures/{sae_name.replace('.pt', '')}/energy/cumsum_energy.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"./figures/{sae_name.replace('.pt', '')}/energy/cumsum_energy.pdf")
    plt.show()

def plot_E_vs_E_B(E, E_B, sae_name):
    plt.hexbin(E.cpu().numpy() + 1e-10, E_B.cpu().numpy() + 1e-10, gridsize=50, yscale='log', xscale="log", cmap='magma_r', C=E.cpu().numpy(), reduce_C_function=np.sum, norm=LogNorm(vmin=max(1e-4, E.cpu().numpy().min())))
    plt.loglog(torch.logspace(-6, 0, steps=100).cpu().numpy(), torch.logspace(-6, 0, steps=100).cpu().numpy(), 'k--', label='y=x')
    plt.colorbar(label='Energy density', extend='min')
    m = (E_B.cpu() / (E.cpu()+1e-10)).mean().item()
    plt.loglog(torch.logspace(-6, 0, steps=100).cpu().numpy(), m * torch.logspace(-6, 0, steps=100).cpu().numpy(), 'k--', label=f'y={m:.2f}x')
    plt.loglog(torch.logspace(-6, 0, steps=100).cpu().numpy(), 0.1 * torch.logspace(-6, 0, steps=100).cpu().numpy(), 'k--', label='y=0.1x')
    plt.xlabel('E')
    plt.ylabel('E_B')
    plt.xlim(1e-5, max(E.cpu().numpy()) * 1.1)
    plt.ylim(3e-7, max(E_B.cpu().numpy()) * 1.1)
    plt.title('E_B vs E')
    plt.legend()
    os.makedirs(f"./figures/{sae_name.replace('.pt', '')}/energy", exist_ok=True)
    plt.savefig(f"./figures/{sae_name.replace('.pt', '')}/energy/E_vs_E_B.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"./figures/{sae_name.replace('.pt', '')}/energy/E_vs_E_B.pdf")
    plt.show()

# plot modality score
def plot_modality_score(modality_score, energy, frequency, sae_name, mu_B=False):
    plt.rcParams['font.size'] = 14
    modality_temp = modality_score[modality_score != -1].cpu()
    energy = energy[modality_score != -1].cpu()
    frequency = frequency[modality_score != -1].cpu()

    modality_temp = modality_temp[frequency < 0.7]
    energy = energy[frequency < 0.7]

    mu = modality_temp.mean()
    std = modality_temp.std()
    
    bins = 30
    range_ = (0, 1)
    cmap = plt.get_cmap("managua")  # Replace with custom colormap if needed
    norm = plt.Normalize(vmin=range_[0], vmax=range_[1])
    
    plt.figure(figsize=(10, 6))
    counts, bin_edges = np.histogram(modality_temp.numpy(), bins=bins, range=range_)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    colors = cmap(norm(bin_centers))

    plt.bar(bin_centers, counts, width=bin_edges[1] - bin_edges[0], color=colors)
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), label='Modality Score', orientation='vertical', ax=plt.gca())
    plt.title("Distribution of Modality Scores")
    plt.xlabel("Modality Score (Image Energy / Total Energy)")
    plt.ylabel("Count")
    # plt.yscale("log")
    # plt.grid(axis='y', alpha=0.75)
    plt.text(0.05, 0.95, f"Mean: {mu:.4f}\nStd: {std:.4f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
    os.makedirs(f"./figures/{sae_name.replace('.pt', '')}/modality", exist_ok=True)
    plt.savefig(f"./figures/{sae_name.replace('.pt', '')}/modality/modality{"_B" if mu_B else ""}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"./figures/{sae_name.replace('.pt', '')}/modality/modality{"_B" if mu_B else ""}.pdf")
    plt.show()
    
    mu_weighted = (modality_temp * energy).sum() / energy.sum()
    std_weighted = torch.sqrt(((modality_temp - mu_weighted) ** 2 * energy).sum() / energy.sum())
    plt.figure(figsize=(10, 6))
    counts, bin_edges = np.histogram(modality_temp.numpy(), bins=bins, range=range_, weights=energy.numpy())
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    colors = cmap(norm(bin_centers))
    
    plt.bar(bin_centers, counts, width=bin_edges[1] - bin_edges[0], color=colors)
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), label='Weighted Modality Score', orientation='vertical', ax=plt.gca())
    plt.title("Distribution of Modality Scores")
    plt.xlabel("Modality Score (Image Energy / Total Energy) weighted by energy")
    plt.ylabel("Density")
    # plt.yscale("log")
    # plt.grid(axis='y', alpha=0.75)
    plt.text(0.05, 0.95, f"Mean: {mu_weighted:.4f}\nStd: {std_weighted:.4f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
    plt.savefig(f"./figures/{sae_name.replace('.pt', '')}/modality/modality{"_B" if mu_B else ""}_weighted.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"./figures/{sae_name.replace('.pt', '')}/modality/modality{"_B" if mu_B else ""}_weighted.pdf")
    plt.show()
    
    plt.rcdefaults()

# scatter points based on image vs text energy
def plot_energy_density(image_energy, text_energy, sae_name, weighted=False, mu_B=False):
    plt.rcParams['font.size'] = 16
    sorted_indices = torch.argsort(image_energy + text_energy, descending=True)
    x = image_energy[sorted_indices].cpu().numpy() + 1e-9
    y = text_energy[sorted_indices].cpu().numpy() + 1e-9

    live_idx = torch.tensor(x + y != 2000).nonzero(as_tuple=True)[0]
    x = x[live_idx]
    y = y[live_idx]

    plt.figure(figsize=(10, 10))
    x_line = torch.logspace(-9, 1, steps=100).cpu()
    plt.plot(x_line, x_line, '--', color='black', label='Modality in {0.1, 0.5, 0.9}')
    plt.plot(x_line, 9 * x_line, '--', color='black')
    plt.plot(x_line, x_line / 9, '--', color='black')
    steps = torch.logspace(-5, 1, steps=10).cpu()
    for i in steps:
        plt.plot(x_line, i - x_line, 'k--', alpha=0.5, label='Iso-Energy' if i == steps[0] else None)

    plt.hexbin(x, y, C=((x + y) / (x + y).sum() if weighted else None), reduce_C_function=(np.sum if weighted else None), gridsize=100, bins='log', cmap='magma_r', xscale='log', yscale='log', norm=(LogNorm(vmin=max(1e-4, ((x + y) / (x + y).sum()).min())) if weighted else None))
    plt.colorbar(label=('density' if weighted else "count"), extend=('min' if weighted else None))
    plt.xlabel('Image Energy')
    plt.ylabel('Text Energy')
    plt.title(f"Image vs Text Energy Density")
    plt.legend()
    #plt.grid()

    # Enforce square log-log plot
    all_vals = np.concatenate([x, y])
    min_val, max_val = all_vals.min(), all_vals.max()
    plt.xlim(min_val * 0.9, max_val * 1.1)
    plt.ylim(min_val * 0.9, max_val * 1.1)
    plt.gca().set_aspect('equal', adjustable='box')

    os.makedirs(f"./figures/{sae_name.replace('.pt', '')}/energy", exist_ok=True)
    plt.savefig(f"./figures/{sae_name.replace('.pt', '')}/energy/energy{"_B" if mu_B else ""}_scatter{("_weighted" if weighted else "")}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"./figures/{sae_name.replace('.pt', '')}/energy/energy{"_B" if mu_B else ""}_scatter{("_weighted" if weighted else "")}.pdf")

    plt.show()
    plt.rcdefaults()

def plot_energy_scatter_plotly(image_energy, text_energy, sae_name, weighted=False, mu_B=False):
    sorted_indices = torch.argsort(image_energy + text_energy, descending=True)
    x = image_energy[sorted_indices].cpu().numpy() + 1e-9
    y = text_energy[sorted_indices].cpu().numpy() + 1e-9
    indices = sorted_indices.cpu().numpy()
    
    live_idx = (x + y) != 2000
    x = x[live_idx]
    y = y[live_idx]
    indices = indices[live_idx]
    
    fig = go.Figure(
        data=go.Scatter(
            x=x,
            y=y,
            # log_x=True,
            # log_y=True,
            mode='markers',
            marker=dict(
                size=5,
                color=((x + y) / (x + y).sum() if weighted else None
                ),
                colorscale='magma_r',
                colorbar=dict(
                    title=('density' if weighted else "count"),
                    tickvals=np.linspace(0, 1, 5) if weighted else None,
                    ticktext=np.round(np.linspace(0, 1, 5), 2)
                ),
                showscale=weighted
            ),
            text=[f"Index: {idx}<br>Image Energy: {img:.2e}<br>Text Energy: {txt:.2e}" for idx, img, txt in zip(indices, x, y)],
            hoverinfo='text'
        )
    )
    
    xmin = x.min() * 0.9
    xmax = x.max() * 1.1
    ymin = y.min() * 0.9
    ymax = y.max() * 1.1
    minmin = min(xmin, ymin)
    maxmax = max(xmax, ymax)
    fig.update_layout(
        title=f"Image vs Text Energy Density {'(Weighted)' if weighted else ''}",
        xaxis_title='Image Energy',
        yaxis_title='Text Energy',
        xaxis=dict(type='log', range=[np.log10(minmin), np.log10(maxmax)]),
        yaxis=dict(type='log', range=[np.log10(minmin), np.log10(maxmax)]),
        showlegend=False,
        width=800,
        height=800,
    )
    
    os.makedirs(f"./figures/{sae_name.replace('.pt', '')}/energy", exist_ok=True)
    fig.write_html(f"./figures/{sae_name.replace('.pt', '')}/energy/energy{'_B' if mu_B else ''}_scatter{('_weighted' if weighted else '')}_plotly.html")
    fig.write_image(f"./figures/{sae_name.replace('.pt', '')}/energy/energy{'_B' if mu_B else ''}_scatter{('_weighted' if weighted else '')}_plotly.png", scale=2)
    fig.write_image(f"./figures/{sae_name.replace('.pt', '')}/energy/energy{'_B' if mu_B else ''}_scatter{('_weighted' if weighted else '')}_plotly.pdf", scale=2)
    fig.show()

def plot_modality_energy(sae, sae_name, mu_B=False):
    E = sae.metrics.E_B if mu_B else sae.metrics.E
    E_i = sae.metrics.E_B_I if mu_B else sae.metrics.E_Img
    E_t = sae.metrics.E_B_T if mu_B else sae.metrics.E_Txt
    f = sae.metrics.f
    mu = sae.metrics.mu_B if mu_B else sae.metrics.mu
    plot_modality_score(mu, E, f, sae_name, mu_B=mu_B)
    plot_energy_density(E_i, E_t, sae_name, weighted=True, mu_B=mu_B)

def mu_vs_mu_B(sae, sae_name, modality_threshold=0.05):
    # E = sae.metrics.E
    E = sae.metrics.E_B
    plt.hexbin(sae.metrics.mu.cpu(), sae.metrics.mu_B.cpu(), C=E.cpu(), reduce_C_function=np.sum, gridsize=50, cmap='magma_r', norm=LogNorm(vmin=max(1e-4, E.cpu().min())))
    plt.colorbar(label='Energy density', extend='min')
    mu = torch.zeros_like(sae.metrics.mu)
    mu[sae.metrics.mu < modality_threshold] = -1
    mu[sae.metrics.mu > 1-modality_threshold] = 1
    mu_B = torch.zeros_like(sae.metrics.mu_B)
    mu_B[sae.metrics.mu_B < modality_threshold] = -1
    mu_B[sae.metrics.mu_B > 1-modality_threshold] = 1
    agree = (mu.cpu() == mu_B.cpu()).cpu().float()
    plt.axhline(modality_threshold, color='black', linestyle='--', label='unimodality thresholds')
    plt.axhline(1-modality_threshold, color='black', linestyle='--')
    plt.axvline(modality_threshold, color='black', linestyle='--')
    plt.axvline(1-modality_threshold, color='black', linestyle='--')
    acc = (agree * E.cpu()).sum() / E.cpu().sum()
    plt.text(0.05, 0.97, f"Proportion of concepts with\nsame modality: {acc:.3f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
    plt.xlabel('mu')
    plt.ylabel('mu_B')
    plt.title('mu vs mu_B')
    plt.savefig(f"./figures/{sae_name.replace('.pt', '')}/modality/mu_vs_mu_B.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"./figures/{sae_name.replace('.pt', '')}/modality/mu_vs_mu_B.pdf")
    plt.show()

# plot all the regions' mass vs eps
def plot_bridge_mass_vs_eps(bridge, title, name, sae_name, modality_score, energy_per_concept, sae, weight_type="cov", skip_plot=False, modality_threshold=0.05):
    # Bridge vs Modality :

    # let eps be the threshold for unimodality : < eps => text, > 1 - eps => image, eps < x < 1 - eps => bimodal
    # How much bridge mass is in bimodal regions ? Bimodal <-> unimodal ? Text <-> image ? Text <-> Text ? Image <-> Image ?
    # Make a plot of ... vs eps, and stack the plots of all these regions' mass (they add up to 1)

    if not skip_plot:
        epss = torch.linspace(0, 0.5, steps=100)
        epss[0] = 1e-3

        bibi = []
        bibi_diagonal = []
        bit = []
        bii = []
        ti = []
        tt = []
        tt_diagonal = []
        ii = []
        ii_diagonal = []

        bridge_mass = bridge.abs().sum()

        for i, eps in enumerate(epss):
            text_mask = (modality_score < eps).cpu()
            text_idxs = torch.arange(modality_score.shape[0])[text_mask]
            image_mask = (modality_score > 1 - eps).cpu()
            image_idxs = torch.arange(modality_score.shape[0])[image_mask]
            bimodal_mask = (modality_score > eps).cpu() & (modality_score < 1 - eps).cpu()
            bimodal_idxs = torch.arange(modality_score.shape[0])[bimodal_mask]

            # Bridge mass between bimodal and bimodal :
            bridge_bimodal = bridge[bimodal_idxs][:, bimodal_idxs]
            bridge_bimodal_diagonal = bridge_bimodal.diagonal().abs().sum() / bridge_mass
            bridge_bimodal_mass = bridge_bimodal.abs().sum() / bridge_mass - bridge_bimodal_diagonal
            bibi.append(bridge_bimodal_mass.item())
            bibi_diagonal.append(bridge_bimodal_diagonal.item())

            # Bridge mass between bimodal and unimodal :
            bi_t = bridge[bimodal_idxs][:, text_idxs]
            t_bi = bridge[text_idxs][:, bimodal_idxs]
            bridge_bimodal_text = (bi_t.abs().sum() + t_bi.abs().sum()) / bridge_mass
            bit.append(bridge_bimodal_text.item())

            bi_i = bridge[bimodal_idxs][:, image_idxs]
            i_bi = bridge[image_idxs][:, bimodal_idxs]
            bridge_bimodal_image = (bi_i.abs().sum() + i_bi.abs().sum()) / bridge_mass
            bii.append(bridge_bimodal_image.item())

            # Bridge mass between unimodal and unimodal :
            bridge_text = bridge[text_idxs][:, text_idxs]
            bridge_text_diagonal = bridge_text.diagonal().abs().sum() / bridge_mass
            bridge_text_mass = bridge_text.abs().sum() / bridge_mass - bridge_text_diagonal
            tt.append(bridge_text_mass.item())
            tt_diagonal.append(bridge_text_diagonal.item())

            bridge_image = bridge[image_idxs][:, image_idxs]
            bridge_image_diagonal = bridge_image.diagonal().abs().sum() / bridge_mass
            bridge_image_mass = bridge_image.abs().sum() / bridge_mass - bridge_image_diagonal
            ii.append(bridge_image_mass.item())
            ii_diagonal.append(bridge_image_diagonal.item())

            i_t = bridge[image_idxs][:, text_idxs]
            t_i = bridge[text_idxs][:, image_idxs]
            bridge_text_image = (i_t.abs().sum() + t_i.abs().sum()) / bridge_mass
            ti.append(bridge_text_image.item())

        plt.figure(figsize=(10, 6))

        colors = [
            "indigo",
            "teal",
            "seagreen",
            "gold",
            "goldenrod",
            "firebrick",
        ]
        # colors = [
        #     "#5a2846",
        #     "#ffc964",
        #     "#80e6ff",
        #     "#00369b",
        #     "#7f1900",
        #     "#263550",
        # ]

        bibi, bibi_diagonal, bit, bii, tt, tt_diagonal, ii, ii_diagonal, ti = np.array(bibi), np.array(bibi_diagonal), np.array(bit), np.array(bii), np.array(tt), np.array(tt_diagonal), np.array(ii), np.array(ii_diagonal), np.array(ti)

        plt.stackplot(
            epss.numpy(),
            bibi + bibi_diagonal, bit, bii, tt + tt_diagonal, ii + ii_diagonal, ti,
            labels=[
                'Bimodal ↔ Bimodal',
                'Bimodal ↔ Text',
                'Bimodal ↔ Image',
                'Text ↔ Text',
                'Image ↔ Image',
                'Text ↔ Image',
            ],
            colors=colors,
        )
        plt.plot(epss.numpy(), bibi_diagonal, color='grey', linestyle='--')
        plt.plot(epss.numpy(), bibi + bibi_diagonal + bit + bii + tt_diagonal, color='grey', linestyle='--')
        plt.plot(epss.numpy(), bibi + bibi_diagonal + bit + bii + tt + tt_diagonal + ii_diagonal, color='grey', linestyle='--')

        plt.plot(epss.numpy(), bibi + bibi_diagonal + bit + bii, color='black', linestyle='--')
        plt.plot(epss.numpy(), bibi + bibi_diagonal + bit + bii + tt + tt_diagonal + ii + ii_diagonal, color='black', linestyle='--')

        plt.xlabel('ε (Unimodality threshold)')
        plt.ylabel('Bridge Mass Fraction')
        plt.title(title)
        # plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        
        plt.tight_layout()
        os.makedirs(f"./figures/{sae_name.replace('.pt', '')}/bridge", exist_ok=True)
        plt.savefig(f"./figures/{sae_name.replace('.pt', '')}/bridge/{name}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"./figures/{sae_name.replace('.pt', '')}/bridge/{name}.pdf")
        plt.show()
        
    # Plot the detail per region of bridge mass fraction
    eps = modality_threshold

    bridge_mass = bridge.abs().sum()

    text_mask = (modality_score < eps).cpu()
    image_mask = (modality_score > 1 - eps).cpu()
    bimodal_mask = (~text_mask) & (~image_mask)

    text_idxs = torch.arange(modality_score.shape[0])[text_mask]
    image_idxs = torch.arange(modality_score.shape[0])[image_mask]
    bimodal_idxs = torch.arange(modality_score.shape[0])[bimodal_mask]

    # Get region energy
    E_T = energy_per_concept[text_idxs].sum().item()
    E_I = energy_per_concept[image_idxs].sum().item()
    E_B = energy_per_concept[bimodal_idxs].sum().item()
    total_energy = E_T + E_I + E_B
    evec = [E_T, E_B, E_I]
    esize = [e / total_energy for e in evec]  # Normalize

    # Compute bridge mass fractions
    def mass(i1, i2):
        b = bridge[i1][:, i2]
        return (b.abs().sum())

    regions = {
        ('T', 'T'): mass(text_idxs, text_idxs).item(),
        ('T', 'B'): mass(text_idxs, bimodal_idxs).item(),
        ('B', 'T'): mass(bimodal_idxs, text_idxs).item(),
        ('T', 'I'): mass(text_idxs, image_idxs).item(),
        ('I', 'T'): mass(image_idxs, text_idxs).item(),
        ('B', 'B'): mass(bimodal_idxs, bimodal_idxs).item(),
        ('B', 'I'): mass(bimodal_idxs, image_idxs).item(),
        ('I', 'B'): mass(image_idxs, bimodal_idxs).item(),
        ('I', 'I'): mass(image_idxs, image_idxs).item(),
    }

    if not skip_plot:
        xpos = [0, esize[0], esize[0] + esize[1]]
        ypos = [0, esize[0], esize[0] + esize[1]]

        img = np.zeros((300, 300), dtype=float)

        labels = ['T', 'B', 'I']

        if weight_type == "gamma":
            baseline_value = 1.0
            baseline_str = f"Maximum cross-domain Stability : {baseline_value:.2f}"
        else:
            baseline_value = energy_per_concept.sum().item()
            baseline_str = f"Maximum cross-domain shared Energy : {baseline_value:.2f}"

        for i in range(3):
            for j in range(3):
                x = xpos[i]
                y = ypos[j]
                w = esize[i]
                h = esize[j]
                val = regions.get((labels[i], labels[j]), 0.0) # / bridge_mass
                img[int(x * 300):int((x + w) * 300), int(y * 300):int((y + h) * 300)] = val
        
        plt.figure(figsize=(6, 6))
        plt.imshow(img, cmap='viridis', vmin=0, vmax=baseline_value)
        plt.colorbar(label='Bridge Mass') # Fraction')
        # # plot boundaries :
        # for i in range(2):
        #     plt.axhline(ypos[i+1] * 300, color='black', linewidth=2)
        #     plt.axvline(xpos[i+1] * 300, color='black', linewidth=2)

        plt.gca().set_aspect('equal')
        plt.axis('off')# Add labels
        for i, label in enumerate(labels):
            x_center = (xpos[i] + esize[i] / 2) * 300
            y_center = (ypos[i] + esize[i] / 2) * 300
            plt.text(x_center, 305, label, ha='center', va='top', fontsize=12)  # Bottom labels
            plt.text(305, y_center, label, ha='left', va='center', fontsize=12)  # Right labels
        plt.title(title)
        plt.tight_layout()

        plt.text(0.05, 0.95, f"Total Bridge Mass {bridge_mass.item():.2f}\n{baseline_str}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

        outdir = f"./figures/{sae_name.replace('.pt', '')}/"
        plt.savefig(f"{outdir}/bridge/{name}_grid.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{outdir}/bridge/{name}_grid.pdf")
        plt.show()

    sae.metrics._add_more({
        name: regions,
    }, save=True)
    getattr(sae.metrics, name)['total_mass'] = bridge_mass.item()
    getattr(sae.metrics, name)['bad_over_good'] = (regions[('T', 'T')] + regions[('I', 'I')] + regions[('T', 'I')] + regions[('I', 'T')]) / (regions[('B', 'B')] + regions[('T', 'B')] + regions[('B', 'T')] + regions[('B', 'I')] + regions[('I', 'B')])

# threshold bridges with 99th percentile of null :
def plot_bridges(bridge_dict, sae, sae_name, modality_score, skip_plot=False):
    for weight_type in ["sigma", "gamma"]:
        bridge = bridge_dict[f"bridge_{weight_type}"].clone()
        plot_bridge_mass_vs_eps(bridge, f"Bridge Mass Distribution Across Modal Regions", f"bridge_mass_vs_eps_{weight_type}", sae_name, modality_score, sae.metrics.E, sae, weight_type=weight_type, skip_plot=skip_plot)

@torch.no_grad()
def plot_cosim_histogram(I, T, sae_name, subtitle=""):
    # Baseline (random isotropic Gaussian)
    Rd = torch.randn_like(I)
    Rd /= Rd.norm(dim=-1, keepdim=True)
    J = Rd[torch.randperm(Rd.shape[0], device='cpu')]
    Rd_baseline = torch.cosine_similarity(Rd, J, dim=1).cpu().numpy()

    # means
    mu_I = I.mean(dim=0)
    mu_T = T.mean(dim=0)
    mu_D = (mu_I + mu_T) / 2

    def _aux_cosim(A, B, mu_A=0, mu_B=0):
        A_rd = A[torch.randperm(A.shape[0], device='cpu')]
        B_rd = B[torch.randperm(B.shape[0], device='cpu')]
        AA = torch.cosine_similarity(A - mu_A, A_rd - mu_A, dim=1).cpu().numpy()
        BB = torch.cosine_similarity(B - mu_B, B_rd - mu_B, dim=1).cpu().numpy()
        AB_aligned = torch.cosine_similarity(A - mu_A, B - mu_B, dim=1).cpu().numpy()
        AB_rd = torch.cosine_similarity(A - mu_A, B_rd - mu_B, dim=1).cpu().numpy()
        return AA, BB, AB_aligned, AB_rd

    II, TT, IT_aligned, IT_rd = _aux_cosim(I, T)
    II_D, TT_D, IT_aligned_D, IT_rd_D = _aux_cosim(I, T, mu_A=mu_D, mu_B=mu_D)
    II_m, TT_m, IT_aligned_m, IT_rd_m = _aux_cosim(I, T, mu_A=mu_I, mu_B=mu_T)

    def plot_histograms(data_dict, title):
        plt.figure(figsize=(10, 5))
        all_data = np.concatenate(list(data_dict.values()))
        bins = np.linspace(all_data.min(), all_data.max(), 100)  # 50 bins
        for label, data in data_dict.items():
            plt.hist(data, bins=bins, alpha=0.6, label=label, density=True)
        # plt.axvline(threshold, color='red', linestyle='--', linewidth=1, label='Threshold')
        plt.title(title)
        # plt.xlabel('Cosine Similarity')
        # plt.ylabel('Density')
        plt.xticks(fontsize=14)
        ax = plt.gca()
        ax.yaxis.set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.legend()
        # plt.grid(True)
        plt.tight_layout()
        os.makedirs(f"./figures/{sae_name.replace('.pt', '')}/geometry", exist_ok=True)
        plt.savefig(f"./figures/{sae_name.replace('.pt', '')}/geometry/{title.replace(' ', '_').lower()}_{subtitle.replace(' ', '_').lower()}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"./figures/{sae_name.replace('.pt', '')}/geometry/{title.replace(' ', '_').lower()}_{subtitle.replace(' ', '_').lower()}.pdf")
        plt.show()

    plot_histograms({
        "Random (N(0,I))": Rd_baseline,
        "I-I": II,
        "I-T": [],
        "T-T": TT,
        "I-T (aligned)": [],
    }, "Uncentered Cosine Similarity Distributions")

    plot_histograms({
        "Random (N(0,I))": Rd_baseline,
        "I-I": II,
        "I-T": IT_rd,
        "T-T": TT,
        "I-T (aligned)": [],
    }, "Uncentered Cosine Similarity Distributions")

    plot_histograms({
        "Random (N(0,I))": Rd_baseline,
        "I-I": II,
        "I-T": IT_rd,
        "T-T": TT,
        "I-T (aligned)": IT_aligned,
    }, "Uncentered Cosine Similarity Distributions")

    plot_histograms({
        "Random (N(0,I))": Rd_baseline,
        "I-I": II_D,
        "I-T": IT_rd_D,
        "T-T": TT_D,
        "I-T (aligned)": IT_aligned_D,
    }, "Centered Cosine Similarity Distributions")

    plot_histograms({
        "Random (N(0,I))": Rd_baseline,
        "I-I": II_m,
        "I-T": IT_rd_m,
        "T-T": TT_m,
        "I-T (aligned)": IT_aligned_m,
    }, "Modality-Centered Cosine Similarity Distributions")

    # Label ZOULOU_69 : threshold for
    # COCO : 0.238
    # CC3M : /
    # LAION 400M : None - already thresholded at 0.3. 

def plot_pca(D, d1, d2, I_mu, T_mu, D_mu, title, save_title, sae_name):
    plt.rcParams['font.size'] = 14
    modality_color = {
        "bimodal": np.array([91/255, 40/255, 68/255]),
        "image": np.array([128/255, 230/255, 255/255]),
        "text": np.array([255/255, 205/255, 102/255]),
    }
    def make_alpha_cmap(base_color, name):
        r, g, b = base_color
        return LinearSegmentedColormap.from_list(
            name,
            [(r, g, b, 0.0), (r, g, b, 1.0)]
        )

    I_cmap = make_alpha_cmap(modality_color["image"], "I")
    T_cmap = make_alpha_cmap(modality_color["text"], "T")
    # Plot the projections, colored by modality (D[:N//2] = I, D[N//2:] = T)
    plt.figure(figsize=(10, 7))
    plt.hexbin(D[:D.shape[0]//2].cpu().numpy() @ d1.reshape(-1, 1), D[:D.shape[0]//2].cpu().numpy() @ d2.reshape(-1, 1), gridsize=50, cmap=I_cmap, bins='log')
    plt.colorbar(label='Image Counts')
    plt.hexbin(D[D.shape[0]//2:].cpu().numpy() @ d1.reshape(-1, 1), D[D.shape[0]//2:].cpu().numpy() @ d2.reshape(-1, 1), gridsize=50, cmap=T_cmap, bins='log')
    plt.colorbar(label='Text Counts')
    # plt.axhline(0, color='k', linestyle='--', linewidth=1)
    # plt.axvline(0, color='k', linestyle='--', linewidth=1)

    # Plot all the means of the modalities
    plt.scatter(D_mu.cpu().numpy() @ d1.reshape(-1, 1), D_mu.cpu().numpy() @ d2.reshape(-1, 1), color='k', marker='x', s=100)
    plt.scatter(I_mu.cpu().numpy() @ d1.reshape(-1, 1), I_mu.cpu().numpy() @ d2.reshape(-1, 1), color='k', marker='x', s=100)
    plt.scatter(T_mu.cpu().numpy() @ d1.reshape(-1, 1), T_mu.cpu().numpy() @ d2.reshape(-1, 1), color='k', marker='x', s=100)

    # plt.xlabel('d1')
    # plt.ylabel('d2')
    # plt.title(title)
    # plt.legend()
    # plt.grid(True)
    plt.axis('off')
    plt.tight_layout()
    os.makedirs(f"./figures/{sae_name.replace('.pt', '')}/geometry", exist_ok=True)
    plt.savefig(f"./figures/{sae_name.replace('.pt', '')}/geometry/{save_title}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"./figures/{sae_name.replace('.pt', '')}/geometry/{save_title}.pdf")
    plt.show()
    plt.rcdefaults()

def fit_and_plot_pca(I, T, D, sae_name, save_title='data_pca_pc1_pc2'):
    # Fit PCA on D
    D_np = D.cpu().numpy()
    pca = PCA(n_components=2)
    pca.fit(D_np)
    mean = pca.mean_
    d1 = pca.components_[0]
    d2 = pca.components_[1]
    D_pca = pca.transform(D_np)

    I_mu = I.mean(dim=0).cpu()
    T_mu = T.mean(dim=0).cpu()
    D_mu = D.mean(dim=0).cpu()

    plot_pca(D.cpu()-mean, d1, d2, I_mu-mean, T_mu-mean, D_mu-mean, title='Projection of D on d1 = PC1 and d2 = PC2', save_title=save_title, sae_name=sae_name)

# Visualize dictionary atoms :

def generate_visualization(sae, sae_name, modality_score, energy_per_concept, frequency_per_concept, bridge):
    # Create a UMAP of the concepts in D, colored by modality score and size by energy

    non_dead_idx = (modality_score != -1).nonzero(as_tuple=True)[0].cpu().numpy()

    D = sae.dictionary._fused_dictionary
    D_np = D.detach().cpu().numpy()[non_dead_idx]
    modality_np = modality_score.detach().cpu().numpy()[non_dead_idx]
    energy_np = energy_per_concept.detach().cpu().numpy()[non_dead_idx]
    frequency_np = frequency_per_concept.detach().cpu().numpy()[non_dead_idx]
    bridge_matrix = bridge.numpy()[non_dead_idx][:, non_dead_idx] ** 2

    non_stupid_mask = (frequency_np < 0.6)
    D_np = D_np[non_stupid_mask]
    modality_np = modality_np[non_stupid_mask]
    energy_np = energy_np[non_stupid_mask]
    bridge_matrix = bridge_matrix[non_stupid_mask][:, non_stupid_mask]

    threshold = 2e-10
    i_idx, j_idx = np.nonzero(bridge_matrix > threshold)
    line_weights = bridge_matrix[i_idx, j_idx]

    # UMAP embedding
    embedding = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine').fit_transform(D_np)

    # Line segments from i to j in the UMAP space
    lines = [ [embedding[i], embedding[j]] for i, j in zip(i_idx, j_idx) ]

    print(f"Number of lines: {len(lines)}")

    # PCA embedding
    pca = PCA(n_components=2)
    embedding_pca = pca.fit_transform(D_np)

    # Line segments from i to j in the PCA space
    lines_pca = [ [embedding_pca[i], embedding_pca[j]] for i, j in zip(i_idx, j_idx) ]

    # Make a weighted PCA :
    W = energy_np / energy_np.sum()
    mu = np.average(D_np, axis=0, weights=W)
    D_centered = D_np - mu
    C = D_centered.T @ (D_centered * W[:, None])
    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    embedding_weighted_pca = D_centered @ eigvecs[:, :2]

    # Line segments from i to j in the PCA space
    lines_weighted_pca = [ [embedding_weighted_pca[i], embedding_weighted_pca[j]] for i, j in zip(i_idx, j_idx) ]


    # Normalize energy for plotting size
    LOG = False
    if LOG:
        # Log scale energy normalization
        log_energy = np.log(energy_np)  # Log transform with offset to avoid log(0)
        size = log_energy - log_energy.min()
        size = 0.1 * (size) ** 2
    else:
        # Linear scale energy normalization
        size = energy_np - energy_np.min()
        size = 500 * np.sqrt((size / size[size<0.1].max()))


    # Normalize weights for line width
    # weights_norm = 10 * (line_weights - line_weights.min()) / (line_weights.max() - line_weights.min()) + 1e-2
    weights_norm = np.clip(line_weights, 0, 100)
    weights_norm = 0.05 * (np.log(weights_norm) - np.log(weights_norm.min()))


    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=modality_np, s=size, cmap='managua')
    plt.colorbar(label='Modality Score')
    plt.title('UMAP of Dictionary Atoms')
    plt.axis('off')
    plt.tight_layout()
    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()
    os.makedirs(f"./figures/{sae_name.replace('.pt', '')}/geometry", exist_ok=True)
    plt.savefig(f"./figures/{sae_name.replace('.pt', '')}/geometry/umap" + (f"_log" if LOG else "") + ".png", dpi=300, bbox_inches='tight')
    plt.savefig(f"./figures/{sae_name.replace('.pt', '')}/geometry/umap" + (f"_log" if LOG else "") + ".pdf")
    plt.show()

    # Plot
    blackred_cmap = LinearSegmentedColormap.from_list('blackred', [(0, 0, 0, 0.1), (1, 0, 0, 1)], N=256)
    blackred = np.zeros_like(modality_np)
    # select randomly 20 points to be red, with probability given by weights
    p = size / size.sum()
    red_indices = np.random.choice(len(modality_np), size=int(20), p=p, replace=False)
    blackred[red_indices] = 1  # Set selected indices to 1 for red color
    plt.figure(figsize=(8, 6), facecolor="#FFE5DF00")
    plt.scatter(embedding[:, 0], embedding[:, 1], c=blackred, s=size, cmap=blackred_cmap, vmin=0, vmax=1)
    plt.colorbar(label='Modality Score')
    plt.title('UMAP of Dictionary Atoms')
    plt.axis('off')
    plt.tight_layout()
    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()
    # make the background of color FFC2B9 (light red)
    plt.gca().set_facecolor("#FFE5DF00")
    plt.savefig(f"./figures/{sae_name.replace('.pt', '')}/geometry/umap_black" + (f"_log" if LOG else "") + ".png", dpi=300, bbox_inches='tight')
    plt.savefig(f"./figures/{sae_name.replace('.pt', '')}/geometry/umap_black" + (f"_log" if LOG else "") + ".pdf")
    plt.show()

    # Plot
    bi_idx = (modality_np > 0.2) & (modality_np < 0.8)
    plt.figure(figsize=(8, 6))
    plt.scatter(embedding[bi_idx, 0], embedding[bi_idx, 1], c=modality_np[bi_idx], s=size[bi_idx], cmap='managua', vmin=0, vmax=1)
    plt.colorbar(label='Modality Score')
    plt.title('UMAP of Dictionary Atoms')
    plt.axis('off')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.tight_layout()
    plt.savefig(f"./figures/{sae_name.replace('.pt', '')}/geometry/umap_bi" + (f"_log" if LOG else "") + ".png", dpi=300, bbox_inches='tight')
    plt.savefig(f"./figures/{sae_name.replace('.pt', '')}/geometry/umap_bi" + (f"_log" if LOG else "") + ".pdf")
    plt.show()


    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=modality_np, s=size, cmap='managua')
    plt.colorbar(label='Modality Score')
    plt.title('UMAP of Dictionary Atoms')


    ax = plt.gca()

    # Self loops will be drawn as circles
    def curved_edge(p1, p2, curvature=0.2):
        # Midpoint and perpendicular vector for control point
        midpoint = 0.5 * (p1 + p2)
        delta = p2 - p1
        norm = np.linalg.norm(delta)
        if norm == 0:
            return None  # Skip self-loops or overlapping points
        perp = np.array([-delta[1], delta[0]]) / norm
        control = midpoint + curvature * norm * perp

        verts = [p1, control, p2]
        codes = [mplPath.MOVETO, mplPath.CURVE3, mplPath.CURVE3]
        return PathPatch(mplPath(verts, codes), facecolor='none', edgecolor='gray', lw=1, alpha=0.2)

    # Add curved patches for each bridge above threshold
    for k, (i, j, w) in enumerate(zip(i_idx, j_idx, line_weights)):
        if i == j:
            continue
        patch = curved_edge(embedding[i], embedding[j], curvature=0.15)
        if patch:
            patch.set_linewidth(weights_norm[k])
            ax.add_patch(patch)

    # Add self loops
    def draw_self_loop(ax, center, radius, color, lw, n_points=100):
        theta = np.linspace(0, 2 * np.pi, n_points)
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta) + radius  # upward offset by radius
        ax.plot(x, y, color=color, lw=lw, alpha=0.2, zorder=0)

    # Add self-loops using parametric circle
    for k, (i, j, w) in enumerate(zip(i_idx, j_idx, line_weights)):
        if i != j:
            continue  # Only self-loops

        center = embedding[i]
        node_size = size[i] ** 0.5 / 100  # approximate radius from scatter size
        loop_radius = node_size + weights_norm[k] * 0.5
        loop_thickness = weights_norm[k]

        draw_self_loop(ax, center, loop_radius, color='gray', lw=loop_thickness)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"./figures/{sae_name.replace('.pt', '')}/geometry/umap_bridge" + (f"_log" if LOG else "") + ".png", dpi=300, bbox_inches='tight')
    plt.savefig(f"./figures/{sae_name.replace('.pt', '')}/geometry/umap_bridge" + (f"_log" if LOG else "") + ".pdf")
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.scatter(embedding_pca[:, 0], embedding_pca[:, 1], c=modality_np, s=size, cmap='managua')
    plt.colorbar(label='Modality Score')
    plt.title('PCA of Dictionary Atoms')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"./figures/{sae_name.replace('.pt', '')}/geometry/pca" + (f"_log" if LOG else "") + ".png", dpi=300, bbox_inches='tight')
    plt.savefig(f"./figures/{sae_name.replace('.pt', '')}/geometry/pca" + (f"_log" if LOG else "") + ".pdf")
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.scatter(embedding_weighted_pca[:, 0], embedding_weighted_pca[:, 1], c=modality_np, s=size, cmap='managua')
    plt.colorbar(label='Modality Score')
    plt.title('Weighted PCA of Dictionary Atoms')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"./figures/{sae_name.replace('.pt', '')}/geometry/weighted_pca" + (f"_log" if LOG else "") + ".png", dpi=300, bbox_inches='tight')
    plt.savefig(f"./figures/{sae_name.replace('.pt', '')}/geometry/weighted_pca" + (f"_log" if LOG else "") + ".pdf")
    plt.show()

    # visualize an idealized version of "linear separability" :
    # 1. Create dummy data : 2D unit vectors following three Von Mises distributions
    # 2. assign a modality score to each point based on its angle + noise
    # 2.5 assign an energy depending on the cosim with the center of the distribution
    # 3. plot the points colored by modality score

    def von_mises_2d(mu, kappa, size):
        # Generate random angles from a von Mises distribution
        angles = np.random.vonmises(mu, kappa, size)
        # Convert angles to 2D unit vectors
        x = np.cos(angles)
        y = np.sin(angles)
        return np.column_stack((x, y))

    def generate_modality_score(points, noise=0.05):
        # Calculate angles of points
        angles = np.arctan2(points[:, 1], points[:, 0])
        # Normalize angles to [0, 1]
        normalized_angles = (1.5 * angles + np.pi) / (2 * np.pi)
        # Add noise
        noise = np.random.uniform(-noise, noise, size=normalized_angles.shape)
        modality_scores = normalized_angles + noise
        modality_scores = np.clip(modality_scores, 0, 1)
        return modality_scores

    def generate_energy(points, center, noise=5):
        # Calculate angles of points
        cosim = np.dot(points, center) / (np.linalg.norm(points, axis=1) * np.linalg.norm(center))
        # Normalize angles to [0, 1]
        normalized_cosim = (cosim + 1) / 2
        # Add noise
        noise = np.random.uniform(-noise, noise, size=normalized_cosim.shape)
        energy_scores = normalized_cosim + noise
        energy_scores -= energy_scores.min()
        energy_scores /= energy_scores.max()
        energy_scores = np.exp(energy_scores * 2) 
        return energy_scores

    def plot_von_mises(points, energies, modality_scores, title):
        plt.figure(figsize=(8, 8))
        plt.scatter(points[:, 0], points[:, 1], c=modality_scores, cmap='managua', s=energies * 30)
        plt.colorbar(label='Modality Score')
        plt.title(title)
        plt.axis('equal')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"./figures/{sae_name.replace('.pt', '')}/geometry/dummy_von_mises.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"./figures/{sae_name.replace('.pt', '')}/geometry/dummy_von_mises.pdf")
        plt.show()

    points1 = von_mises_2d(mu=0, kappa=5, size=100)
    energy1 = generate_energy(points1, np.array([1, 0]), noise=0.5)
    points2 = von_mises_2d(mu=2 * np.pi/3, kappa=100, size=100)
    energy2 = generate_energy(points2, np.array([-1, 0]), noise=0.5)
    points3 = von_mises_2d(mu=4 * np.pi/3, kappa=100, size=100)
    energy3 = generate_energy(points3, np.array([-1, 0]), noise=0.5)

    points = np.vstack((points1, points2, points3))
    energies = np.hstack((energy1, energy2, energy3))
    # plot_von_mises(points, energies, generate_modality_score(points), "Von Mises Distribution with Modality Score")

def linear_separability(sae, sae_name, eps=0.05):
    plt.rcParams['font.size'] = 14
    D = sae.dictionary._fused_dictionary.cpu().detach()
    modality_score = sae.metrics.mu
    energy_per_concept = sae.metrics.E
    print(D.shape)
    I_mask = (modality_score > 1 - eps).cpu()
    T_mask = (modality_score < eps).cpu()
    B_mask = (modality_score <= 1 - eps).cpu() & (modality_score >= eps).cpu()
    print(f"Number of unimodal concepts: {I_mask.sum()}, {T_mask.sum()}, {B_mask.sum()}")

    mu_weighted_I = (D[I_mask] * energy_per_concept[I_mask].unsqueeze(1).cpu()).sum(dim=0).cpu() / energy_per_concept[I_mask].sum().cpu()
    mu_weighted_T = (D[T_mask] * energy_per_concept[T_mask].unsqueeze(1).cpu()).sum(dim=0).cpu() / energy_per_concept[T_mask].sum().cpu()
    mu_weighted_B = (D[B_mask] * energy_per_concept[B_mask].unsqueeze(1).cpu()).sum(dim=0).cpu() / energy_per_concept[B_mask].sum().cpu()

    d1 = (mu_weighted_B - mu_weighted_I).cpu().numpy()
    d1 /= np.linalg.norm(d1)
    d2 = (mu_weighted_B - mu_weighted_T).cpu().numpy()
    d2 /= np.linalg.norm(d2)

    mu_tout_court = D.mean(dim=0).cpu().numpy()
    mu_weighted_tout_court = (D * energy_per_concept.unsqueeze(1).cpu()).sum(dim=0).cpu().numpy() / energy_per_concept.sum().cpu().numpy()

    D1 = D.cpu().numpy() # shape (N, 512)
    D1 = D1 @ d1.reshape(-1, 1) # shape (N, 1)
    D2 = D.cpu().numpy() # shape (N, 512)
    D2 = D2 @ d2.reshape(-1, 1) # shape (N, 1)
    D_proj = np.concatenate([D1, D2], axis=1) # shape (N, 2)

    modality_color = {
        "bimodal": colorsys.rgb_to_hls(*np.array([91/255, 40/255, 68/255])),
        "image": colorsys.rgb_to_hls(*np.array([128/255, 230/255, 255/255])),
        "text": colorsys.rgb_to_hls(*np.array([255/255, 205/255, 102/255])),
    }
    
    # Normalize energy for plotting size
    energy_np = energy_per_concept.cpu().numpy()
    E_I_np = sae.metrics.E_Img.cpu().numpy()
    E_T_np = sae.metrics.E_Txt.cpu().numpy()
    LOG = False
    if LOG:
        # Log scale energy normalization
        log_energy = np.log(energy_np)  # Log transform with offset to avoid log(0)
        size = log_energy - log_energy.min()
        size = 0.1 * (size) ** 2
    else:
        # Linear scale energy normalization
        size = energy_np - energy_np.min()
        size = 300 * (np.sqrt(size / size.max()))
        # Linear scale energy normalization
        size_I = E_I_np - E_I_np.min()
        size_I = 300 * (np.sqrt(size_I / size_I.max()))
        # Linear scale energy normalization
        size_T = E_T_np - E_T_np.min()
        size_T = 300 * (np.sqrt(size_T / size_T.max()))
        
    X = sae.dictionary._fused_dictionary.cpu().detach()
    dead_mask = (energy_per_concept < 1e-9).cpu().numpy()
    a = E_I_np
    a[dead_mask] = 0
    a /= a.sum()
    b = E_T_np
    b[dead_mask] = 0
    b /= b.sum()
    c, gamma = metrics.Wasserstein(X, X, a, b, return_plan=True, metric='cosim')
    bb = gamma[B_mask, :][:, B_mask].sum()
    ib = gamma[I_mask, :][:, B_mask].sum()
    tb = gamma[T_mask, :][:, B_mask].sum()
    bi = gamma[B_mask, :][:, I_mask].sum()
    ii = gamma[I_mask, :][:, I_mask].sum()
    ti = gamma[T_mask, :][:, I_mask].sum()
    bt = gamma[B_mask, :][:, T_mask].sum()
    it = gamma[I_mask, :][:, T_mask].sum()
    tt = gamma[T_mask, :][:, T_mask].sum()

    def _plot_proj_mu():
        plt.figure(figsize=(8, 6))
        plt.scatter(D_proj[:, 0], D_proj[:, 1], c=modality_score.cpu().numpy(), s=size, cmap='managua')
        plt.colorbar(label='Modality Score')
        plt.axhline(0, color='k', linestyle='--', linewidth=1)
        plt.axvline(0, color='k', linestyle='--', linewidth=1)

        # Plot all the means of the modalities
        plt.scatter(mu_weighted_I.cpu().numpy() @ d1.reshape(-1, 1), mu_weighted_I.cpu().numpy() @ d2.reshape(-1, 1), color=colorsys.hls_to_rgb(modality_color["image"][0], modality_color["image"][1] * 0.3, modality_color["image"][2]), marker='+', s=100, label='mu_I')
        plt.scatter(mu_weighted_T.cpu().numpy() @ d1.reshape(-1, 1), mu_weighted_T.cpu().numpy() @ d2.reshape(-1, 1), color=colorsys.hls_to_rgb(modality_color["text"][0], modality_color["text"][1] * 0.3, modality_color["text"][2]), marker='+', s=100, label='mu_T')
        plt.scatter(mu_weighted_B.cpu().numpy() @ d1.reshape(-1, 1), mu_weighted_B.cpu().numpy() @ d2.reshape(-1, 1), color=colorsys.hls_to_rgb(modality_color["bimodal"][0], modality_color["bimodal"][1] * 0.3, modality_color["bimodal"][2]), marker='+', s=100, label='mu_B')
        plt.scatter(mu_weighted_tout_court @ d1.reshape(-1, 1), mu_weighted_tout_court @ d2.reshape(-1, 1), color='k', marker='+', s=100, label='barycenter')
        plt.xlabel('Projection on mu_B - mu_I')
        plt.ylabel('Projection on mu_B - mu_T')
        plt.title('Projection of D on mu_B - mu_I and mu_B - mu_T')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        os.makedirs(f"./figures/{sae_name.replace('.pt', '')}/geometry", exist_ok=True)
        plt.savefig(f"./figures/{sae_name.replace('.pt', '')}/geometry/projection_mu_B.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"./figures/{sae_name.replace('.pt', '')}/geometry/projection_mu_B.pdf")
        plt.show()
    # _plot_proj_mu()

    # Baseline 1 : mu_weighted_{I,T,B}, bias : mu_weighted_tout_court

    def evaluate_custom_probe(weight, bias, trg_mask):
        probe = torch.nn.Linear(D.shape[1], 1, bias=True)
        probe.weight.data = weight
        probe.bias.data = torch.tensor(bias)
        probe.eval()

        with torch.no_grad():
            preds = (D - probe.bias.unsqueeze(0)) @ probe.weight
            
            # print(f"probe_I : {(probe_I > 0).float().mean().item():.4f}, probe_T : {(probe_T > 0).float().mean().item():.4f}, probe_B : {(probe_B > 0).float().mean().item():.4f}")
            acc = (((preds > 0) == trg_mask).float() * energy_per_concept.cpu()).sum() / energy_per_concept.sum()

        return acc.item()
    
    acc_I = evaluate_custom_probe(mu_weighted_I, mu_weighted_tout_court, I_mask)
    acc_T = evaluate_custom_probe(mu_weighted_T, mu_weighted_tout_court, T_mask)
    acc_B = evaluate_custom_probe(mu_weighted_B, mu_weighted_tout_court, B_mask)
    print("Probe 1 : mu_weighted_I", acc_I)
    print("Probe 2 : mu_weighted_T", acc_T)
    print("Probe 3 : mu_weighted_B", acc_B)

    # baseline 2 : fit scikit-learn linear regression for each region :

    def evaluate_probe_sklearn(P, N1, N2, P_mask, N1_mask, N2_mask):
        N = np.concatenate([N1, N2], axis=0)
        N_mask = (N1_mask | N2_mask).cpu().numpy()
        X = np.concatenate([P, N], axis=0)
        y = np.concatenate([np.ones(P.shape[0]), np.zeros(N.shape[0])], axis=0)
        
        # Shuffle
        perm = np.random.permutation(X.shape[0])
        X, y = X[perm], y[perm]
        energy_perm = energy_per_concept.cpu().numpy()[perm]
        P_mask = P_mask.cpu().numpy()[perm]
        N_mask = N_mask[perm]
        
        # Compute sample weights
        energy_perm[P_mask] = energy_perm[P_mask] / energy_perm[P_mask].sum() / 2
        energy_perm[N_mask] = energy_perm[N_mask] / energy_perm[N_mask].sum() / 2
        
        model = LinearRegression()
        model.fit(X, y, sample_weight=energy_perm)
        y_pred = model.predict(X)
        y_pred = (y_pred > 0.5).astype(int)
        y_pred = (y_pred == y).astype(int)
        return (y_pred * energy_perm).sum() / energy_perm.sum(), model
    
    acc_lr_B, lr_probe_B = evaluate_probe_sklearn(D[B_mask], D[I_mask], D[T_mask], B_mask, I_mask, T_mask)
    acc_lr_I, lr_probe_I = evaluate_probe_sklearn(D[I_mask], D[B_mask], D[T_mask], I_mask, B_mask, T_mask)
    acc_lr_T, lr_probe_T = evaluate_probe_sklearn(D[T_mask], D[I_mask], D[B_mask], T_mask, I_mask, B_mask)
    print("Probe 4 : Linear Regression B", acc_lr_B)
    print("Probe 5 : Linear Regression I", acc_lr_I)
    print("Probe 6 : Linear Regression T", acc_lr_T)
    
    # Plot projection on probe directions :
    lr_d_B = lr_probe_B.coef_.reshape(-1)
    d1 = lr_d_B / np.linalg.norm(lr_d_B)
    lr_d_I = lr_probe_I.coef_.reshape(-1)
    d2 = lr_d_I / np.linalg.norm(lr_d_I)
    lr_d_T = lr_probe_T.coef_.reshape(-1)
    d3 = lr_d_T / np.linalg.norm(lr_d_T)
    
    D_ = D.cpu().numpy()
    x1 = D_ @ d1.reshape(-1, 1)  # projection on I
    x2 = D_ @ d2.reshape(-1, 1)  # projection on T
    x3 = D_ @ d3.reshape(-1, 1)  # projection on B
    D_ = np.vstack([x1.squeeze(), x2.squeeze(), x3.squeeze()]).T
    
    def interpolate_colors(c1, c2, n, opacity=1, dotted=True):
        # Normalize RGB to [0, 1]
        rgb1 = np.array(c1)
        rgb2 = np.array(c2)

        lab1 = rgb2lab(rgb1.reshape(1, 1, 3))[0, 0]
        lab2 = rgb2lab(rgb2.reshape(1, 1, 3))[0, 0]

        t = np.linspace(0, 1, n)
        lab_interp = (1 - t)[:, None] * lab1 + t[:, None] * lab2
        rgb_interp = lab2rgb(lab_interp.reshape(n, 1, 3)).reshape(n, 3)

        # Clip and convert to 'rgb(r,g,b)' strings
        rgb_interp = np.clip(rgb_interp * 255, 0, 255).astype(int)
        res = []
        for i, (r, g, b) in enumerate(rgb_interp):
            if dotted and (i%4 == 1 or i%4==2):
                res.append(f'rgba({r},{g},{b}, {0})')
            else:
                res.append(f'rgba({r},{g},{b}, {opacity})')
        return res
    
    def mpl_to_plotly(cmap_name, pl_entries=255, luminance_scale=1.0, return_array=False):
        cmap = plt.cm.get_cmap(cmap_name, pl_entries)
        h = 1.0 / (pl_entries - 1)
        colorscale = []

        for k in range(pl_entries):
            r, g, b, _ = cmap(k)
            hls = colorsys.rgb_to_hls(r, g, b)
            l_scaled = max(0.0, min(1.0, hls[1] * luminance_scale))
            r_new, g_new, b_new = colorsys.hls_to_rgb(hls[0], l_scaled, hls[2])
            if not return_array:
                colorscale.append([
                    round(k * h, 4),
                    f'rgb({int(r_new * 255)}, {int(g_new * 255)}, {int(b_new * 255)})'
                ])
            else:
                colorscale.append([
                    int(r_new * 255), int(g_new * 255), int(b_new * 255)
                ])

        return np.array(colorscale) if return_array else colorscale
    
    def add_arrow(fig, origin, c1, name="", c2=None, width=6, end=None, direction=None):
        if end is None and direction is None:
            raise ValueError("One of end or direction must be provided.")
        if c2 is None:
            c2 = c1
        if width < 0.1:
            return
        # Normalize and scale arrow length
        if direction is not None:
            norm_dir = direction / np.linalg.norm(direction)
            length = np.linalg.norm(direction)
            end = origin + norm_dir * length
        # if color is a tuple of numpy arrays, convert it to a string RGB format
        if isinstance(c1, tuple) and all(isinstance(c, np.ndarray) for c in c1):
            c1 = f'rgb({int(c1[0] * 255)}, {int(c1[1] * 255)}, {int(c1[2] * 255)})'
        if isinstance(c2, tuple) and all(isinstance(c, np.ndarray) for c in c2):
            c2 = f'rgb({int(c2[0] * 255)}, {int(c2[1] * 255)}, {int(c2[2] * 255)})'
        fig.add_trace(go.Scatter3d(
            x=[origin[0], end[0]],
            y=[origin[1], end[1]],
            z=[origin[2], end[2]],
            mode='lines',
            name=name,
            line=dict(color=[c1, c2], width=width),
        ))
       
    def add_arrowhead(fig, start, end, color, sizeref=0.1, opacity=1):
        # Create a cone pointing from start to end
        direction = end - start
        norm = np.linalg.norm(direction)
        if norm == 0:
            return  # avoid division by zero
        direction_unit = direction / norm

        fig.add_trace(go.Cone(
            x=[end[0]],
            y=[end[1]],
            z=[end[2]],
            u=[direction_unit[0]],
            v=[direction_unit[1]],
            w=[direction_unit[2]],
            colorscale=[[0, color], [1, color]],
            showscale=False,
            sizemode="absolute",
            sizeref=sizeref,
            anchor="tip",
            opacity=opacity
        ))
       
    def add_curved_arrow(fig, origin, c1, name="", c2=None, width=6, end=None, direction=None, curve_strength=0.2, curve_axis=None, opacity=1):
        if end is None and direction is None:
            raise ValueError("One of end or direction must be provided.")
        if c2 is None:
            c2 = c1
        if width < 0.1:
            return

        if direction is not None:
            direction = np.asarray(direction)
            end = origin + direction

        origin = np.asarray(origin)
        end = np.asarray(end)
        mid = (origin + end) / 2

        # Default orthogonal axis if none provided
        base_vec = end - origin
        base_vec_norm = base_vec / np.linalg.norm(base_vec)

        if curve_axis is None:
            # Choose a consistent orthogonal vector
            if not np.allclose(base_vec_norm, [0, 0, 1]):
                ref = np.array([0, 0, 1])
            else:
                ref = np.array([0, 1, 0])
            curve_axis = np.cross(base_vec_norm, ref)
            curve_axis /= np.linalg.norm(curve_axis)
        
        # Control point for Bézier curve
        ctrl = mid + curve_strength * np.linalg.norm(base_vec) * curve_axis

        # Bézier curve sampling
        t = np.linspace(0, 1, 20)
        t2 = t + 1e-4
        t_ = []
        for i in range(len(t)):
            t_ += [t[i], t2[i]]
        t = np.array(t_)
        bez = (
            (1 - t)[:, None]**2 * origin +
            2 * (1 - t)[:, None] * t[:, None] * ctrl +
            t[:, None]**2 * end
        )

        colors = interpolate_colors(c1, c2, len(t), opacity=opacity)
        
        fig.add_trace(go.Scatter3d(
            x=bez[:, 0],
            y=bez[:, 1],
            z=bez[:, 2],
            mode='lines',
            name=name,
            line=dict(color=colors, width=width),
            showlegend=False
        ))
        
        add_arrowhead(fig, bez[2], bez[0], colors[0], sizeref=width*0.0045*opacity, opacity=opacity**2)
      
    managua_plotly = mpl_to_plotly('managua')
    dark_managua_plotly = mpl_to_plotly('managua', luminance_scale=0.5)

    class_score = modality_score.cpu().numpy()
    
    def _plot_3d():
        x_min, x_max = min(x1), max(x1)
        y_min, y_max = min(x2), max(x2)
        x=np.linspace(x_min, x_max, 20)
        y=np.linspace(y_min, y_max, 10)
        x_grid, y_grid = np.meshgrid(x, y)
        z_grid = np.full_like(x_grid, min(x3))
        
        fig = go.Figure(data=[go.Surface(
            x=x_grid,
            y=y_grid,
            z=z_grid,
            opacity=0.1,
            showscale=False,
            colorscale=[[0, 'black'], [1, 'black']],
            hoverinfo='skip'
        )])
        
        fig.add_trace(go.Scatter3d(
                x=x1.squeeze(), y=x2.squeeze(), z=np.zeros_like(x3.squeeze()) + min(x3),
                mode='markers',
                marker=dict(
                    size=np.power(size**2, 1/3) * 0.5,
                    color=class_score,
                    colorscale=dark_managua_plotly,  # or 'managua' if you have a custom Plotly scale
                    # colorbar=dict(title='Modality Score'),
                    opacity=0.1,
                    line=dict(width=0),
                    symbol='circle',
                )
            )
        )

        fig.add_trace(go.Scatter3d(
                x=x1.squeeze(), y=x2.squeeze(), z=x3.squeeze(),
                mode='markers',
                marker=dict(
                    size=np.power(size**2, 1/3) * 0.5,
                    color=class_score,
                    colorscale=managua_plotly,  # or 'managua' if you have a custom Plotly scale
                    colorbar=dict(title='Modality Score'),
                    opacity=0.8,
                    line=dict(width=0),
                )
            )
        )

        # Get mean vectors projected into the 3D space
        mu_B = (mu_weighted_B.cpu().numpy() @ np.stack([d1, d2, d3], axis=1)).flatten()
        mu_I = (mu_weighted_I.cpu().numpy() @ np.stack([d1, d2, d3], axis=1)).flatten()
        mu_T = (mu_weighted_T.cpu().numpy() @ np.stack([d1, d2, d3], axis=1)).flatten()

        # Add arrows from origin to means
        add_arrow(fig, origin=np.zeros(3), direction=mu_I, c1=colorsys.hls_to_rgb(modality_color["image"][0], modality_color["image"][1] * 0.3, modality_color["image"][2]), name='μ_I')
        add_arrow(fig, origin=np.zeros(3), direction=mu_T, c1=colorsys.hls_to_rgb(modality_color["text"][0], modality_color["text"][1] * 0.3, modality_color["text"][2]), name='μ_T')
        add_arrow(fig, origin=np.zeros(3), direction=mu_B, c1=colorsys.hls_to_rgb(modality_color["bimodal"][0], modality_color["bimodal"][1] * 0.3, modality_color["bimodal"][2]), name='μ_B')
        add_arrow(fig, origin=np.zeros(3), direction=mu_weighted_tout_court @ np.stack([d1, d2, d3], axis=1), c1='black', name='μ')

        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    backgroundcolor='rgba(0,0,0,0)',
                    gridcolor='rgba(0,0,0,0)',
                    zeroline=False,
                    showbackground=False
                ),
                yaxis=dict(
                    backgroundcolor='rgba(0,0,0,0)',
                    gridcolor='rgba(0,0,0,0)',
                    zeroline=False,
                    showbackground=False
                ),
                zaxis=dict(
                    backgroundcolor='rgba(0,0,0,0)',
                    gridcolor='rgba(0,0,0,0)',
                    zeroline=False,
                    showbackground=False
                ),
                xaxis_title=r'LR_B',
                yaxis_title=r'LR_I',
                zaxis_title=r'LR_T',
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            title="3D Projection on Linear Regression Directions",
            scene_camera=dict(
                eye=dict(x=1., y=2., z=0.6),
            )
        )
        # save html and pdf fig :
        fig.write_html(f"./figures/{sae_name.replace('.pt', '')}/geometry/projection_lr.html")
        fig.write_image(f"./figures/{sae_name.replace('.pt', '')}/geometry/projection_lr.pdf")
        fig.write_image(f"./figures/{sae_name.replace('.pt', '')}/geometry/projection_lr.png", scale=5)
        fig.show()
    _plot_3d()
    
    def _plot_OT_arrow():
        x_spread = max(x1.squeeze()) - min(x1.squeeze())
        shiftb = x_spread
        shifti = 0
        shiftt = 0
        
        x_min, x_max = min(x1), max(x1) + shiftb
        y_min, y_max = min(x2), max(x2) + shifti
        x=np.linspace(x_min, x_max, 20)
        y=np.linspace(y_min, y_max, 10)
        x_grid, y_grid = np.meshgrid(x, y)
        z_grid = np.full_like(x_grid, min(x3))
        
        fig = go.Figure(data=[go.Surface(
            x=x_grid,
            y=y_grid,
            z=z_grid,
            opacity=0.1,
            showscale=False,
            colorscale=[[0, 'black'], [1, 'black']],
            hoverinfo='skip'
        )])
        
        fig.add_trace(go.Scatter3d(
                x=x1.squeeze(), y=x2.squeeze(), z=np.zeros_like(x3.squeeze()) + min(x3),
                mode='markers',
                marker=dict(
                    size=np.power(size_I**2, 1/3) * 0.5,
                    color=class_score,
                    colorscale=dark_managua_plotly,  # or 'managua' if you have a custom Plotly scale
                    # colorbar=dict(title='Modality Score'),
                    opacity=0.1,
                    line=dict(width=0),
                    symbol='circle',
                )
            )
        )
        
        fig.add_trace(go.Scatter3d(
                x=x1.squeeze() + shiftb, y=x2.squeeze() + shifti, z=np.zeros_like(x3.squeeze()) + min(x3) + shiftt,
                mode='markers',
                marker=dict(
                    size=np.power(size_T**2, 1/3) * 0.5,
                    color=class_score,
                    colorscale=dark_managua_plotly,  # or 'managua' if you have a custom Plotly scale
                    # colorbar=dict(title='Modality Score'),
                    opacity=0.1,
                    line=dict(width=0),
                    symbol='circle',
                )
            )
        )
        
        fig.add_trace(go.Scatter3d(
                x=x1.squeeze(), y=x2.squeeze(), z=x3.squeeze(),
                mode='markers',
                marker=dict(
                    size=np.power(size_I**2, 1/3) * 0.5,
                    color=class_score,
                    colorscale=managua_plotly,  # or 'managua' if you have a custom Plotly scale
                    colorbar=dict(title='Modality Score'),
                    opacity=0.8,
                    line=dict(width=0),
                )
            ))
        
        fig.add_trace(go.Scatter3d(
            x=x1.squeeze() + shiftb, y=x2.squeeze() + shifti, z=x3.squeeze() + shiftt,
            mode='markers',
            marker=dict(
                size=np.power(size_T**2, 1/3) * 0.5,
                color=class_score,
                colorscale=managua_plotly,  # or 'managua' if you have a custom Plotly scale
                colorbar=dict(title='Modality Score'),
                opacity=0.8,
                line=dict(width=0),
            )
        )
    )
        
        # Get mean vectors projected into the 3D space
        mu_B_I = (mu_weighted_B.cpu().numpy() @ np.stack([d1, d2, d3], axis=1)).flatten()
        mu_I_I = (mu_weighted_I.cpu().numpy() @ np.stack([d1, d2, d3], axis=1)).flatten()
        mu_T_I = (mu_weighted_T.cpu().numpy() @ np.stack([d1, d2, d3], axis=1)).flatten()
        
        mu_B_T = mu_B_I + np.array([shiftb, shifti, shiftt])
        mu_I_T = mu_I_I + np.array([shiftb, shifti, shiftt])
        mu_T_T = mu_T_I + np.array([shiftb, shifti, shiftt])

        # Add arrows from origin to means
        colorI = colorsys.hls_to_rgb(modality_color["image"][0], modality_color["image"][1] * 0.5, modality_color["image"][2])
        colorT = colorsys.hls_to_rgb(modality_color["text"][0], modality_color["text"][1] * 0.5, modality_color["text"][2])
        colorB = colorsys.hls_to_rgb(modality_color["bimodal"][0], modality_color["bimodal"][1] * 0.8, modality_color["bimodal"][2])
        add_curved_arrow(fig, origin=mu_B_I, end=mu_B_T, c1=colorB, c2=colorB, width=24*bb, curve_axis=np.array([0, -1, -1]))
        add_curved_arrow(fig, origin=mu_B_I, end=mu_I_T, c1=colorB, c2=colorI, width=24*bi, curve_axis=np.array([0, 0.71, -0.71]))
        add_curved_arrow(fig, origin=mu_B_I, end=mu_T_T, c1=colorB, c2=colorT, width=24*bt, curve_axis=np.array([0, 0, 0]))
        add_curved_arrow(fig, origin=mu_I_I, end=mu_B_T, c1=colorI, c2=colorB, width=24*ib, curve_axis=np.array([0, -1, 0]))
        add_curved_arrow(fig, origin=mu_I_I, end=mu_I_T, c1=colorI, c2=colorI, width=24*ii, curve_axis=np.array([0, 0, 1]))
        add_curved_arrow(fig, origin=mu_I_I, end=mu_T_T, c1=colorI, c2=colorT, width=24*it, curve_axis=np.array([0, 0.5, 1]))
        add_curved_arrow(fig, origin=mu_T_I, end=mu_B_T, c1=colorT, c2=colorB, width=24*tb, curve_axis=np.array([0, 0, -1]))
        add_curved_arrow(fig, origin=mu_T_I, end=mu_I_T, c1=colorT, c2=colorI, width=24*ti, curve_axis=np.array([0, 0, 0]))
        add_curved_arrow(fig, origin=mu_T_I, end=mu_T_T, c1=colorT, c2=colorT, width=24*tt, curve_axis=np.array([0, 1, 0]))
        
        mu_B_I[2] = min(x3)
        mu_I_I[2] = min(x3)
        mu_T_I[2] = min(x3)
        mu_B_T[2] = min(x3)
        mu_I_T[2] = min(x3)
        mu_T_T[2] = min(x3)
        add_curved_arrow(fig, origin=mu_B_I, end=mu_B_T, c1=colorB, c2=colorB, width=15*bb, curve_axis=np.array([0, -1, 0]), opacity=0.8)
        add_curved_arrow(fig, origin=mu_B_I, end=mu_I_T, c1=colorB, c2=colorI, width=15*bi, curve_axis=np.array([0, 0.71, 0]), opacity=0.8)
        add_curved_arrow(fig, origin=mu_B_I, end=mu_T_T, c1=colorB, c2=colorT, width=15*bt, curve_axis=np.array([0, 0, 0]), opacity=0.8)
        add_curved_arrow(fig, origin=mu_I_I, end=mu_B_T, c1=colorI, c2=colorB, width=15*ib, curve_axis=np.array([0, -1, 0]), opacity=0.8)
        add_curved_arrow(fig, origin=mu_I_I, end=mu_I_T, c1=colorI, c2=colorI, width=15*ii, curve_axis=np.array([0, 0, 0]), opacity=0.8)
        add_curved_arrow(fig, origin=mu_I_I, end=mu_T_T, c1=colorI, c2=colorT, width=15*it, curve_axis=np.array([0, 0.5, 0]), opacity=0.8)
        add_curved_arrow(fig, origin=mu_T_I, end=mu_B_T, c1=colorT, c2=colorB, width=15*tb, curve_axis=np.array([0, 0, 0]), opacity=0.8)
        add_curved_arrow(fig, origin=mu_T_I, end=mu_I_T, c1=colorT, c2=colorI, width=15*ti, curve_axis=np.array([0, 0, 0]), opacity=0.8)
        add_curved_arrow(fig, origin=mu_T_I, end=mu_T_T, c1=colorT, c2=colorT, width=15*tt, curve_axis=np.array([0, 1, 0]), opacity=0.8)

        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    backgroundcolor='rgba(0,0,0,0)',
                    gridcolor='rgba(0,0,0,0)',
                    zeroline=False,
                    showbackground=False
                ),
                yaxis=dict(
                    backgroundcolor='rgba(0,0,0,0)',
                    gridcolor='rgba(0,0,0,0)',
                    zeroline=False,
                    showbackground=False
                ),
                zaxis=dict(
                    backgroundcolor='rgba(0,0,0,0)',
                    # gridcolor='rgba(1,1,1,1)',
                    # zeroline=False,
                    # showbackground=False
                ),
                xaxis_title=r'LR_B',
                yaxis_title=r'LR_I',
                zaxis_title=r'LR_T',
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            title="3D Projection on Linear Regression Directions",
            scene_camera=dict(
                eye=dict(x=x_spread/2, y=2., z=0.6),
            )
        )
        # save html and pdf fig :
        fig.write_html(f"./figures/{sae_name.replace('.pt', '')}/geometry/OT_arrow.html")
        fig.write_image(f"./figures/{sae_name.replace('.pt', '')}/geometry/OT_arrow.pdf")
        fig.write_image(f"./figures/{sae_name.replace('.pt', '')}/geometry/OT_arrow.png", scale=5)
        fig.show()
    # _plot_OT_arrow()
    
    def _plot_OT_Gamma(gamma):
        # Plot Gamma, entries sorted by modality score, and sized by total energy
        pass
    _plot_OT_Gamma(gamma)
    # _plot_OT_Gamma(gamma * DDT)
    
    def _plot_OT_interpol():
        # for t in linspace 0 10, plot 3D distrib of (1-t) * orig + t * end, where end = T(orig), T the barycentric projection.
        # color orig : untouched, color end : modality = barycenter of end points modalities, color = that
        # color at t : a bit darkened (luminance * 1 -> 0.5 -> 1) and linear interpol between c orig and c end.
        # Legend : left : Projection of D, points sized by E^(img), right : ... E^(txt),
        #   middle (make a arch, with smaller and smaller then larger and larger plots) : Distributional shift / transport
        # -> Feature's functional behavior is consistent with their geometric repartition, which is not the case in MP SAEs (see Figure appendix) 

        TX = (gamma @ D_) / (a[:, None] + 1e-8)
        Tmu = (gamma @ modality_score.cpu().numpy()) / (a + 1e-8)
        N = 8
        for t in torch.linspace(0, 1, N):
            t = t.item()
            Points = (1-t) * D_ + t * TX
            dimmed_managua = mpl_to_plotly(
                'managua',
                pl_entries=256,
                luminance_scale=max(t, 1-t),
                return_array=True,
            )
            color_X = rgb2lab(dimmed_managua[np.clip((modality_score.cpu().numpy() * 255).astype(int), 0, 255)] / 255)
            color_TX = rgb2lab(dimmed_managua[np.clip((Tmu * 255).astype(int), 0, 255)] / 255)
            colors = lab2rgb((1-t) * color_X + t * color_TX)
            hex_colors = [to_hex(c) for c in colors]
            
            dark_dimmed_managua = mpl_to_plotly(
                'managua',
                pl_entries=256,
                luminance_scale=max(t, 1-t) * 0.5,
                return_array=True,
            )
            dark_color_X = rgb2lab(dark_dimmed_managua[np.clip((modality_score.cpu().numpy() * 255).astype(int), 0, 255)] / 255)
            dark_color_TX = rgb2lab(dark_dimmed_managua[np.clip((Tmu * 255).astype(int), 0, 255)] / 255)
            dark_colors = lab2rgb((1-t) * dark_color_X + t * dark_color_TX)
            dark_hex_colors = [to_hex(c) for c in dark_colors]
            
            
            x_min, x_max = min(x1), max(x1)
            y_min, y_max = min(x2), max(x2)
            x=np.linspace(x_min, x_max, 20)
            y=np.linspace(y_min, y_max, 10)
            x_grid, y_grid = np.meshgrid(x, y)
            z_grid = np.full_like(x_grid, min(x3))
            
            fig = go.Figure(data=[go.Surface(
                x=x_grid,
                y=y_grid,
                z=z_grid,
                opacity=0.1,
                showscale=False,
                colorscale=[[0, 'black'], [1, 'black']],
                hoverinfo='skip'
            )])
        
            fig.add_trace(go.Scatter3d(
                    x=Points[:, 0], y=Points[:, 1], z=np.zeros_like(Points[:, 2]) + min(Points[:, 2]),
                    mode='markers',
                    marker=dict(
                        size=np.power(size_I**2, 1/3) * 0.5,
                        color=dark_hex_colors,
                        opacity=0.1,
                        line=dict(width=0),
                    )
                )
            )

            fig.add_trace(go.Scatter3d(
                    x=Points[:, 0], y=Points[:, 1], z=Points[:, 2],
                    mode='markers',
                    marker=dict(
                        size=np.power(size_I**2, 1/3) * 0.5,
                        color=hex_colors,
                        opacity=0.8,
                        line=dict(width=0),
                    )
                )
            )
            
            fig.update_layout(
                scene=dict(
                    xaxis=dict(
                        backgroundcolor='rgba(0,0,0,0)',
                        gridcolor='rgba(0,0,0,0)',
                        zeroline=False,
                        showbackground=False
                    ),
                    yaxis=dict(
                        backgroundcolor='rgba(0,0,0,0)',
                        gridcolor='rgba(0,0,0,0)',
                        zeroline=False,
                        showbackground=False
                    ),
                    zaxis=dict(
                        backgroundcolor='rgba(0,0,0,0)',
                        gridcolor='rgba(0,0,0,0)',
                        zeroline=False,
                        showbackground=False
                    ),
                    xaxis_title=r'LR_B',
                    yaxis_title=r'LR_I',
                    zaxis_title=r'LR_T',
                ),
                margin=dict(l=0, r=0, b=0, t=30),
                paper_bgcolor='rgba(0,0,0,0)',
                title="3D Projection on Linear Regression Directions",
                scene_camera=dict(
                    eye=dict(x=1., y=2., z=0.6),
                )
            )
            # save html and pdf fig :
            fig.write_html(f"./figures/{sae_name.replace('.pt', '')}/geometry/OT_interpol_{t:4f}.html")
            fig.write_image(f"./figures/{sae_name.replace('.pt', '')}/geometry/OT_interpol_{t:4f}.pdf")
            fig.write_image(f"./figures/{sae_name.replace('.pt', '')}/geometry/OT_interpol_{t:4f}.png", scale=5)
            fig.show()
    _plot_OT_interpol()
    
    # Now, consider each D_i as a probe and measure it's accuracy :
    accs = [None] * D.shape[0]
    
    def compute_accuracy(i):
        probe = D[i]
        if B_mask[i]:
            trg_mask = B_mask
        elif I_mask[i]:
            trg_mask = I_mask
        elif T_mask[i]:
            trg_mask = T_mask
        else:
            raise ValueError("Unknown modality")
        return evaluate_custom_probe(probe, mu_weighted_tout_court, trg_mask)
    
    async def run_in_executor(i, sem, pbar, loop):
        async with sem:
            acc = await loop.run_in_executor(None, compute_accuracy, i)
            accs[i] = acc
            pbar.update(1)
    
    async def main():
        sem = asyncio.Semaphore(4)
        loop = asyncio.get_event_loop()
        pbar = tqdm(total=D.shape[0], desc="Evaluating probes")
        tasks = [run_in_executor(i, sem, pbar, loop) for i in range(D.shape[0])]
        await asyncio.gather(*tasks)
        pbar.close()

    nest_asyncio.apply()
    asyncio.run(main())
    
    print(f"Mean acc : {np.mean(accs):.4f}, std : {np.std(accs):.4f}")
    e = energy_per_concept.cpu().numpy() / energy_per_concept.sum().item()
    print(f"Mean acc : {(np.array(accs) * e).sum():.4f}")

    def _plot_acc_hist():
        plt.figure(figsize=(10, 6))
        plt.hist(accs, bins=100, range=(0, 1), color='blue', label='Probe Accuracy', weights=energy_per_concept.cpu().numpy() / energy_per_concept.sum().item())
        plt.title("Distribution of Probe Accuracy")
        plt.xlabel("Probe Accuracy")
        plt.ylabel("density")
        #plt.yscale("log")
        plt.grid(axis='y', alpha=0.75)
        plt.legend()
        # Add mean and std information to the plot
        # plt.text(0.05, 0.95, f"Mean: {np.mean(accs):.4f}\nStd: {np.std(accs):.4f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
        plt.savefig(f"./figures/{sae_name.replace('.pt', '')}/geometry/probe_accuracy.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"./figures/{sae_name.replace('.pt', '')}/geometry/probe_accuracy.pdf")
        plt.show()

        # plot accuracy vs modality score :
        plt.figure(figsize=(10, 6))
        plt.hexbin(modality_score.cpu().numpy(), accs, gridsize=30, cmap='magma_r', bins='log', C=energy_per_concept.cpu().numpy(), reduce_C_function=np.sum, norm=LogNorm(vmin=max(1e-4, energy_per_concept.cpu().numpy().min())))
        plt.colorbar(label='density', extend='min')
        plt.title("Probe Accuracy vs Modality Score")
        plt.xlabel("Modality Score")
        plt.ylabel("Probe Accuracy")
        plt.grid(axis='y', alpha=0.75)
        plt.savefig(f"./figures/{sae_name.replace('.pt', '')}/geometry/probe_accuracy_vs_modality_score.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"./figures/{sae_name.replace('.pt', '')}/geometry/probe_accuracy_vs_modality_score.pdf")
        plt.show()
    _plot_acc_hist()

    sae.metrics._add_more({
        'linear_separability': {
            'acc_I': acc_I,
            'acc_T': acc_T,
            'acc_B': acc_B,
            'acc_lr_B': acc_lr_B,
            'acc_lr_I': acc_lr_I,
            'acc_lr_T': acc_lr_T,
            'feature_probe': (np.array(accs) * energy_per_concept.cpu().numpy()).sum() / energy_per_concept.sum(),
        }
    }, save=True)
    
    def _plot_probe_accuracy_bar_colored_by_modality():
        plt.figure(figsize=(10, 6))
        cmap = plt.get_cmap("managua")
        norm = plt.Normalize(vmin=0, vmax=1)
        bins = np.linspace(0, 1, 101)  # 30 bins
        bin_centers = 0.5 * (bins[:-1] + bins[1:])  # Bin centers for plotting
        digitized = np.digitize(accs, bins) - 1
        weights = energy_per_concept.cpu().numpy() / energy_per_concept.sum().item()
        plt.figure(figsize=(10, 6))
        for b in range(len(bins) - 1):
            b_idx = np.where(digitized == b)[0]
            if len(b_idx) == 0:
                continue
            sorted_idx = b_idx[np.argsort(modality_score[b_idx].cpu().numpy())]
            bottom = 0
            for i in sorted_idx:
                height = weights[i]
                color = cmap(norm(modality_score[i].cpu().numpy()))  # modality -> color
                plt.bar(bin_centers[b], height, width=(bins[1] - bins[0]), bottom=bottom, color=color, edgecolor='none')
                bottom += height
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])  # Dummy array for ScalarMappable
        cbar = plt.colorbar(sm, ax=plt.gca(), orientation='vertical', label='Modality Score') 
        plt.title("Distribution of Probe Accuracy")
        plt.xlabel("Probe Accuracy")
        plt.ylabel("density")
        plt.grid(axis='y', alpha=0.75)
        plt.savefig(f"./figures/{sae_name.replace('.pt', '')}/geometry/probe_accuracy_bar_colored_by_modality.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"./figures/{sae_name.replace('.pt', '')}/geometry/probe_accuracy_bar_colored_by_modality.pdf")
        plt.show()
    _plot_probe_accuracy_bar_colored_by_modality()
    plt.rcdefaults()

@torch.no_grad()
def data_classifiability(sae, I, T, sae_name, eps=0.05, skip_plot=False):
    plt.rcParams['font.size'] = 14
    I = I[torch.randperm(I.shape[0])[:int(1e4)]].cpu()
    T = T[torch.randperm(T.shape[0])[:int(1e4)]].cpu()
    D_ata = torch.cat([I, T], dim=0) # shape (N, 512)
    D_ict = sae.dictionary._fused_dictionary.cpu().detach() # shape (M, 512)
    modality_score = sae.metrics.mu
    energy_per_concept = sae.metrics.E
    print(D_ict.shape)
    I_mask = (modality_score > 1 - eps).cpu()
    T_mask = (modality_score < eps).cpu()
    B_mask = (modality_score <= 1 - eps).cpu() & (modality_score >= eps).cpu()
    print(f"Number of unimodal concepts: {I_mask.sum()}, {T_mask.sum()}, {B_mask.sum()}")

    mu_weighted_I = (D_ict[I_mask] * energy_per_concept[I_mask].unsqueeze(1).cpu()).sum(dim=0).cpu() / energy_per_concept[I_mask].sum().cpu()
    mu_weighted_T = (D_ict[T_mask] * energy_per_concept[T_mask].unsqueeze(1).cpu()).sum(dim=0).cpu() / energy_per_concept[T_mask].sum().cpu()
    mu_weighted_B = (D_ict[B_mask] * energy_per_concept[B_mask].unsqueeze(1).cpu()).sum(dim=0).cpu() / energy_per_concept[B_mask].sum().cpu()

    mu_weighted_tout_court = (D_ict * energy_per_concept.unsqueeze(1).cpu()).sum(dim=0).cpu().numpy() / energy_per_concept.sum().cpu().numpy()

    # Baseline 1 : mu_weighted_{I,T,B}, bias : mu_weighted_tout_court

    def evaluate_custom_probe(weight, bias, trg_mask):
        probe = torch.nn.Linear(D_ict.shape[1], 1, bias=True)
        probe.weight.data = weight
        probe.bias.data = torch.tensor(bias)
        probe.eval()

        with torch.no_grad():
            preds = (D_ata - probe.bias.unsqueeze(0)) @ probe.weight
            
            # print(f"probe_I : {(probe_I > 0).float().mean().item():.4f}, probe_T : {(probe_T > 0).float().mean().item():.4f}, probe_B : {(probe_B > 0).float().mean().item():.4f}")
            acc = ((preds > 0) == trg_mask).float().mean()

        return acc.item()
    
    Image_Data_Mask = torch.zeros(D_ata.shape[0], dtype=torch.bool)
    Image_Data_Mask[:I.shape[0]] = True
    Image_Data_Mask = Image_Data_Mask.cpu().numpy()
    Text_Data_Mask = torch.zeros(D_ata.shape[0], dtype=torch.bool)
    Text_Data_Mask[I.shape[0]:] = True
    Text_Data_Mask = Text_Data_Mask.cpu().numpy()
    acc_I = evaluate_custom_probe(mu_weighted_I, mu_weighted_tout_court, Image_Data_Mask) # Target accuracy : 1
    acc_T = evaluate_custom_probe(mu_weighted_T, mu_weighted_tout_court, Text_Data_Mask) # Target accuracy : 1
    acc_B = evaluate_custom_probe(mu_weighted_B, mu_weighted_tout_court, Image_Data_Mask) # Target accuracy : 0.5
    acc_B = max(acc_B, 1-acc_B)
    print("Probe 1 : mu_weighted_I", acc_I)
    print("Probe 2 : mu_weighted_T", acc_T)
    print("Probe 3 : mu_weighted_B", acc_B)

    # baseline 2 : fit scikit-learn linear regression for each region :

    def evaluate_probe_sklearn(P, N1, N2, P_mask, N1_mask, N2_mask):
        N = np.concatenate([N1, N2], axis=0)
        N_mask = (N1_mask | N2_mask).cpu().numpy()
        X = np.concatenate([P, N], axis=0)
        y = np.concatenate([np.ones(P.shape[0]), np.zeros(N.shape[0])], axis=0)
        
        # Shuffle
        perm = np.random.permutation(X.shape[0])
        X, y = X[perm], y[perm]
        energy_perm = energy_per_concept.cpu().numpy()[perm]
        P_mask = P_mask.cpu().numpy()[perm]
        N_mask = N_mask[perm]
        
        # Compute sample weights
        energy_perm[P_mask] = energy_perm[P_mask] / energy_perm[P_mask].sum() / 2
        energy_perm[N_mask] = energy_perm[N_mask] / energy_perm[N_mask].sum() / 2
        
        model = LinearRegression()
        model.fit(X, y, sample_weight=energy_perm)
        y_pred = model.predict(D_ata.cpu().numpy())
        y_pred = (y_pred > 0.5).astype(int)
        y_pred = (y_pred == Image_Data_Mask).astype(int)
        return y_pred.mean()
    
    acc_lr_B = evaluate_probe_sklearn(D_ict[B_mask], D_ict[I_mask], D_ict[T_mask], B_mask, I_mask, T_mask) # Target accuracy : 0.5
    acc_lr_B = max(acc_lr_B, 1-acc_lr_B)
    acc_lr_I = evaluate_probe_sklearn(D_ict[I_mask], D_ict[B_mask], D_ict[T_mask], I_mask, B_mask, T_mask) # Target accuracy : 1
    acc_lr_T = evaluate_probe_sklearn(D_ict[T_mask], D_ict[I_mask], D_ict[B_mask], T_mask, I_mask, B_mask) # Target accuracy : 0
    acc_lr_T = 1 - acc_lr_T # because we want to predict the text data
    print("Probe 4 : Linear Regression B", acc_lr_B)
    print("Probe 5 : Linear Regression I", acc_lr_I)
    print("Probe 6 : Linear Regression T", acc_lr_T)

    # Now, consider each D_i as a probe and measure it's accuracy :

    accs = []
    accs_trg = []
    for i in tqdm(range(D_ict.shape[0])):
        probe = D_ict[i]
        if B_mask[i]:
            trg_mask = Image_Data_Mask
        elif I_mask[i]:
            trg_mask = Image_Data_Mask
        elif T_mask[i]:
            trg_mask = Text_Data_Mask
        else:
            raise ValueError("Unknown modality")
        acc = evaluate_custom_probe(probe, mu_weighted_tout_court, trg_mask)
        if B_mask[i]:
            acc = max(acc, 1 - acc)
            accs.append(acc)
            accs_trg.append(1.5 - acc)
        else:
            accs.append(acc)
            accs_trg.append(acc)

    print(f"Mean unweighted acc : {np.mean(accs):.4f}, std : {np.std(accs):.4f}")
    print(f"Mean acc : {(np.array(accs_trg) * energy_per_concept.cpu().numpy()).sum() / energy_per_concept.sum():.4f}")

    if not skip_plot:
        plt.figure(figsize=(10, 6))
        plt.hist(accs, bins=100, range=(0, 1), color='blue', label='Probe Accuracy', weights=energy_per_concept.cpu().numpy() / energy_per_concept.sum().item())
        plt.title("Distribution of Probe Accuracy")
        plt.xlabel("Probe Accuracy")
        plt.ylabel("density")
        #plt.yscale("log")
        plt.grid(axis='y', alpha=0.75)
        plt.legend()
        # Add mean and std information to the plot
        # plt.text(0.05, 0.95, f"Mean: {np.mean(accs):.4f}\nStd: {np.std(accs):.4f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
        plt.savefig(f"./figures/{sae_name.replace('.pt', '')}/geometry/data_probe_accuracy.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"./figures/{sae_name.replace('.pt', '')}/geometry/data_probe_accuracy.pdf")
        plt.show()

        # plot accuracy vs modality score :
        plt.figure(figsize=(10, 6))
        plt.hexbin(modality_score.cpu().numpy(), accs, gridsize=30, cmap='magma_r', bins='log', C=energy_per_concept.cpu().numpy(), reduce_C_function=np.sum, norm=LogNorm(vmin=max(1e-4, energy_per_concept.cpu().numpy().min())))
        plt.colorbar(label='density', extend='min')
        plt.title("Probe Accuracy vs Modality Score")
        plt.xlabel("Modality Score")
        plt.ylabel("Probe Accuracy")
        plt.grid(axis='y', alpha=0.75)
        plt.savefig(f"./figures/{sae_name.replace('.pt', '')}/geometry/data_probe_accuracy_vs_modality_score.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"./figures/{sae_name.replace('.pt', '')}/geometry/data_probe_accuracy_vs_modality_score.pdf")
        plt.show()

        modality_perm = torch.argsort(modality_score).cpu()

        modality_sorted = modality_score[modality_perm]
        accs_sorted = np.array(accs)[modality_perm.cpu().numpy()]

        values = modality_sorted.cpu().numpy()
        weights = (energy_per_concept[modality_perm].cpu().numpy())
        weights = weights / weights.sum()

        bins = np.linspace(0, 1, 31)
        digitized = np.digitize(values, bins) - 1 # shape (N,), where N is the number of concepts. Each value is the index of the bin it belongs to.
        digitized = np.clip(digitized, 0, len(bins) - 2)

        # Weighted mean per bin
        numerator = np.bincount(digitized, weights=accs_sorted * weights, minlength=len(bins)-1)
        denominator = np.bincount(digitized, weights=weights, minlength=len(bins)-1)
        mean_per_bin = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
        for b in range(len(mean_per_bin)):
            b_idx = np.where(digitized == b)[0]
            if len(b_idx) == 0:
                mean_per_bin[b] = 0
            else:
                mean_per_bin[b] = np.sum(accs_sorted[b_idx] * weights[b_idx]) / np.sum(weights[b_idx]) if np.sum(weights[b_idx]) > 0 else 0

        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        plt.figure(figsize=(10, 6))
        plt.plot(bin_centers, mean_per_bin, color='blue')
        plt.xlabel("Modality Score")
        plt.ylabel("Weighted Mean per Bin")
        plt.title("Histogram of Modality Score Means (Weighted)")
        plt.grid(axis='y', alpha=0.75)
        plt.savefig(f"./figures/{sae_name.replace('.pt', '')}/geometry/data_probe_accuracy_vs_modality_score.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"./figures/{sae_name.replace('.pt', '')}/geometry/data_probe_accuracy_vs_modality_score.pdf")
        plt.show()

        plt.figure(figsize=(10, 6))
        cmap = plt.get_cmap("managua")
        norm = plt.Normalize(vmin=0, vmax=1)
        bins = np.linspace(0, 1, 101)  # 30 bins
        bin_centers = 0.5 * (bins[:-1] + bins[1:])  # Bin centers for plotting
        digitized = np.digitize(accs_sorted, bins) - 1
        plt.figure(figsize=(10, 6))
        for b in range(len(bins) - 1):
            b_idx = np.where(digitized == b)[0]
            if len(b_idx) == 0:
                continue
            # Sort i_j by energy
            sorted_idx = b_idx[np.argsort(values[b_idx])]
            bottom = 0
            for i in sorted_idx:
                height = weights[i]
                color = cmap(norm(values[i]))  # modality -> color
                plt.bar(bin_centers[b], height, width=(bins[1] - bins[0]), bottom=bottom, color=color, edgecolor='none')
                bottom += height
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])  # Dummy array for ScalarMappable
        cbar = plt.colorbar(sm, ax=plt.gca(), orientation='vertical', label='Modality Score') 
        plt.title("Distribution of Probe Accuracy")
        plt.xlabel("Probe Accuracy")
        plt.ylabel("density")
        plt.grid(axis='y', alpha=0.75)
        plt.ylim(0, 0.2)
        plt.savefig(f"./figures/{sae_name.replace('.pt', '')}/geometry/data_probe_accuracy_bar_colored_by_modality.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"./figures/{sae_name.replace('.pt', '')}/geometry/data_probe_accuracy_bar_colored_by_modality.pdf")
        plt.show()
        
    sae.metrics._add_more({
        'data_cassifiability': {
            'acc_I': acc_I,
            'acc_T': acc_T,
            'acc_B': acc_B,
            'acc_lr_B': acc_lr_B,
            'acc_lr_I': acc_lr_I,
            'acc_lr_T': acc_lr_T,
            'feature_probe': (np.array(accs_trg) * energy_per_concept.cpu().numpy()).sum() / energy_per_concept.sum(),
        }
    }, save=True)
    plt.rcdefaults()
