#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tokenizers import Tokenizer
import torch

tokenizer = Tokenizer.from_file("tokenizer_captions.json")

VOCAB_SIZE = tokenizer.get_vocab_size()

def tokenize(texts, context_length = 256, add_start = False, add_end = False, truncate_text = False):
    if isinstance(texts, str):
        texts = [texts]

    sot_tokens = tokenizer.encode("<|startoftext|>").ids if add_start else []
    eot_tokens = tokenizer.encode("<|endoftext|>").ids if add_end else []
    all_tokens = [sot_tokens + tokenizer.encode(text).ids + eot_tokens for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate_text:
                tokens = tokens[:context_length]
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result

get_ipython().system('wget --no-clobber <dropbox_url>')


# In[ ]:


get_ipython().system('wget --no-clobber https://www.dropbox.com/s/hl5hyzhyal3vfye/dalle_iconic_butterfly_149.pt')
get_ipython().run_line_magic('pip', 'install tokenizers')

get_ipython().run_line_magic('pip', 'install gpustat')
get_ipython().system('git clone https://github.com/lucidrains/DALLE-pytorch')
get_ipython().run_line_magic('cd', './DALLE-pytorch/')
get_ipython().system('python3 setup.py install')
get_ipython().system('sudo apt-get -y install llvm-9-dev cmake')
get_ipython().system('git clone https://github.com/microsoft/DeepSpeed.git /tmp/Deepspeed')
get_ipython().run_line_magic('cd', '/tmp/Deepspeed')
get_ipython().system('DS_BUILD_SPARSE_ATTN=1 ./install.sh -r')


# In[ ]:


checkpoint_path = "dalle_iconic_butterfly_149.pt"

import os
import glob
text = "an armchair imitating a pikachu. an armchair in the shape of a pikachu." #@param
get_ipython().system('python /content/DALLE-pytorch/generate.py --batch_size=32 --taming --dalle_path=$checkpoint_path --num_images=128 --text="$text"; wait;')
text_cleaned = text.replace(" ", "_")
_folder = f"/content/outputs/{text_cleaned}/"


# In[ ]:


import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')

text_cleaned = text.replace(" ", "_")
output_dir = f"/content/outputs/{text_cleaned}/" #@param
images = []
for img_path in glob.glob(f'{output_dir}*.jpg'):
    images.append(mpimg.imread(img_path))

plt.figure(figsize=(128,128))
columns = 5
for i, image in enumerate(images):
    plt.subplot(len(images) / columns + 1, columns, i + 1)
    plt.imshow(image)


# In[ ]:


get_ipython().run_line_magic('pip', 'install "git+https://github.com/openai/CLIP.git"')
import clip
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


# In[ ]:


""" Get rank by CLIP! """
image = F.interpolate(images, size=224)
text = clip.tokenize(["this colorful bird has a yellow breast , with a black crown and a black cheek patch."]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_text.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)


# In[ ]:


np_images = images.cpu().numpy()
scores = probs[0]

def show_reranking(images, scores, sort=True):
    img_shape = images.shape
    if sort:
        scores_sort = scores.argsort()
        scores = scores[scores_sort[::-1]]
        images = images[scores_sort[::-1]]

    rows = 4
    cols = img_shape[0] // 4
    img_idx = 0

    for col in range(cols):
        fig, axs = plt.subplots(1, rows, figsize=(20,20))
        plt.subplots_adjust(wspace=0.01)
        for row in range(rows):
            tran_img = np.transpose(images[img_idx], (1,2,0))
            axs[row].imshow(tran_img, interpolation='nearest')
            axs[row].set_title("{}%".format(np.around(scores[img_idx]*100, 5)))
            axs[row].set_xticks([])
            axs[row].set_yticks([])
            img_idx += 1

show_reranking(np_images, scores)


# In[ ]:


from torchvision import transforms

txt = "this bird has wings that are brown with a white belly"
img_path = "images/Yellow_Headed_Blackbird_0013_8362.jpg"

img = Image.open(img_path)
tf = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
    transforms.RandomResizedCrop(256, scale=(0.6, 1.0), ratio=(1.0, 1.0)),
    transforms.ToTensor(),
])
img = tf(img).cuda()

sot_token = vocab.encode("<|startoftext|>").ids[0]
eot_token = vocab.encode("<|endoftext|>").ids[0]
codes = [0] * dalle_dict['hparams']['text_seq_len']
text_token = vocab.encode(txt).ids
tokens = [sot_token] + text_token + [eot_token]
codes[:len(tokens)] = tokens
caption_token = torch.LongTensor(codes).cuda()

imgs = img.repeat(16,1,1,1)
caps = caption_token.repeat(16,1)

mask = (caps != 0).cuda()

images = dalle.generate_images(
        caps,
        mask = mask,
        img = imgs,
        num_init_img_tokens = (100),  # you can set the size of the initial crop, defaults to a little less than ~1/2 of the tokens, as done in the paper
        filter_thres = 0.9,
        temperature = 1.0
)

grid = make_grid(images, nrow=4, normalize=False, range=(-1, 1)).cpu()
show(grid)


# In[ ]:


# import wandb
# run = wandb.init()
# artifact = run.use_artifact('afiaka87/dalle_train_transformer/trained-dalle:v14', type='model')
# artifact_dir = artifact.download()

