# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for reproducing the figures of the StyleGAN paper using pre-trained generators."""

import os
import pickle
import numpy as np
import PIL.Image
from PIL import ImageDraw
import dnnlib
import dnnlib.tflib as tflib
import config

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--load', default='', help='load model pkl')
#parser.add_argument('--nsample', type=int, default=64, help='number of samples')
parser.add_argument('--nlabel', type=int, default=0, help='number of labels')
#parser.add_argument('--label', type=int, default=0, help='index of labels')
#parser.add_argument('--resolution', type=int, default=256, help='resolution, must be 2**n')
args = parser.parse_args()

synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)

_Gs_cache = dict()

def load_Gs(f):
#    if url not in _Gs_cache:
#        with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
#            _G, _D, Gs = pickle.load(f)
    _G, _D, Gs = pickle.load(open(f, 'rb'))
    return Gs
#    _Gs_cache[url] = Gs
#    return _Gs_cache[url]

def draw_with_label(png, Gs, w, h, seeds, nlabel):
    print(png)
    txt=('As is', '180C 25 hrs', '180C 200 hrs', '180C 2300 hrs', '300C 25 hrs', '300C 200 hrs', '300C 2300 hrs', '400C 25 hrs', '400C 200 hrs', '400C 2300 hrs')
    nZ=Gs.input_shape[1]
    nseed = len(seeds)
    latents = np.stack(np.random.RandomState(seed).randn(nZ) for seed in seeds)
    latents = np.repeat(latents[None,:,:], nlabel, axis=0).reshape((-1, nZ))
    labels = np.repeat(np.eye(nlabel)[:,None,:], nseed, axis=1).reshape((-1, nlabel))
    dlatents = Gs.components.mapping.run(latents, labels)
    images = Gs.components.synthesis.run(dlatents, randomize_noise=False, **synthesis_kwargs)

    canvas = PIL.Image.new('RGB', (w * (nseed+1), h * nlabel), 'white')
    d = ImageDraw. Draw(canvas)
    for row in range(nlabel):
        d.text((10,row*h), txt[row], fill=(255,0,0))
        for col in range(nseed):
            canvas.paste(PIL.Image.fromarray(images[row*nseed+col], 'RGB'), ((col+1) * w, (row ) * h))
    canvas.save(png)

#----------------------------------------------------------------------------
# Figure 4: Noise detail.

def draw_noise_detail_figure(png, Gs, w, h, num_samples, seeds):
    print(png)
    canvas = PIL.Image.new('RGB', (w * 3, h * len(seeds)), 'white')
    for row, seed in enumerate(seeds):
        latents = np.stack([np.random.RandomState(seed).randn(Gs.input_shape[1])] * num_samples)
        images = Gs.run(latents, None, truncation_psi=1, **synthesis_kwargs)
        canvas.paste(PIL.Image.fromarray(images[0], 'RGB'), (0, row * h))
        for i in range(4):
            crop = PIL.Image.fromarray(images[i + 1], 'RGB')
            crop = crop.crop((650, 180, 906, 436))
            crop = crop.resize((w//2, h//2), PIL.Image.NEAREST)
            canvas.paste(crop, (w + (i%2) * w//2, row * h + (i//2) * h//2))
        diff = np.std(np.mean(images, axis=3), axis=0) * 4
        diff = np.clip(diff + 0.5, 0, 255).astype(np.uint8)
        canvas.paste(PIL.Image.fromarray(diff, 'L'), (w * 2, row * h))
    canvas.save(png)

#----------------------------------------------------------------------------
# Figure 5: Noise components.

def draw_noise_components_figure(png, Gs, w, h, seeds, noise_ranges, flips):
    print(png)
    Gsc = Gs.clone()
    noise_vars = [var for name, var in Gsc.components.synthesis.vars.items() if name.startswith('noise')]
    noise_pairs = list(zip(noise_vars, tflib.run(noise_vars))) # [(var, val), ...]
    latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in seeds)
    all_images = []
    for noise_range in noise_ranges:
        tflib.set_vars({var: val * (1 if i in noise_range else 0) for i, (var, val) in enumerate(noise_pairs)})
        range_images = Gsc.run(latents, None, truncation_psi=1, randomize_noise=False, **synthesis_kwargs)
        range_images[flips, :, :] = range_images[flips, :, ::-1]
        all_images.append(list(range_images))

    canvas = PIL.Image.new('RGB', (w * 2, h * 2), 'white')
    for col, col_images in enumerate(zip(*all_images)):
        canvas.paste(PIL.Image.fromarray(col_images[0], 'RGB').crop((0, 0, w//2, h)), (col * w, 0))
        canvas.paste(PIL.Image.fromarray(col_images[1], 'RGB').crop((w//2, 0, w, h)), (col * w + w//2, 0))
        canvas.paste(PIL.Image.fromarray(col_images[2], 'RGB').crop((0, 0, w//2, h)), (col * w, h))
        canvas.paste(PIL.Image.fromarray(col_images[3], 'RGB').crop((w//2, 0, w, h)), (col * w + w//2, h))
    canvas.save(png)

#----------------------------------------------------------------------------
# Figure 8: Truncation trick.

def draw_truncation_trick_figure(png, Gs, w, h, seeds, psis):
    print(png)
    latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in seeds)
    dlatents = Gs.components.mapping.run(latents, None) # [seed, layer, component]
    dlatent_avg = Gs.get_var('dlatent_avg') # [component]

    canvas = PIL.Image.new('RGB', (w * len(psis), h * len(seeds)), 'white')
    for row, dlatent in enumerate(list(dlatents)):
        row_dlatents = (dlatent[np.newaxis] - dlatent_avg) * np.reshape(psis, [-1, 1, 1]) + dlatent_avg
        row_images = Gs.components.synthesis.run(row_dlatents, randomize_noise=False, **synthesis_kwargs)
        for col, image in enumerate(list(row_images)):
            canvas.paste(PIL.Image.fromarray(image, 'RGB'), (col * w, row * h))
    canvas.save(png)

#----------------------------------------------------------------------------
# Main program.

def main():
    tflib.init_tf()
    os.makedirs(config.result_dir, exist_ok=True)
    net = load_Gs(args.load)
    draw_with_label(os.path.join(config.result_dir, 'draw_with_label.png'), net, w=108, h=108, seeds=np.random.randint(1,99999,size=9), nlabel=args.nlabel)


#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
