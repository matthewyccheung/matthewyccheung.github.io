---
layout: post
title: "Handling large images in nnUNet Coarse to Fine"
math: true
---

nnU-Net handles very large images/volumes through:

1. Patch based inference: full-res model sees crops and inference uses overlapping windows and blends them with gaussian weighting.

``` python
# patch X: [H, W]
# Y = model(X): [H, W]
# full image X_full = [FH, FW]
FH, FW = X_full.shape
stride, H, W = 2, 100, 100
# initialize
Yhat_new = torch.zeros_like(X_full)
weights = torch.zeros_like(X_full)
# make gaussian 
gaussian = make_gaussian2d((H, W), sigma_scale=1/8)

# run sliding window inference

# compute indices for iteration
def starts(length, patch, step_fraction=0.5):
    # calculate number of patches
    # length - patch: last valid starting coordinate
    # patch*step fraction: target step
    # ceil(length-patch)/(patch*step fraction): number of patches in length-patch
    # add 1 for what's left in length
    n = math.ceil((length-patch)/(patch*step_fraction))+1
    return np.round(np.linspace(0, length-patch, n)).astype(int)

# assume FH>=H and FW>=W
# pad first otherwise
for c in starts(FH, H):
    for r in starts(FW, W):
        # run inference on patch
        logits = model(X_full[c:c+H, r:r+W])
        # multiply by gaussian
        logits *= gaussian
        # add gaussian weighted logits to new image
        Yhat_new[c:c+H, r:r+W] += logits
        # store weights
        weights[c:c+H, r:r+W] += gaussian
# weighted mean = sum(w*X)/sum(w)
Yhat_new /= weights
```

2. Low-res to high-res cascade: when full-res patch covers too little of the volume, first run a low resolution U-Net (with large image) and then feed its segmentation into a second full-res U-Net for refinement.

``` python
# X_low: full low res image [H_low, W_low]
# X_high: cropped high res image [H_high, W_high]
# X: full res image [H, W]
patch_h, patch_w
Yhat_low = low_res_model(X_low) # logits: [C, H_low, W_low]
Chat_low = argmax(Chat_low, 0)
# convert to one hot encoding
Chat_low_one_hot = one_hot(Yhat_low) # [C, H_low, W_low]
# interpolate to low res to full image size
Chat_low_one_hot_interp = interpolate(Chat_low_one_hot, size=(H, W), method='nearest')
# crop full image size to patch
Chat_low_one_hot_interp[patch_h:patch_h+H_high, patch_w:patch_w+W_high]
# stack
X_concat = stack([Yhat_low_one_hot_interp, X_high]) # [C+1, H_low, W_low]
Yhat_high = high_res_model(X_concat) # [H_high, W_high]
```

#### Improving inference speed and memory usage

To make this more GPU friendly, one can process patches in chunks: include every global patch that overlaps with the chunk, including patches starting outside of it. Trade-off would be processing images twice, but this can be resolved with caching and other coodinate bookkeeping.