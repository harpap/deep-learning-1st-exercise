# DeepLabv3+ explained from beginner level to full depth

DeepLabv3+ is a **deep learning architecture for semantic image segmentation**.

Semantic segmentation means: given an image, predict a class label for **every pixel**.

For example, in the Oxford-IIIT Pet dataset, the input image might be:

> a photo of a cat or dog

The output is not just “cat” or “dog.” Instead, the model predicts:

> this pixel is pet, this pixel is background, this pixel is border, etc.

So the model produces a **segmentation mask**.

---

# 1. The basic problem DeepLabv3+ solves

In image classification, a CNN answers:

> “What is in this image?”

Example:

```text
Input image -> CNN -> "dog"
```

In semantic segmentation, the model answers:

> “What class does each pixel belong to?”

Example:

```text
Input image -> segmentation model -> mask
```

So instead of one label per image, we need one label per pixel.

For an image of size:

```text
H × W × 3
```

the model outputs something like:

```text
H × W × C
```

where `C` is the number of classes.

For example, if you have 3 segmentation classes:

```text
background
pet
border
```

then each pixel gets 3 scores, one per class.

---

# 2. Why normal CNNs are not enough

A normal CNN usually does this:

```text
image -> convolutions -> pooling/striding -> smaller feature maps -> classification
```

CNNs repeatedly reduce spatial resolution.

Example:

```text
Input:        256 × 256
After layer:  128 × 128
After layer:   64 × 64
After layer:   32 × 32
After layer:   16 × 16
```

This is useful for classification because the model only needs to know **what** is in the image.

But segmentation needs to know **where** things are.

If the model compresses the image too much, it loses fine details such as:

```text
object boundaries
thin structures
fur edges
legs
ears
small objects
```

So segmentation models need two things:

1. **High-level semantic understanding**
   The model must know what object it is looking at.

2. **Precise spatial detail**
   The model must know exactly where object boundaries are.

DeepLabv3+ is designed to solve this tradeoff.

---

# 3. Big idea of DeepLabv3+

DeepLabv3+ combines three major ideas:

```text
1. A CNN backbone to extract visual features
2. Atrous/dilated convolutions to see a larger area without losing resolution
3. An encoder-decoder structure to recover sharp segmentation boundaries
```

At a high level:

```text
Input image
   ↓
CNN backbone
   ↓
DeepLabv3 encoder with ASPP
   ↓
Decoder
   ↓
Pixel-wise segmentation mask
```

More visually:

```text
Image
  ↓
Backbone CNN
  ↓
Atrous Spatial Pyramid Pooling
  ↓
Upsampling + low-level feature fusion
  ↓
Final segmentation mask
```

---

# 4. Key concept: feature maps

A CNN does not directly “see” objects the way humans do.

It transforms the image into many **feature maps**.

Early CNN layers detect simple patterns:

```text
edges
corners
colors
textures
```

Middle layers detect object parts:

```text
eyes
fur
legs
wheels
windows
```

Deep layers detect semantic concepts:

```text
dog
cat
person
car
road
sky
```

So deeper layers understand meaning better, but they usually have lower resolution.

Example:

```text
Input image:       256 × 256 × 3
Early features:    128 × 128 × 64
Middle features:    64 × 64 × 128
Deep features:      32 × 32 × 256
Very deep features: 16 × 16 × 512
```

The deep feature maps are semantically powerful but spatially coarse.

DeepLabv3+ tries to preserve and recover spatial precision.

---

# 5. Key concept: receptive field

The **receptive field** of a neuron is the part of the input image it can “see.”

A small convolution, such as a `3 × 3` convolution, sees a small local region.

```text
3 × 3 convolution sees nearby pixels
```

But segmentation often requires wider context.

For example, to label a pixel correctly, the model may need to know:

```text
Is this fur part of a dog?
Is this gray pixel road or wall?
Is this edge part of the object or background?
```

So the model needs both:

```text
local detail
global context
```

One way to increase receptive field is to use pooling or strided convolution.

But that reduces resolution.

DeepLab uses a better trick: **atrous convolution**, also called **dilated convolution**.

---

# 6. Key concept: atrous/dilated convolution

A normal `3 × 3` convolution looks like this:

```text
x x x
x x x
x x x
```

It samples adjacent pixels.

An atrous convolution inserts spaces between kernel elements.

With dilation rate 2:

```text
x . x . x
. . . . .
x . x . x
. . . . .
x . x . x
```

With dilation rate 3:

```text
x . . x . . x
. . . . . . .
. . . . . . .
x . . x . . x
. . . . . . .
. . . . . . .
x . . x . . x
```

The kernel still has only 9 learned weights, like a normal `3 × 3` convolution, but it covers a larger area.

So atrous convolution gives the model a larger receptive field **without reducing the feature map resolution**.

That is one of the most important ideas in DeepLab.

---

# 7. Why dilation is useful for segmentation

Suppose the model is labeling this pixel:

```text
one pixel on the body of a dog
```

Looking only at nearby pixels may not be enough. The local texture could look like:

```text
fur
grass
carpet
shadow
```

But if the model sees a larger area, it can understand:

```text
This region belongs to a dog shape.
```

Atrous convolution helps the model use broader context while keeping spatial resolution.

This is critical for segmentation.

---

# 8. Key concept: encoder-decoder architecture

Many segmentation models use an **encoder-decoder** structure.

The encoder compresses the image into abstract features.

The decoder expands those features back into a full-resolution segmentation mask.

```text
Encoder:
image -> smaller, deeper, semantic feature maps

Decoder:
semantic feature maps -> full-size pixel mask
```

Example:

```text
Input image:        256 × 256
Encoder output:      16 × 16
Decoder output:     256 × 256
```

The encoder answers:

> “What is in the image?”

The decoder answers:

> “Where exactly is it?”

DeepLabv3+ is an encoder-decoder model.

---

# 9. What was DeepLabv3 before DeepLabv3+?

DeepLabv3 used:

```text
CNN backbone + atrous convolution + ASPP
```

The main strength of DeepLabv3 was capturing multi-scale context.

But its output boundaries could still be coarse because the final feature maps had reduced resolution.

DeepLabv3+ added a stronger decoder.

That `+` is important.

DeepLabv3+ means:

```text
DeepLabv3 encoder + decoder module
```

The decoder helps recover sharper object boundaries.

---

# 10. DeepLabv3+ architecture overview

The architecture has two main parts:

```text
1. Encoder
2. Decoder
```

The encoder contains:

```text
Backbone CNN
Atrous convolution
ASPP module
```

The decoder contains:

```text
Upsampling
Low-level feature fusion
Several convolutions
Final upsampling
```

Full flow:

```text
Input image
   ↓
Backbone CNN
   ↓
High-level feature map
   ↓
ASPP module
   ↓
Upsample
   ↓
Concatenate with low-level features from earlier backbone layer
   ↓
Convolutions
   ↓
Upsample to original image size
   ↓
Pixel-wise class prediction
```

---

# 11. The backbone CNN

DeepLabv3+ does not usually start from scratch.

It uses a strong CNN backbone such as:

```text
ResNet
Xception
MobileNet
```

The backbone extracts feature maps from the image.

For example, with ResNet:

```text
Input image
   ↓
Conv layers
   ↓
Residual blocks
   ↓
Deep feature maps
```

The backbone produces both:

```text
low-level features
high-level features
```

Low-level features come from early layers.

They contain:

```text
edges
textures
fine spatial details
```

High-level features come from deep layers.

They contain:

```text
object-level meaning
semantic context
```

DeepLabv3+ uses both.

---

# 12. Output stride

A key parameter in DeepLabv3+ is **output stride**.

Output stride means:

```text
input image resolution / final encoder feature resolution
```

For example, if the input is:

```text
256 × 256
```

and the encoder output is:

```text
16 × 16
```

then:

```text
output stride = 256 / 16 = 16
```

Common output strides are:

```text
16
8
```

An output stride of 16 means the feature map is 16 times smaller than the input.

An output stride of 8 means it is only 8 times smaller.

```text
Output stride 16:
256 × 256 -> 16 × 16

Output stride 8:
256 × 256 -> 32 × 32
```

Output stride 8 gives better spatial detail but uses more memory and computation.

Output stride 16 is cheaper and often still works well.

---

# 13. Atrous convolution inside the backbone

Normally, deep CNNs reduce resolution using stride.

DeepLab modifies the backbone.

Instead of downsampling too much, it replaces some strided convolutions with atrous convolutions.

So instead of:

```text
reduce resolution aggressively
```

DeepLab does:

```text
keep higher resolution
increase dilation rate
preserve receptive field
```

This means the model can still see a large context without making the feature maps too small.

---

# 14. ASPP: Atrous Spatial Pyramid Pooling

ASPP is the core module of DeepLabv3 and DeepLabv3+.

ASPP stands for:

```text
Atrous Spatial Pyramid Pooling
```

Its purpose is to capture information at multiple scales.

Why multiple scales?

Because objects appear in different sizes.

In pet segmentation, a pet can be:

```text
large and close to the camera
small and far away
long and thin
curled up
partially occluded
```

A single convolution scale may not be enough.

ASPP applies several parallel operations to the same feature map.

Simplified:

```text
Input feature map
   ├── 1 × 1 convolution
   ├── 3 × 3 atrous convolution, small dilation
   ├── 3 × 3 atrous convolution, medium dilation
   ├── 3 × 3 atrous convolution, large dilation
   └── image-level pooling
```

Then it concatenates the results.

Diagram:

```text
                 ┌─ 1×1 conv
                 │
Feature map ─────┼─ 3×3 atrous conv, rate r1
                 │
                 ├─ 3×3 atrous conv, rate r2
                 │
                 ├─ 3×3 atrous conv, rate r3
                 │
                 └─ global image pooling
                         ↓
                 concatenate
                         ↓
                   1×1 convolution
```

Each branch sees the image differently.

Small dilation captures local detail.

Large dilation captures wider context.

Global pooling captures whole-image context.

---

# 15. Why ASPP matters

Imagine a model trying to segment a dog.

One branch may focus on local fur texture.

Another branch may capture the whole body shape.

Another branch may understand the global context of the image.

ASPP combines all of these.

So instead of using one fixed receptive field, DeepLabv3+ uses several receptive fields in parallel.

This helps with objects of different sizes.

---

# 16. Image-level pooling branch

The ASPP module often includes an image-level pooling branch.

This branch performs global average pooling over the feature map.

It compresses the entire feature map into a global context vector.

Conceptually:

```text
Look at the whole image and summarize what is present.
```

Then this global feature is upsampled and combined with the other ASPP branches.

This helps the model understand global scene context.

For example:

```text
If the image contains a pet, ambiguous pixels near the object are more likely pet-related.
```

---

# 17. The decoder in DeepLabv3+

After ASPP, the feature map is still lower resolution than the input image.

A naive approach would be:

```text
ASPP output -> upsample directly to image size
```

But this often gives blurry boundaries.

DeepLabv3+ improves this by using a decoder.

The decoder does this:

```text
1. Take high-level ASPP features
2. Upsample them
3. Take low-level features from an earlier backbone layer
4. Concatenate them
5. Apply convolutions
6. Upsample to full image size
```

Diagram:

```text
Input image
   ↓
Backbone CNN
   ├──────────── low-level features ───────────┐
   ↓                                           │
High-level features                            │
   ↓                                           │
ASPP                                           │
   ↓                                           │
Upsample                                      │
   ↓                                           │
Concatenate ◄─────────────────────────────────┘
   ↓
3×3 convolutions
   ↓
Upsample
   ↓
Segmentation mask
```

---

# 18. Why low-level features are used

High-level features know **what** something is.

Low-level features know **where edges and details are**.

For segmentation, we need both.

Example:

```text
High-level feature:
"This region is probably a dog."

Low-level feature:
"The dog boundary is here."
```

So the decoder combines them.

This is similar in spirit to U-Net skip connections, but DeepLabv3+ uses a lighter decoder.

---

# 19. Full DeepLabv3+ pipeline

Let’s walk through one image.

Suppose the input image is:

```text
256 × 256 × 3
```

Step 1: Backbone extracts features.

```text
Input image
   ↓
CNN backbone
   ↓
Low-level feature map: 64 × 64 × channels
High-level feature map: 16 × 16 × channels
```

Step 2: High-level features go into ASPP.

```text
16 × 16 high-level feature map
   ↓
ASPP with multiple dilation rates
   ↓
16 × 16 multi-scale semantic feature map
```

Step 3: ASPP output is upsampled.

```text
16 × 16 -> 64 × 64
```

Step 4: Low-level features are processed.

Usually a `1 × 1` convolution reduces their channel count.

Why?

Because early feature maps can have many channels, and if we concatenate too many low-level features, they may dominate the high-level semantic features.

So DeepLabv3+ does:

```text
low-level features -> 1 × 1 conv -> fewer channels
```

Step 5: Concatenate.

```text
Upsampled ASPP features: 64 × 64
Low-level features:      64 × 64

Concatenate along channels
```

Step 6: Apply a few `3 × 3` convolutions.

These refine the combined representation.

Step 7: Upsample to original image size.

```text
64 × 64 -> 256 × 256
```

Step 8: Final classifier predicts class scores per pixel.

```text
256 × 256 × C
```

Each pixel gets one predicted class.

---

# 20. Final prediction: logits, softmax, and mask

The model outputs raw scores called **logits**.

For each pixel, the model gives one score per class.

Example with 3 classes:

```text
Pixel at position (x, y):

background score = 0.2
pet score        = 3.7
border score     = 1.1
```

The highest score is `pet`, so that pixel is predicted as pet.

Usually, during training, we apply softmax:

```text
softmax(logits) -> class probabilities
```

Example:

```text
background: 0.02
pet:        0.91
border:     0.07
```

Final prediction:

```text
pet
```

For binary segmentation, the model may output either:

```text
1 channel + sigmoid
```

or:

```text
2 channels + softmax
```

For multi-class segmentation, it usually outputs:

```text
C channels + softmax
```

---

# 21. DeepLabv3+ in one compact diagram

```text
                  ┌─────────────────────────────┐
                  │        Input image           │
                  └──────────────┬──────────────┘
                                 │
                                 ▼
                  ┌─────────────────────────────┐
                  │        CNN backbone          │
                  │  ResNet / Xception / etc.    │
                  └───────┬─────────────────────┘
                          │
          low-level       │ high-level
          features        │ features
              │           ▼
              │    ┌──────────────────────┐
              │    │        ASPP           │
              │    │ multi-scale context   │
              │    └──────────┬───────────┘
              │               │
              │               ▼
              │          Upsampling
              │               │
              ▼               ▼
        1×1 conv       ASPP features
              │               │
              └───────┬───────┘
                      ▼
              Concatenation
                      ▼
               3×3 convolutions
                      ▼
                 Upsampling
                      ▼
              Segmentation mask
```

---

# 22. Why DeepLabv3+ works well

DeepLabv3+ works well because it balances three competing needs.

## 1. Semantic understanding

The deep CNN backbone learns high-level object information.

It can recognize meaningful structures like:

```text
cat body
dog face
background
road
sky
person
```

## 2. Large context

Atrous convolution and ASPP allow the model to see large regions of the image.

This helps it classify ambiguous pixels.

## 3. Sharp boundaries

The decoder uses low-level features to recover spatial precision.

This improves object edges.

So DeepLabv3+ is strong because it combines:

```text
deep semantic features
multi-scale context
spatial detail recovery
```

---

# 23. Comparison with U-Net

U-Net and DeepLabv3+ are both segmentation architectures, but they emphasize different ideas.

U-Net:

```text
encoder-decoder
many skip connections
strong boundary recovery
commonly used in medical imaging
simple and effective
```

DeepLabv3+:

```text
encoder-decoder
atrous convolutions
ASPP multi-scale context
stronger global context modeling
often used for natural images
```

A simple comparison:

| Feature                    |                   U-Net |                      DeepLabv3+ |
| -------------------------- | ----------------------: | ------------------------------: |
| Encoder-decoder            |                     Yes |                             Yes |
| Skip connections           |                    Many | Usually one main low-level skip |
| Atrous convolution         |             Not central |                         Central |
| Multi-scale context        | Limited unless modified |              Strong ASPP module |
| Boundary detail            |                  Strong |                          Strong |
| Natural image segmentation |                    Good |                       Very good |
| Beginner implementation    |                  Easier |                    More complex |

For your pet segmentation project, both are valid.

U-Net is usually easier to implement from scratch.

DeepLabv3+ is more advanced and usually gives stronger results when implemented correctly.

---

# 24. DeepLabv3+ for Oxford-IIIT Pet segmentation

The Oxford-IIIT Pet dataset usually provides segmentation masks with labels like:

```text
1 = pet
2 = border
3 = background
```

Depending on your project, you can formulate the task as:

## Option A: Binary segmentation

Classes:

```text
pet
background
```

You merge the border into either pet or background.

This is easier.

Output:

```text
H × W × 1
```

with sigmoid, or:

```text
H × W × 2
```

with softmax.

## Option B: Three-class segmentation

Classes:

```text
pet
border
background
```

This is harder but closer to the original dataset.

Output:

```text
H × W × 3
```

with softmax.

For a beginner project, I would usually recommend binary segmentation first.

---

# 25. Loss functions used with DeepLabv3+

DeepLabv3+ itself is an architecture. It does not force one specific loss function.

Common losses are:

```text
cross-entropy loss
Dice loss
cross-entropy + Dice loss
focal loss
```

For multi-class segmentation, cross-entropy is common.

For binary pet/background segmentation, binary cross-entropy or Dice-based losses are common.

A common practical choice is:

```text
cross-entropy + soft Dice loss
```

Cross-entropy helps classify each pixel correctly.

Dice loss helps improve overlap between predicted mask and ground-truth mask.

---

# 26. More detailed internal structure

A typical DeepLabv3+ model contains:

```text
Backbone:
    CNN feature extractor

Encoder:
    atrous convolutions
    ASPP

Decoder:
    low-level feature projection
    concatenation
    refinement convolutions

Classifier:
    final 1×1 convolution
```

More concretely:

```text
Input image
   ↓
Backbone block 1
   ↓
Low-level features saved
   ↓
Backbone block 2
   ↓
Backbone block 3
   ↓
Backbone block 4 with atrous convolution
   ↓
High-level features
   ↓
ASPP
   ↓
Decoder
   ↓
Final 1×1 conv
   ↓
Upsampled logits
```

The final `1 × 1` convolution maps feature channels to class channels.

For example:

```text
256 feature channels -> 3 class channels
```

That means each pixel gets 3 logits.

---

# 27. What the `1 × 1` convolution does

A `1 × 1` convolution does not look at neighboring pixels.

It mixes channel information at each spatial location.

Example:

```text
At each pixel location:
[feature1, feature2, feature3, ..., feature256]
      ↓
[class_score_background, class_score_pet, class_score_border]
```

So the final classifier is often a `1 × 1` convolution.

It converts features into class scores.

---

# 28. What the `3 × 3` convolutions in the decoder do

After concatenating ASPP features and low-level features, the decoder applies `3 × 3` convolutions.

These help blend:

```text
semantic information from ASPP
spatial detail from low-level features
```

They refine the segmentation map before the final upsampling.

Without these convolutions, the combination would be too crude.

---

# 29. Why not just upsample directly?

Suppose the ASPP output is:

```text
16 × 16
```

and the original image is:

```text
256 × 256
```

Directly resizing from `16 × 16` to `256 × 256` is a big jump.

The model may produce masks that are:

```text
blurry
blocky
bad around edges
poor on thin regions
```

The decoder improves this by first upsampling to an intermediate resolution and combining with earlier features.

That gives the model access to details before producing the final mask.

---

# 30. Training DeepLabv3+

During training, the model receives:

```text
input image
ground-truth mask
```

The model predicts:

```text
predicted mask
```

Then the loss function compares the prediction with the ground truth.

Training loop:

```text
1. Feed image into DeepLabv3+
2. Get predicted pixel logits
3. Compare logits with ground-truth mask
4. Compute loss
5. Backpropagate
6. Update model weights
```

The goal is to minimize pixel-wise prediction error.

---

# 31. Inference with DeepLabv3+

During inference:

```text
1. Input image goes through model
2. Model outputs logits
3. Apply softmax or sigmoid
4. Choose class per pixel
5. Produce segmentation mask
```

For multi-class segmentation:

```python
pred_mask = argmax(logits, dim=class_channel)
```

For binary segmentation:

```python
pred_mask = sigmoid(logits) > threshold
```

Usually the threshold is `0.5`, but it can be tuned.

---

# 32. The main strengths of DeepLabv3+

DeepLabv3+ is strong because:

```text
It captures multi-scale context.
It preserves more spatial resolution using atrous convolution.
It recovers boundaries using a decoder.
It works well with powerful pretrained backbones.
It is effective on natural image segmentation tasks.
```

For pet segmentation, it can produce good masks around animals, especially when using a pretrained backbone.

---

# 33. The main weaknesses of DeepLabv3+

DeepLabv3+ is more complex than simple models.

Possible disadvantages:

```text
heavier than U-Net
more difficult to implement from scratch
more hyperparameters
ASPP dilation rates need care
can be memory-intensive
may need pretrained backbone for best results
```

For a course project, this matters.

If your project requirement says “only CNN,” DeepLabv3+ is still a CNN-based segmentation architecture. It uses convolutions, atrous convolutions, and encoder-decoder CNN components.

---

# 34. Most important terms to remember

| Term                       | Meaning                                          |
| -------------------------- | ------------------------------------------------ |
| Semantic segmentation      | Classify every pixel                             |
| Backbone                   | CNN feature extractor                            |
| Encoder                    | Extracts compressed semantic features            |
| Decoder                    | Recovers spatial resolution                      |
| Atrous/dilated convolution | Convolution with gaps to enlarge receptive field |
| Receptive field            | Region of input image seen by a feature          |
| ASPP                       | Multi-scale atrous convolution module            |
| Output stride              | Downsampling ratio of encoder output             |
| Low-level features         | Early CNN features with spatial detail           |
| High-level features        | Deep CNN features with semantic meaning          |
| Logits                     | Raw model outputs before softmax/sigmoid         |

---

# 35. DeepLabv3+ in one sentence

DeepLabv3+ is a CNN-based semantic segmentation model that uses **atrous convolutions** and **ASPP** to capture large multi-scale context, then uses a **decoder with low-level feature fusion** to recover sharper pixel-level boundaries.

---

# 36. Simplified mental model

Think of DeepLabv3+ like this:

```text
Backbone:
    Understands the image.

ASPP:
    Looks at the image at multiple scales.

Decoder:
    Restores object boundaries.

Final classifier:
    Assigns a class to every pixel.
```

For pet segmentation:

```text
Backbone:
    detects animal-like features

ASPP:
    understands the whole pet shape and surrounding context

Decoder:
    refines ears, legs, fur boundaries, and outline

Classifier:
    decides pet/background/border for every pixel
```

That is the essence of DeepLabv3+.
