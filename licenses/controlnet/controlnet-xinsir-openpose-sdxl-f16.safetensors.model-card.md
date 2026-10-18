---
license: apache-2.0
tags:
- openpose
- controlnet
- diffusers
- controlnet-openpose-sdxl-1.0
- text_to_image
pipeline_tag: text-to-image
---
# ***State of the art ControlNet-openpose-sdxl-1.0 model, below are the result for midjourney and anime, just for show***
![images](./masonry_real.webp)
![images](./masonry0.webp)


### controlnet-openpose-sdxl-1.0

<!-- Provide a longer summary of what this model is. -->

- **Developed by:** xinsir
- **Model type:** ControlNet_SDXL
- **License:** apache-2.0
- **Finetuned from model [optional]:** stabilityai/stable-diffusion-xl-base-1.0

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Paper [optional]:** https://arxiv.org/abs/2302.05543
-

### Examples
![images10](./000010_scribble_concat.webp)
![images20](./000024_scribble_concat.webp)
![images30](./000028_scribble_concat.webp)
![images40](./000030_scribble_concat.webp)
![images50](./000044_scribble_concat.webp)
![images60](./000101_scribble_concat.webp)
![images70](./000127_scribble_concat.webp)
![images80](./000128_scribble_concat.webp)
![images90](./000155_scribble_concat.webp)
![images99](./000180_scribble_concat.webp)

![images0](./000001_scribble_concat.webp)
![images1](./000003_scribble_concat.webp)
![images2](./000005_scribble_concat.webp)
![images3](./000008_scribble_concat.webp)
![images4](./000015_scribble_concat.webp)
![images5](./000031_scribble_concat.webp)
![images6](./000042_scribble_concat.webp)
![images7](./000047_scribble_concat.webp)
![images8](./000048_scribble_concat.webp)
![images9](./000083_scribble_concat.webp)

## Replace the default draw pose function to get better result
thanks feiyuuu for report the problem. When using the default pose line the performance may be unstable, this is because the pose label use more thick line in training to have a better look.
This difference can be fix by using the following method:

Find the util.py in controlnet_aux python package, usually the path is like: /your anaconda3 path/envs/your env name/lib/python3.8/site-packages/controlnet_aux/open_pose/util.py
Replace the draw_bodypose function with the following code:
```python
def draw_bodypose(canvas: np.ndarray, keypoints: List[Keypoint]) -> np.ndarray:
    """
    Draw keypoints and limbs representing body pose on a given canvas.

    Args:
        canvas (np.ndarray): A 3D numpy array representing the canvas (image) on which to draw the body pose.
        keypoints (List[Keypoint]): A list of Keypoint objects representing the body keypoints to be drawn.

    Returns:
        np.ndarray: A 3D numpy array representing the modified canvas with the drawn body pose.

    Note:
        The function expects the x and y coordinates of the keypoints to be normalized between 0 and 1.
    """
    H, W, C = canvas.shape


    if max(W, H) < 500:
        ratio = 1.0
    elif max(W, H) >= 500 and max(W, H) < 1000:
        ratio = 2.0
    elif max(W, H) >= 1000 and max(W, H) < 2000:
        ratio = 3.0
    elif max(W, H) >= 2000 and max(W, H) < 3000:
        ratio = 4.0
    elif max(W, H) >= 3000 and max(W, H) < 4000:
        ratio = 5.0
    elif max(W, H) >= 4000 and max(W, H) < 5000:
        ratio = 6.0
    else:
        ratio = 7.0

    stickwidth = 4

    limbSeq = [
        [2, 3], [2, 6], [3, 4], [4, 5],
        [6, 7], [7, 8], [2, 9], [9, 10],
        [10, 11], [2, 12], [12, 13], [13, 14],
        [2, 1], [1, 15], [15, 17], [1, 16],
        [16, 18],
    ]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    for (k1_index, k2_index), color in zip(limbSeq, colors):
        keypoint1 = keypoints[k1_index - 1]
        keypoint2 = keypoints[k2_index - 1]

        if keypoint1 is None or keypoint2 is None:
            continue

        Y = np.array([keypoint1.x, keypoint2.x]) * float(W)
        X = np.array([keypoint1.y, keypoint2.y]) * float(H)
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), int(stickwidth * ratio)), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, [int(float(c) * 0.6) for c in color])

    for keypoint, color in zip(keypoints, colors):
        if keypoint is None:
            continue

        x, y = keypoint.x, keypoint.y
        x = int(x * W)
        y = int(y * H)
        cv2.circle(canvas, (int(x), int(y)), int(4 * ratio), color, thickness=-1)

    return canvas
```

## How to Get Started with the Model

Use the code below to get started with the model.

```python
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers import DDIMScheduler, EulerAncestralDiscreteScheduler
from controlnet_aux import OpenposeDetector
from PIL import Image
import torch
import numpy as np
import cv2



controlnet_conditioning_scale = 1.0
prompt = "your prompt, the longer the better, you can describe it as detail as possible"
negative_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'



eulera_scheduler = EulerAncestralDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")


controlnet = ControlNetModel.from_pretrained(
    "xinsir/controlnet-openpose-sdxl-1.0",
    torch_dtype=torch.float16
)

# when test with other base model, you need to change the vae also.
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)


pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    safety_checker=None,
    torch_dtype=torch.float16,
    scheduler=eulera_scheduler,
)

processor = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')


controlnet_img = cv2.imread("your image path")
controlnet_img = processor(controlnet_img, hand_and_face=False, output_type='cv2')


# need to resize the image resolution to 1024 * 1024 or same bucket resolution to get the best performance
height, width, _  = controlnet_img.shape
ratio = np.sqrt(1024. * 1024. / (width * height))
new_width, new_height = int(width * ratio), int(height * ratio)
controlnet_img = cv2.resize(controlnet_img, (new_width, new_height))
controlnet_img = Image.fromarray(controlnet_img)

images = pipe(
    prompt,
    negative_prompt=negative_prompt,
    image=controlnet_img,
    controlnet_conditioning_scale=controlnet_conditioning_scale,
    width=new_width,
    height=new_height,
    num_inference_steps=30,
    ).images

images[0].save(f"your image save path, png format is usually better than jpg or webp in terms of image quality but got much bigger")
```


## Evaluation Data
HumanArt [https://github.com/IDEA-Research/HumanArt], select 2000 images with ground truth pose annotations to generate images and calculate mAP.



## Quantitative Result
| metric | xinsir/controlnet-openpose-sdxl-1.0 |  lllyasviel/control_v11p_sd15_openpose | thibaud/controlnet-openpose-sdxl-1.0 |
|-------|-------|-------|-------|
| mAP | **0.357** | 0.326 | 0.209 |

We are the SOTA openpose model compared with other opensource models.
