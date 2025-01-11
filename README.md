# Char-art
 Character art (ASCII art) image & video generator. Support literally ANY characters.

![](demo/bliss.png)

## A simple example

```python
from PIL import Image, ImageFont
import torch
import moviepy

from charart import Charart

UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LOWER = "abcdefghijklmnopqrstuvwxyz"
NUMBER = "0123456789"
OTHER = "!@#$%^&*()-_=+`~[]{}\\|/?;:'\",.<>"

font_size = 8
font = ImageFont.truetype(r'Arial.ttf', font_size)
charart = Charart(
    font_size,
    font,
    hspace=0, # specify the horizontal spaces between characters
    vspace=0, # specify the vertical spaces between characters
    hparts=2, # horizontal granularity for character matching
    vparts=2, # vertical granularity for character matching
    backend='pytorch',
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
)

charart.add_character(
    UPPER, LOWER, NUMBER, OTHER,
)

# Image to character art
in_image = Image.open('test_image.png')
out_image = charart.transform(in_image, set_color=True)
out_image.save('test_image_out.png')

# Video to character art
in_video = moviepy.VideoFileClip('test_video.mp4')
# To change the lightness of the video to avoid extremely dark or bright frames
norm_filter = charart.filter.video_norm_filter(in_video, outlier_ratio=0.01, skip_frames=30)
out_video = charart.video_transform(in_video, skip_frames=4, filter=norm_filter)
out_video.write_videofile('test_video_out.mp4')
```

