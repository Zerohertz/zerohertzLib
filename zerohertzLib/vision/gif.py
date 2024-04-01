"""
MIT License

Copyright (c) 2023 Hyogeun Oh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
from typing import List, Optional

import cv2
from PIL import Image


def _create_gif_from_frames(
    frames: List[Image.Image], file_name: str, duration: int
) -> None:
    frames[0].save(
        f"{file_name}.gif",
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=duration,
    )


def img2gif(
    path: str,
    file_name: Optional[str] = "tmp",
    duration: Optional[int] = 500,
) -> None:
    """Directory 내 image들을 GIF로 변환

    Args:
        path (``str``): GIF로 변환할 image들이 존재하는 경로
        file_name (``Optional[str]``): 출력될 GIF file 이름
        duration (``Optional[int]``): ms 단위의 사진 간 간격

    Returns:
        ``None``: 현재 directory에 바로 GIF 저장

    Examples:
        >>> zz.vision.img2gif("./")

        .. image:: _static/examples/static/vision.img2gif.gif
            :align: center
            :width: 200px
    """
    ext = (
        "jpg",
        "JPG",
        "jpeg",
        "JPEG",
        "png",
        "PNG",
        "tif",
        "TIF",
        "tiff",
        "TIFF",
    )
    image_files = [f for f in os.listdir(path) if f.endswith(ext)]
    image_files.sort()
    images = [Image.open(os.path.join(path, image_file)) for image_file in image_files]
    _create_gif_from_frames(images, file_name, duration)


def vid2gif(
    path: str,
    file_name: Optional[str] = "tmp",
    quality: Optional[int] = 100,
    fps: Optional[int] = 15,
    speed: Optional[float] = 1.0,
) -> None:
    """동영상을 GIF로 변환

    Args:
        path (``str``): GIF로 변환할 동영상이 존재하는 경로
        file_name (``Optional[str]``): 출력될 GIF file 이름
        quality (``Optional[int]``): 출력될 GIF의 품질
        fps (``Optional[int]``): 출력될 GIF의 FPS (Frames Per Second)
        speed (``Optional[float]``): 출력될 GIF의 배속

    Returns:
        ``None``: 현재 directory에 바로 GIF 저장

    Examples:
        >>> zz.vision.vid2gif("test.mp4")

        .. image:: _static/examples/static/vision.vid2gif.gif
            :align: center
            :width: 300px
    """
    frames = []
    cap = cv2.VideoCapture(path)
    original_fps = round(cap.get(cv2.CAP_PROP_FPS))
    fps = min(original_fps, fps)
    frame_count_speed = frame_count_fps = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count_speed += 1
        if round(frame_count_speed % speed) != 0:
            continue
        if frame_count_fps % (int(original_fps / fps)) == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            width, height = pil_img.size
            new_width = int(width * quality / 100)
            new_height = int(height * quality / 100)
            resized_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
            frames.append(resized_img)
        frame_count_fps += 1
    cap.release()
    duration = int(1000 / fps)
    _create_gif_from_frames(frames, file_name, duration)
