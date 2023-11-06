import os
from typing import Optional

import cv2
from PIL import Image


def _create_gif_from_frames(frames, output_filename, duration):
    frames[0].save(
        f"{output_filename}.gif",
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=duration,
    )


def img2gif(
    images_path: str,
    output_filename: Optional[str] = "tmp",
    duration: Optional[int] = 500,
) -> None:
    """Directory 내 이미지들을 GIF로 변환

    .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/280735315-2eaa059e-0cf5-4cb7-8983-5b7e1f2d3f21.gif
        :alt: GIF Result
        :align: center

    Args:
        images_path (``str``): GIF로 변환할 이미지들이 존재하는 경로
        output_filename (``Optional[str]``): 출력될 GIF 파일명
        duration: (``Optional[int]``): ms 단위의 사진 간 간격

    Returns:
        ``None``: 현재 directory에 바로 GIF 저장

    Examples:
        >>> import zerohertzLib as zz
        >>> zz.vision.img2gif("./")
    """
    image_files = [
        f for f in os.listdir(images_path) if f.endswith((".png", ".jpg", ".jpeg"))
    ]
    image_files.sort()
    images = [
        Image.open(os.path.join(images_path, image_file)) for image_file in image_files
    ]
    _create_gif_from_frames(images, output_filename, duration)


def vid2gif(
    video_path: str,
    output_filename: Optional[str] = "tmp",
    quality: Optional[int] = 100,
    fps: Optional[int] = 15,
):
    """동영상을 GIF로 변환

    .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/280735327-ba590c0b-6180-4dce-a256-ccf12d0ba64b.gif
        :alt: GIF Result
        :align: center

    Args:
        video_path (``str``): GIF로 변환할 동영상이 존재하는 경로
        output_filename (``Optional[str]``): 출력될 GIF 파일명
        quality: (``Optional[int]``): 출력될 GIF의 품질
        fps: (``Optional[int]``): 출력될 GIF의 FPS (Frames Per Second)

    Returns:
        ``None``: 현재 directory에 바로 GIF 저장

    Examples:
        >>> import zerohertzLib as zz
        >>> zz.vision.vid2gif("tmp.mp4")
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    original_fps = round(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % (original_fps // fps) == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            width, height = pil_img.size
            new_width = int(width * quality / 100)
            new_height = int(height * quality / 100)
            resized_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
            frames.append(resized_img)
        frame_count += 1
    cap.release()
    duration = 1000 // fps
    _create_gif_from_frames(frames, output_filename, duration)
