import dlib
from imutils import face_utils
import imutils
from typing import Union
import cv2
import numpy as np

predictor_path = "shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()  # type: ignore
predictor = dlib.shape_predictor(predictor_path)  # type: ignore


def _detect_bounding_rect(img, face_part):
    """Given the image and the face part, this function returns the bounding rectangle of the face part.

    Parameters
    ----------
    img : np.ndarray
        the image to detect the face
    face_part : str
        the face part to detect the bounding rectangle. It can be any of FACIAL_LANDMARKS_68_IDXS, and also "eyes", in which case the bounding rectangle of both eyes will be returned.

    Returns
    -------
    tuple[int, int, int, int]
        starting x, starting y, width, and height of the bounding rectangle
    """
    dets = detector(img, 1)

    assert (
        len(dets) == 1
    ), f"The image should have exactly one face, but got {len(dets)}"

    shape = face_utils.shape_to_np(predictor(img, dets[0]))

    if face_part == "eyes":
        rect_points = [
            shape[i:j]
            for name, (i, j) in face_utils.FACIAL_LANDMARKS_68_IDXS.items()
            if name == "left_eye" or name == "right_eye"
        ]
    else:
        rect_points = [
            shape[i:j]
            for name, (i, j) in face_utils.FACIAL_LANDMARKS_68_IDXS.items()
            if name == face_part
        ]
    rect_points = np.concatenate(rect_points)
    (x1, y1, w1, h1) = cv2.boundingRect(np.array([rect_points]))
    return x1, y1, w1, h1


def _insert_sunglasses(img, bounding_rect):
    """Inserts clipart, given the image ndarray and the bounding rectangle where it should be inserted."""
    x1, y1, w1, h1 = bounding_rect
    sunglasses = cv2.imread("sunglasses.png", cv2.IMREAD_UNCHANGED)
    sunglasses_width = int(np.floor(w1 * 1.2))

    sunglasses = imutils.resize(sunglasses, width=sunglasses_width)

    y1, y2 = y1 - int(sunglasses.shape[0] / 2.4), y1 + sunglasses.shape[0] - int(
        sunglasses.shape[0] / 2.4
    )
    x1, x2 = x1 - int(sunglasses.shape[1] * 0.1), x1 + sunglasses.shape[1] - int(
        sunglasses.shape[1] * 0.1
    )

    alpha_multichannel_broadcasted = np.dstack([sunglasses[:, :, 3]] * 3) / 255

    img[y1:y2, x1:x2, :3] = (1 - alpha_multichannel_broadcasted) * img[
        y1:y2, x1:x2, :3
    ] + alpha_multichannel_broadcasted * sunglasses[:, :, :3]

    return img


def _crop_clipart(clipart_array, image_array, x1, y1, x2, y2):
    """Crops clipart if it goes outside the image boundaries. It uses bounding box and image shape to determine the crop."""
    y1 = np.clip(y1, 0, image_array.shape[0])
    y2 = np.clip(y2, 0, image_array.shape[0])
    x1 = np.clip(x1, 0, image_array.shape[1])
    x2 = np.clip(x2, 0, image_array.shape[1])
    clipart_array = clipart_array[: y2 - y1, : x2 - x1]

    assert (
        clipart_array.shape[0] > 0 and clipart_array.shape[1] > 0
    ), f"It seems that the clipart to be inserted will be entirely otside of an image"

    return clipart_array


def _insert_lips(img, bounding_rect):
    """Inserts clipart, given the image ndarray and the bounding rectangle where it should be inserted."""
    x1, y1, w1, _ = bounding_rect

    lips = cv2.imread("lips.png", cv2.IMREAD_UNCHANGED)
    lips_width = int(np.floor(w1 * 1.2))

    lips = imutils.resize(lips, width=lips_width)
    lips = cv2.cvtColor(lips, cv2.COLOR_BGRA2RGBA)

    y1, y2 = y1 - (lips.shape[0] // 3), y1 + lips.shape[0] - (lips.shape[0] // 3)
    x1, x2 = (
        x1 - int(lips.shape[1] * 0.1),
        x1 + lips.shape[1] - int(lips.shape[1] * 0.1),
    )
    lips = _crop_clipart(lips, img, x1, y1, x2, y2)

    alpha_multichannel_broadcasted = np.dstack([lips[:, :, 3]] * 3) / 255

    img[y1:y2, x1:x2, :3] = (1 - alpha_multichannel_broadcasted) * img[
        y1:y2, x1:x2, :3
    ] + alpha_multichannel_broadcasted * lips[:, :, :3]

    return img


def _insert_flowers(img):
    # 1. get image shape
    # 2. load flowers image (path: "flowers.png")
    # 3. resize the flowers image to cover the entire image
    # 4. insert the flowers image to the image
    # 5. return the image

    # 1. get image shape
    img_height, img_width, _ = img.shape

    # 2. load flowers image (path: "flowers.png")
    flowers = cv2.imread("flowers.png", cv2.IMREAD_UNCHANGED)
    flowers = cv2.cvtColor(flowers, cv2.COLOR_BGRA2RGBA)

    # 3. resize the flowers image to cover the entire image
    flowers = cv2.resize(flowers, (img_width, img_height))

    # 4. insert the flowers image to the image
    alpha_multichannel_broadcasted = np.dstack([flowers[:, :, 3]] * 3) / 255

    img[:, :, :3] = (1 - alpha_multichannel_broadcasted) * img[
        :, :, :3
    ] + alpha_multichannel_broadcasted * flowers[:, :, :3]

    return img


def add_sunglasses(face_image_path: Union[str, np.ndarray]) -> np.ndarray:
    """Add sunglasses to the image, given the image path or the image ndarray."""
    if isinstance(face_image_path, str):
        img = dlib.load_rgb_image(face_image_path)  # type: ignore
    else:
        img = face_image_path
        assert (
            img.shape[2] == 3
        ), "The image is required to have 3 channels. but it has {}".format(
            img.shape[2]
        )

    x1, y1, w1, h1 = _detect_bounding_rect(img, "eyes")

    img = _insert_sunglasses(img, (x1, y1, w1, h1))

    return img


def add_lips(face_image_path: Union[str, np.ndarray]) -> np.ndarray:
    """Add lips to the image, given the image path or the image ndarray."""
    if isinstance(face_image_path, str):
        img = dlib.load_rgb_image(face_image_path)  # type: ignore
    else:
        img = face_image_path
        assert (
            img.shape[2] == 3
        ), "The image is required to have 3 channels. but it has {}".format(
            img.shape[2]
        )

    x1, y1, w1, h1 = _detect_bounding_rect(img, "mouth")

    img = _insert_lips(img, (x1, y1, w1, h1))

    return img


def add_lips_and_sunglasses(face_image_path: Union[str, np.ndarray]) -> np.ndarray:
    """Add lips and sunglasses to the image, given the image path or the image ndarray."""
    if isinstance(face_image_path, str):
        img = dlib.load_rgb_image(face_image_path)  # type: ignore
    else:
        img = face_image_path
        assert (
            img.shape[2] == 3
        ), "The image is required to have 3 channels. but it has {}".format(
            img.shape[2]
        )

    eyes = _detect_bounding_rect(img, "eyes")
    mouth = _detect_bounding_rect(img, "mouth")

    img = _insert_lips(img, mouth)
    img = _insert_sunglasses(img, eyes)

    return img


def add_flowers(face_image_path: Union[str, np.ndarray]) -> np.ndarray:
    """Add flowers to the image, given the image path or the image ndarray."""
    if isinstance(face_image_path, str):
        img = dlib.load_rgb_image(face_image_path)  # type: ignore
    else:
        img = face_image_path
        assert (
            img.shape[2] == 3
        ), "The image is required to have 3 channels. but it has {}".format(
            img.shape[2]
        )

    img = _insert_flowers(img)

    return img
