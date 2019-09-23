import logging
import os.path

import requests

logger = logging.getLogger(__name__)

ignore_label = 255
id2label = {0: ignore_label,
            1: 10,
            2: 2,
            3: 0,
            4: 1,
            5: 4,
            6: 8,
            7: 5,
            8: 13,
            9: 7,
            10: 11,
            11: 18,
            12: 17,
            13: ignore_label,
            14: ignore_label,
            15: 6,
            16: 9,
            17: 12,
            18: 14,
            19: 15,
            20: 16,
            21: 3,
            22: ignore_label}

classes = ['road',
           'sidewalk',
           'building',
           'wall',
           'fence',
           'pole',
           'traffic light',
           'traffic sign',
           'vegetation',
           'terrain',
           'sky',
           'person',
           'rider',
           'car',
           'truck',
           'bus',
           'train',
           'motorcycle',
           'bicycle']

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]

def maybe_download(url, dest):
    """Download the url to dest if necessary, optionally checking file
    integrity.
    """
    if not os.path.exists(dest):
        logger.info('Downloading %s to %s', url, dest)
        download(url, dest)


def download(url, dest):
    """Download the url to dest, overwriting dest if it already exists."""
    response = requests.get(url, stream=True)
    with open(dest, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

