# coding=utf-8
# Copyright 2024  Bofeng Huang

"""
Zip utils.

Adapted from https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_text/data_utils.py
"""

import zipfile
from pathlib import Path
from typing import Union

# import soundfile as sf
from tqdm import tqdm


def create_zip(data_root: Union[Path, str], zip_path: Union[Path, str], ext: str = "wav"):
    data_root = Path(data_root) if isinstance(data_root, str) else data_root
    # paths = list(data_root.glob("*.npy"))
    # paths.extend(data_root.glob("*.flac"))
    paths = list(data_root.glob(f"*.{ext}"))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as f:
        for path in tqdm(paths, desc="Writing to zip", unit=" samples"):
            f.write(path, arcname=path.name)


def get_zip_manifest(zip_path: str):
    with zipfile.ZipFile(zip_path, mode="r") as f:
        info = f.infolist()
    # paths, lengths = {}, {}
    paths = {}
    for i in tqdm(info, desc="Reading zip", unit=" samples"):
        utt_id = Path(i.filename).stem
        # 30 bytes for zip file header
        offset, file_size = i.header_offset + 30 + len(i.filename), i.file_size
        paths[utt_id] = f"{zip_path}:{offset}:{file_size}"
        # with open(zip_path, "rb") as f:
        #     f.seek(offset)
        #     byte_data = f.read(file_size)
        #     assert len(byte_data) > 1
        #     assert is_sf_audio_data(byte_data), i
        #     byte_data_fp = io.BytesIO(byte_data)
        #     lengths[utt_id] = sf.info(byte_data_fp).frames
    # return paths, lengths
    return paths
