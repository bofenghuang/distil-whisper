# coding=utf-8
# Copyright 2024  Bofeng Huang

"""
Audio utils.

Adapted from https://github.com/facebookresearch/fairseq/blob/main/fairseq/data/audio/audio_utils.py
"""

import io
import mmap
import wave
from pathlib import Path
from typing import Any, BinaryIO, List, Optional, Tuple, Union

import numpy as np
import torch

try:
    import soundfile as sf
except ImportError:
    raise ImportError("Please install soundfile: pip install soundfile")

try:
    import torchaudio.sox_effects as ta_sox
except ImportError:
    raise ImportError("Please install torchaudio: pip install torchaudio")

SF_AUDIO_FILE_EXTENSIONS = {".wav", ".flac", ".ogg"}
FEATURE_OR_SF_AUDIO_FILE_EXTENSIONS = {".npy", ".wav", ".flac", ".ogg"}


def convert_waveform(
    waveform: Union[np.ndarray, torch.Tensor],
    sample_rate: int,
    normalize_volume: bool = False,
    to_mono: bool = False,
    to_sample_rate: Optional[int] = None,
) -> Tuple[Union[np.ndarray, torch.Tensor], int]:
    """convert a waveform:
    - to a target sample rate
    - from multi-channel to mono channel
    - volume normalization

    Args:
        waveform (numpy.ndarray or torch.Tensor): 2D original waveform
            (channels x length)
        sample_rate (int): original sample rate
        normalize_volume (bool): perform volume normalization
        to_mono (bool): convert to mono channel if having multiple channels
        to_sample_rate (Optional[int]): target sample rate
    Returns:
        waveform (numpy.ndarray): converted 2D waveform (channels x length)
        sample_rate (float): target sample rate
    """

    effects = []
    if normalize_volume:
        effects.append(["gain", "-n"])
    if to_sample_rate is not None and to_sample_rate != sample_rate:
        effects.append(["rate", f"{to_sample_rate}"])
    if to_mono and waveform.shape[0] > 1:
        effects.append(["channels", "1"])
    if len(effects) > 0:
        is_np_input = isinstance(waveform, np.ndarray)
        _waveform = torch.from_numpy(waveform) if is_np_input else waveform
        converted, converted_sample_rate = ta_sox.apply_effects_tensor(
            _waveform, sample_rate, effects
        )
        if is_np_input:
            converted = converted.numpy()
        return converted, converted_sample_rate
    return waveform, sample_rate


def get_waveform(
    path_or_fp: Union[str, BinaryIO],
    normalization: bool = True,
    mono: bool = True,
    frames: int = -1,
    start: int = 0,
    always_2d: bool = True,
    output_sample_rate: Optional[int] = None,
    normalize_volume: bool = False,
    # waveform_transforms: Optional[CompositeAudioWaveformTransform] = None,
    waveform_transforms: Any = None,
) -> Tuple[np.ndarray, int]:
    """Get the waveform and sample rate of a 16-bit WAV/FLAC/OGG Vorbis audio.

    Args:
        path_or_fp (str or BinaryIO): the path or file-like object
        normalization (bool): normalize values to [-1, 1] (Default: True)
        mono (bool): convert multi-channel audio to mono-channel one
        frames (int): the number of frames to read. (-1 for reading all)
        start (int): Where to start reading. A negative value counts from the end.
        always_2d (bool): always return 2D array even for mono-channel audios
        output_sample_rate (Optional[int]): output sample rate
        normalize_volume (bool): normalize volume
    Returns:
        waveform (numpy.ndarray): 1D or 2D waveform (channels x length)
        sample_rate (float): sample rate
    """
    if isinstance(path_or_fp, str):
        ext = Path(path_or_fp).suffix
        if ext not in SF_AUDIO_FILE_EXTENSIONS:
            raise ValueError(f"Unsupported audio format: {ext}")

    waveform, sample_rate = sf.read(
        path_or_fp, dtype="float32", always_2d=True, frames=frames, start=start
    )
    waveform = waveform.T  # T x C -> C x T

    # waveform, sample_rate = torchaudio.load(file_path, channels_first=True, frame_offset=start, num_frames=frames)

    waveform, sample_rate = convert_waveform(
        waveform,
        sample_rate,
        normalize_volume=normalize_volume,
        to_mono=mono,
        to_sample_rate=output_sample_rate,
    )

    if not normalization:
        waveform *= 2**15  # denormalized to 16-bit signed integers

    if waveform_transforms is not None:
        waveform, sample_rate = waveform_transforms(waveform, sample_rate)

    if not always_2d:
        waveform = waveform.squeeze(axis=0)

    return waveform, sample_rate


def get_waveform_from_stored_zip(
    path,
    byte_offset,
    byte_size,
    use_sample_rate=None,
    waveform_transforms=None,
):
    assert path.endswith(".zip")
    data = read_from_stored_zip(path, byte_offset, byte_size)
    f = io.BytesIO(data)
    # error on empty audio
    assert is_sf_audio_data(data), path
    return get_waveform(
        f,
        always_2d=False,
        output_sample_rate=use_sample_rate,
        waveform_transforms=waveform_transforms,
    )


def get_waveform_from_audio_or_stored_zip(path: str, use_sample_rate=None, waveform_transforms=None):
    """Get speech features from .npy file or waveform from .wav/.flac file.
    The file may be inside an uncompressed ZIP file and is accessed via byte
    offset and length.

    Args:
        path (str): File path in the format of "<.npy/.wav/.flac path>" or
        "<zip path>:<byte offset>:<byte length>".
        need_waveform (bool): return waveform instead of features.
        use_sample_rate (int): change sample rate for the input wave file

    Returns:
        features_or_waveform (numpy.ndarray): speech features or waveform.
    """
    _path, slice_ptr = parse_path(path)
    if len(slice_ptr) == 0:
        return get_waveform(
            _path,
            always_2d=False,
            output_sample_rate=use_sample_rate,
            waveform_transforms=waveform_transforms,
        )
    elif len(slice_ptr) == 2:
        return get_waveform_from_stored_zip(
            _path,
            slice_ptr[0],
            slice_ptr[1],
            use_sample_rate=use_sample_rate,
            waveform_transforms=waveform_transforms,
        )
    else:
        raise ValueError(f"Invalid path: {path}")


def get_waveform_bytes_from_audio_or_stored_zip(path: str) -> bytes:
    _path, slice_ptr = parse_path(path)
    if len(slice_ptr) == 0:
        return get_waveform_bytes(_path)
    elif len(slice_ptr) == 2:
        return get_waveform_bytes_from_stored_zip(
            _path,
            slice_ptr[0],
            slice_ptr[1],
        )
    else:
        raise ValueError(f"Invalid path: {path}")


def get_waveform_bytes(path: str) -> bytes:
    with wave.open(path, "rb") as f:
        return f.readframes(f.getnframes())


def get_waveform_bytes_from_stored_zip(zip_path: str, offset: int, length: int) -> bytes:
    # shift head of 44 bytes
    return read_from_stored_zip(zip_path, offset, length)[44:]


def is_sf_audio_data(data: bytes) -> bool:
    is_wav = data[0] == 82 and data[1] == 73 and data[2] == 70
    is_flac = data[0] == 102 and data[1] == 76 and data[2] == 97
    is_ogg = data[0] == 79 and data[1] == 103 and data[2] == 103
    return is_wav or is_flac or is_ogg


def mmap_read(path: str, offset: int, length: int) -> bytes:
    with open(path, "rb") as f:
        # f.fileno() for file handle
        # length in bytes of the memory map
        # 0 is a special value indicating that the system should create a memory map large enough to hold the entire file
        # mmap mode should be compatible with open
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_o:
            data = mmap_o[offset: offset + length]
    return data


def read_from_stored_zip(zip_path: str, offset: int, length: int) -> bytes:
    return mmap_read(zip_path, offset, length)


def parse_path(path: str) -> Tuple[str, List[int]]:
    """Parse data path which is either a path to
    1. a .npy/.wav/.flac/.ogg file
    2. a stored ZIP file with slicing info: "[zip_path]:[offset]:[length]"

      Args:
          path (str): the data path to parse

      Returns:
          file_path (str): the file path
          slice_ptr (list of int): empty in case 1;
            byte offset and length for the slice in case 2
    """

    if Path(path).suffix in FEATURE_OR_SF_AUDIO_FILE_EXTENSIONS:
        _path, slice_ptr = path, []
    else:
        _path, *slice_ptr = path.split(":")
        if not Path(_path).is_file():
            raise FileNotFoundError(f"File not found: {_path}")
    assert len(slice_ptr) in {0, 2}, f"Invalid path: {path}"
    slice_ptr = [int(i) for i in slice_ptr]
    return _path, slice_ptr
