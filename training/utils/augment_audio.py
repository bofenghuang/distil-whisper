#!/usr/bin/env python
# coding=utf-8
# Copyright 2022  Bofeng Huang

"""
Augment audio using audiomentations.

Better do this on-the-fly, somewhere in dataset/data_collator.

Usage:
# pip install audiomentations

"""

from typing import Optional

from audiomentations import AddBackgroundNoise  # FrequencyMask,; TimeMask,
from audiomentations import AddGaussianNoise, Compose, Gain, OneOf, PitchShift, PolarityInversion, SomeOf, TimeStretch


class SpeechAugmentator:
    def __init__(self, background_noise_dir: Optional[str] = None, prob: float = 1.0):
        # bh: tried proba O.5 for all, seems to be too aggressive
        # Compose applies the given sequence of transforms when called, optionally shuffling the sequence for every call
        # self.transform = Compose(
        #     [
        #         OneOf(
        #             [
        #                 AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.015, p=1.0),
        #                 AddBackgroundNoise(
        #                     sounds_path=background_noise_dir,
        #                     min_snr_in_db=3.0,
        #                     max_snr_in_db=30.0,
        #                     noise_transform=PolarityInversion(),
        #                     p=1.0,
        #                 ),
        #             ],
        #             p=0.2,
        #         ),
        #         # TimeStretch(min_rate=0.9, max_rate=1.1, p=0.2, leave_length_unchanged=False),
        #         # todo: set to True since don't want it to exceed 30s for whisper models, may need to set to False for other models
        #         TimeStretch(min_rate=0.9, max_rate=1.1, p=0.2, leave_length_unchanged=True),
        #         PitchShift(min_semitones=-4, max_semitones=4, p=0.2),
        #         # Gain(min_gain_in_db=-6, max_gain_in_db=6, p=0.2),
        #         # TimeMask(min_band_part=0.1, max_band_part=0.15, fade=True, p=0.05),
        #         # FrequencyMask(min_frequency_band=0.2, max_frequency_band=0.4, p=0.05),
        #     ],
        #     p=prob,
        # )

        # SomeOf randomly picks several of the given transforms when called
        # pick exactly two of the transforms
        # SomeOf(2, [transform1, transform2, transform3])
        # pick 1 to 3 of the transforms
        # SomeOf((1, 3), [transform1, transform2, transform3])
        # pick 2 to all of the transforms
        # SomeOf((2, None), [transform1, transform2, transform3, transform4])
        # self.transform = SomeOf(
        #     (1, None),
        #     [
        #         OneOf(
        #             [
        #                 AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.015, p=1.0),
        #                 AddBackgroundNoise(
        #                     sounds_path=background_noise_dir,
        #                     min_snr_in_db=3.0,
        #                     max_snr_in_db=30.0,
        #                     noise_transform=PolarityInversion(),
        #                     p=1.0,
        #                 ),
        #             ],
        #             # p=0.2,
        #         ),
        #         TimeStretch(min_rate=0.9, max_rate=1.1, p=0.2, leave_length_unchanged=False),
        #         PitchShift(min_semitones=-4, max_semitones=4, p=0.2),
        #         # Gain(min_gain_in_db=-6, max_gain_in_db=6, p=0.2),
        #         # TimeMask(min_band_part=0.1, max_band_part=0.15, fade=True, p=0.05),
        #         # FrequencyMask(min_frequency_band=0.2, max_frequency_band=0.4, p=0.05),
        #     ],
        #     p=prob,
        # )

        # todo: reverb, check more augmentation in NeMo and Speechbrain
        transforms = [
            AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.015, p=1.0),
            # todo: set to True since don't want it to exceed 30s for whisper models, may need to set to False for other models
            TimeStretch(min_rate=0.9, max_rate=1.1, leave_length_unchanged=True, p=1.0),
            PitchShift(min_semitones=-4, max_semitones=4, p=1.0),
            # not helpful when processor normalize waveform
            # Gain(min_gain_in_db=-6, max_gain_in_db=6, p=1.0),
            # TimeMask(min_band_part=0.1, max_band_part=0.15, fade=True, p=0.05),
            # FrequencyMask(min_frequency_band=1.0, max_frequency_band=0.4, p=0.05),
        ]

        if background_noise_dir is not None:
            transforms += [
                AddBackgroundNoise(
                    sounds_path=background_noise_dir,
                    min_snr_in_db=3.0,
                    max_snr_in_db=30.0,
                    noise_transform=PolarityInversion(),
                    p=1.0,
                )
            ]

        # OneOf randomly picks one of the given transforms when called
        self.transform = OneOf(transforms, p=prob)

    def __call__(self, waveform, sample_rate):
        waveform = self.transform(waveform, sample_rate=sample_rate)
        return waveform


def main(
    input_file_path: str,
    output_file_path: str,
    background_noise_dir: str,
    prob: float = 1.0,
    audio_column_name: str = "audio",
    audio_dir_suffix: str = "_augmented",
    num_workers: int = 16,
):
    import json
    import os
    import time
    from pathlib import Path

    import librosa
    import soundfile as sf
    import tqdm
    from datasets import Audio, load_dataset

    SAMPLING_RATE = 16000

    dataset = load_dataset("json", data_files=input_file_path, split="train")
    print(dataset)

    dataset = dataset.cast_column(audio_column_name, Audio(sampling_rate=SAMPLING_RATE, mono=True))

    augmentator = SpeechAugmentator(background_noise_dir, prob=prob)

    def augment_dataset(batch):
        # load and resample
        sample = batch[audio_column_name]

        # apply augmentation
        augmented_waveform = augmentator(sample["array"], sample_rate=sample["sampling_rate"])
        # batch[audio_column_name]["array"] = augmented_waveform

        p = Path(sample["path"])
        p = p.parent.parent / f"{p.parent.name}{audio_dir_suffix}" / p.name
        audio_filepath = p.as_posix()

        os.makedirs(p.parent.as_posix(), exist_ok=True)

        sf.write(audio_filepath, augmented_waveform, samplerate=sample["sampling_rate"], format="wav")

        batch[f"tmp_{audio_column_name}"] = audio_filepath

        batch["duration"] = librosa.get_duration(y=augmented_waveform, sr=sample["sampling_rate"])

        return batch

    start_time = time.perf_counter()

    dataset = dataset.map(augment_dataset, num_proc=num_workers, desc="augment train dataset")

    dataset = dataset.remove_columns([audio_column_name])
    dataset = dataset.rename_column(f"tmp_{audio_column_name}", audio_column_name)

    # dataset.to_json(output_file_path, orient="records", lines=True, force_ascii=False)
    # better handle backslash in path
    with open(output_file_path, "w") as manifest_f:
        for _, sample in enumerate(tqdm.tqdm(dataset, desc="Saving", total=dataset.num_rows, unit=" samples")):
            manifest_f.write(f"{json.dumps(sample, ensure_ascii=False)}\n")

    print(
        f"Generation completed in {time.strftime('%Hh%Mm%Ss', time.gmtime(time.perf_counter() - start_time))}. The generated"
        f" data is saved in {output_file_path}"
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
