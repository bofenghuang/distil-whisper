#!/usr/bin/env python
# coding=utf-8
# Copyright 2024  Bofeng Huang

# HF_HUB_ENABLE_HF_TRANSFER=1


from huggingface_hub import HfApi
api = HfApi()

api.upload_large_folder(
# api.upload_folder(
    # repo_id="bofenghuang/stt-pseudo-labeled-whisper-large-v3-multilingual",
    repo_id="bofenghuang/stt-pseudo-labeled-whisper-large-v3-multilingual-test",
    repo_type="dataset",
    # folder_path="/projects/bhuang/corpus/speech/nemo_manifests",
    folder_path="/projects/bhuang/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual-test",
    # path_in_repo="facebook/voxpopuli/fr/train_concatenated",
    # allow_patterns="*.zip",
    # allow_patterns="espnet/yodas/fr000/train_concatenated",
    # allow_patterns="google/fleurs/fr_fr/*",
    # ignore_patterns="**/logs/*.txt", # Ignore all text logs
    ignore_patterns=[
        # valid, test data
        "*/valid*/*",
        "*/test*/*",
        # specific datasets
        "BrunoHays/*",
        "eustlb/*",
        "gigant/*",
        "speech-recognition-community-v2/*",
        # intermediate files
        # "*manifest.json",
        # "*manifest_whisper_large_v3.json",
        # "*manifest_whisper_large_v3_norm.json",
        # "*manifest_whisper_large_v3_norm_upprev.json",
        # "*manifest_whisper_large_v3_norm_upprev_wer_filt.json",
        # # mcv
        # "*manifest_whisper_large_v3_norm_wer_filt.json"
        # "*manifest_whisper_large_v3_norm_wer_filt_wer.json"
        # big files
        # "facebook/multilingual_librispeech/*"
        # "espnet/yodas/it100/*"
        # "espnet/yodas/es000/*"
    ],
    # delete_patterns="*.txt", # Delete all remote text files before
    num_workers=16,
)


# HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli upload-large-folder --repo-type dataset bofenghuang/stt-pseudo-labeled-whisper-large-v3-multilingual /projects/bhuang/corpus/speech/nemo_manifests --include google/fleurs/fr_fr/*
# HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli upload-large-folder --exclude "*/valid*/*" "*/test*/*" "BrunoHays/*" "google/*" "eustlb/*" "gigant/*" "speech-recognition-community-v2/*" --repo-type dataset bofenghuang/stt-pseudo-labeled-whisper-large-v3-multilingual /projects/bhuang/corpus/speech/nemo_manifests --include facebook/voxpopuli/fr/train_concatenated/*.zip

# Usage:  huggingface-cli upload [repo_id] [local_path] [path_in_repo]
# HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli upload --repo-type dataset bofenghuang/stt-pseudo-labeled-whisper-large-v3-multilingual /projects/bhuang/corpus/speech/nemo_manifests/google/fleurs/fr_fr google/fleurs/fr_fr

# api.delete_files(
#     repo_id="bofenghuang/stt-pseudo-labeled-whisper-large-v3-multilingual",
#     repo_type="dataset",
#     delete_patterns=[
#         # "*/valid*",
#         # "*.wav",
#         "tmp/*",
#         "code-switching/yodas/*",
#     ],
# )
