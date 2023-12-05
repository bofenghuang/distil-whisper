import os
import sys

sys.path.append("/home/bhuang/myscripts")
os.environ["HF_HOME"] = "/projects/bhuang/.cache/huggingface"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["BITSANDBYTES_NOWELCOME"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from run_distillation_b import main

teacher_model_name_or_path = "bofenghuang/whisper-large-v3-french"
model_name_or_path = "/home/bhuang/asr/distil-whisper/training/outputs/models/bofenghuang-whisper_large_v3_french_dec2_init"
output_dir = "/home/bhuang/asr/distil-whisper/training/outputs/models/tmp"

train_file = "/home/bhuang/asr/distil-whisper/training/outputs/data/bofenghuang-whisper_large_v3_french/train-data-wer.json"
validation_file = "/projects/bhuang/corpus/speech/nemo_manifests/final/2023-11-21/test_asr_mcv13_manifest_normalized_pnc.json+/projects/bhuang/corpus/speech/nemo_manifests/final/2023-11-21/test_asr_mls_manifest_normalized_pnc_cleaned.json+/projects/bhuang/corpus/speech/nemo_manifests/final/2023-11-21/test_asr_voxpopuli_manifest_normalized_pnc.json+/projects/bhuang/corpus/speech/nemo_manifests/final/2023-11-21/test_asr_fleurs_manifest_normalized_pnc.json+/projects/bhuang/corpus/speech/nemo_manifests/final/2023-11-21/test_asr_african_accented_french_manifest_normalized_pnc_cleaned.json+/projects/bhuang/corpus/speech/nemo_manifests/final/2023-11-21/test_asr_mtedx_manifest_normalized_pnc.json"


main(
    [
        "--model_name_or_path",
        model_name_or_path,
        "--teacher_model_name_or_path",
        teacher_model_name_or_path,
        "--train_file",
        train_file,
        "--validation_file",
        validation_file,
        "--audio_column_name",
        "audio_filepath",
        "--text_column_name",
        # "text",
        "whisper_transcript",
        "--eval_text_column_name",
        "text",
        "--max_duration_in_seconds",
        "30",
        "--wer_threshold",
        "10",
        "--timestamp_probability",
        # "0.2",
        "1.0",
        "--condition_on_prev_probability",
        "0",
        "--language",
        "french",
        "--task",
        "transcribe",
        "--preprocessing_num_workers",
        "32",
        "--dataloader_num_workers",
        "1",
        "--output_dir",
        "output_dir",
        "--overwrite_output_dir",
        "True",
        "--num_train_epochs",
        "6",
        "--per_device_train_batch_size",
        "16",
        "--per_device_eval_batch_size",
        "16",
        "--gradient_accumulation_steps",
        "1",
        "--kl_weight",
        "1.0",
        "--temperature",
        "2.0",
        "--optim",
        "adamw_bnb_8bit",
        "--learning_rate",
        "4.375e-6",
        "--warmup_ratio",
        "0.05",
        "--lr_scheduler_type",
        "cosine",
        "--dtype",
        "float16",
        "--gradient_checkpointing",
        "True",
        "--freeze_encoder",
        "True",
        "--logging_steps",
        "10",
        "--eval_steps",
        "1",
        "--save_steps",
        "500",
        "--save_total_limit",
        "3",
        "--predict_with_generate",
        "True",
        "--generation_num_beams",
        "1",
        "--ddp_timeout",
        "7200",
        "--wandb_project",
        "hf-whisper-v3",
        "--do_train",
        "True",
        "--do_eval",
        "True",
    ]
)
