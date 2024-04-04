python verification.py \
    --model_name unispeech_sat \
    --wav1 "./vox1_data/David_Faustino/hn8GyCJIfLM_0000012.wav" \
    --wav2 "./vox1_data/David_Faustino/xTOk1Jz-F_g_0000015.wav" \
    --checkpoint ../checkpoints/unispeech_sat_large_finetune.pth

python verification.py \
    --model_name unispeech_sat \
    --wav1 "./vox1_data/David_Faustino/hn8GyCJIfLM_0000012.wav" \
    --wav2 "./vox1_data/David_Faustino/xTOk1Jz-F_g_0000015.wav" \
    --checkpoint ../checkpoints/unispeech_sat_large_finetune.pth