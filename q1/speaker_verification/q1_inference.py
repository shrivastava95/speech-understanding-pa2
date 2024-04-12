import torch
from verification import verification_multi
import os

MODEL_LIST = ['hubert_large', 'wav2vec2_xlsr', 'unispeech_sat', "wavlm_large"]
CHECKPOINT_LIST = [
    '/workspace/ishaan/su/checkpoints/hubert_large_finetune.pth',
    '/workspace/ishaan/su/checkpoints/wav2vec2_xlsr_finetune.pth',
    '/workspace/ishaan/su/checkpoints/unispeech_sat_large_finetune.pth',
    '/workspace/ishaan/su/checkpoints/wavlm_large_finetune.pth',    
]

voxceleb1_subset_path = '/workspace/ishaan/su/datasets/voxceleb1h_subset/VoxCeleb1_subset'
pathextend = lambda path: os.path.join(voxceleb1_subset_path, path)


list_test_annotations = torch.load(pathextend('list_test_hard_filtered.pt'))
list_wav12 = [[pathextend(annot[1]), pathextend(annot[2])] for annot in list_test_annotations]


GROUND_TRUTH = [annot[0] for annot in list_test_annotations]
PREDICTION_LIST = []

if not os.path.exists('results_q1_inference.pt'):
    results = {'GROUND_TRUTH':GROUND_TRUTH, 'PREDICTION_LIST':PREDICTION_LIST, 'MODEL_LIST':MODEL_LIST, 'CHECKPOINT_LIST':CHECKPOINT_LIST}
    torch.save(results, 'results_q1_inference.pt')
results = torch.load('results_q1_inference.pt')
PREDICTION_LIST = results['PREDICTION_LIST']

with torch.no_grad():
    for i, (model_name, checkpoint) in enumerate(list(zip(MODEL_LIST, CHECKPOINT_LIST))):
        if i < len(PREDICTION_LIST):
            continue
        predictions = verification_multi(model_name, list_wav12, use_gpu=True, checkpoint=checkpoint)
        predictions = [pred.item() for pred in predictions]
        PREDICTION_LIST.append(predictions)

        results = {'GROUND_TRUTH':GROUND_TRUTH, 'PREDICTION_LIST':PREDICTION_LIST, 'MODEL_LIST':MODEL_LIST, 'CHECKPOINT_LIST':CHECKPOINT_LIST}
        torch.save(results, 'results_q1_inference.pt')
