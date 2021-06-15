import sys
import torch
import torchaudio
import urllib.request
import tarfile
import pandas as pd
from tqdm.auto import tqdm
from datasets import load_metric
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

wer = load_metric("wer")

processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-russian")
model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-russian")
model.to("cuda")

cv_test = []
clips_path = sys.argv[2]

for line in open(sys.argv[1]):
    items = line.split()
    row = {"path": items[0].split("_")[-1] + ".wav", "sentence": " ".join(items[1:])}
    cv_test.append(row)

cv_test = pd.DataFrame(cv_test)
print (cv_test)

def clean_sentence(sent):
    sent = sent.lower()
    # these letters are considered equivalent in written Russian
    sent = sent.replace('ё', 'е')
    # replace non-alpha characters with space
    sent = "".join(ch if ch.isalpha() else " " for ch in sent)
    # remove repeated spaces
    sent = " ".join(sent.split())
    return sent

targets = []
preds = []

for i, row in tqdm(cv_test.iterrows(), total=cv_test.shape[0]):
    row["sentence"] = clean_sentence(row["sentence"])
    speech_array, sampling_rate = torchaudio.load(clips_path + "/" + row["path"])
    resampler = torchaudio.transforms.Resample(sampling_rate, 16_000)
    row["speech"] = resampler(speech_array).squeeze().numpy()

    inputs = processor(row["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values.to("cuda"), attention_mask=inputs.attention_mask.to("cuda")).logits

    pred_ids = torch.argmax(logits, dim=-1)

    targets.append(row["sentence"])
    preds.append(processor.batch_decode(pred_ids)[0].lower())


for x, y in zip(preds, targets):
    print (x, "|", y)

print("WER: {:2f}".format(100 * wer.compute(predictions=preds, references=targets)))
