import sys
import math
import torch
import torchaudio
import urllib.request
import tarfile
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from datasets import load_metric
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


from flashlight.lib.sequence.criterion import CpuViterbiPath, get_data_ptr_as_bytes
from flashlight.lib.text.dictionary import create_word_dict, load_words
from flashlight.lib.text.decoder import (
        CriterionType,
        LexiconDecoderOptions,
        KenLM,
        LM,
        LMState,
        SmearingMode,
        Trie,
        LexiconDecoder)
from flashlight.lib.text.decoder import LexiconFreeDecoder, LexiconFreeDecoderOptions



wer = load_metric("wer")


kenlm_args = {
    "kenlm_model_path": "ru.lm.bin",
    "lexicon_path": "ru-jon.lexicon",
    "beam": 1000,
    "nbest": 1,
    "beam_threshold": 20,
    "lm_weight": 1.0,
    "word_score": 1.0,
    "sil_weight": 0
}

class KenLMDecoder(object):
    def __init__(self, kenlm_args, vocab_dict, blank="<pad>", silence="|", unk="<unk>"):

        self.vocab_size = len(vocab_dict)
        self.blank_token = (vocab_dict[blank])
        self.silence_token = vocab_dict[silence]
        self.unk_token = vocab_dict[unk]

        self.nbest = kenlm_args['nbest']

        if kenlm_args['lexicon_path']:
            vocab_keys = vocab_dict.keys()
            self.lexicon = load_words(kenlm_args['lexicon_path'])
            self.word_dict = create_word_dict(self.lexicon)
            self.unk_word = self.word_dict.get_index(unk)

            self.lm = KenLM(kenlm_args['kenlm_model_path'], self.word_dict)
            self.trie = Trie(self.vocab_size, self.silence_token)

            start_state = self.lm.start(False)
            for i, (word, spellings) in enumerate(self.lexicon.items()):
                word_idx = self.word_dict.get_index(word)
                _, score = self.lm.score(start_state, word_idx)

                for spelling in spellings:
                    spelling_idxs = []
                    for token in spelling:
                        if token.upper() in vocab_keys:
                            spelling_idxs.append(vocab_dict[token.upper()])
                        elif token.lower() in vocab_keys:
                            spelling_idxs.append(vocab_dict[token.lower()])
                        else:
                            print("WARNING: The token", token, "not exist in your vocabulary, using <unk> token instead")
                            spelling_idxs.append(self.unk_token)
                    self.trie.insert(spelling_idxs, word_idx, score)
            self.trie.smear(SmearingMode.MAX)

            self.decoder_opts = LexiconDecoderOptions(
                beam_size=kenlm_args['beam'],
                beam_size_token=kenlm_args['beam_size_token'] if "beam_size_token" in kenlm_args else len(vocab_dict),
                beam_threshold=kenlm_args['beam_threshold'],
                lm_weight=kenlm_args['lm_weight'],
                word_score=kenlm_args['word_score'],
                unk_score=-math.inf,
                sil_score=kenlm_args['sil_weight'],
                log_add=False,
                criterion_type=CriterionType.CTC,
            )

            self.decoder = LexiconDecoder(
                self.decoder_opts,
                self.trie,
                self.lm,
                self.silence_token,
                self.blank_token,
                self.unk_word,
                [],
                False,
            )
        else:
            d = {w: [[w]] for w in vocab_dict.keys()}
            self.word_dict = create_word_dict(d)
            self.lm = KenLM(kenlm_args['kenlm_model_path'], self.word_dict)
            self.decoder_opts = LexiconFreeDecoderOptions(
                beam_size=kenlm_args['beam'],
                beam_size_token=kenlm_args['beam_size_token'] if "beam_size_token" in kenlm_args else len(vocab_dict),
                beam_threshold=kenlm_args['beam_threshold'],
                lm_weight=kenlm_args['lm_weight'],
                sil_score=kenlm_args['sil_weight'],
                log_add=False,
                criterion_type=CriterionType.CTC,
            )
            self.decoder = LexiconFreeDecoder(
                self.decoder_opts, self.lm, self.silence_token, self.blank_token, []
            )

    def get_tokens(self, idxs):
        """Normalize tokens by handling CTC blank"""
        idxs = (g[0] for g in it.groupby(idxs))
        idxs = filter(lambda x: x != self.blank_token, idxs)
        return torch.LongTensor(list(idxs))

    def decode(self, emissions):
        B, T, N = emissions.size()
        # print(emissions.shape)
        tokens = []
        scores = []
        for b in range(B):
            emissions_ptr = emissions.data_ptr() + 4 * b * emissions.stride(0)
            results = self.decoder.decode(emissions_ptr, T, N)
            nbest_results = results[: self.nbest]
            tokens_nbest = []
            scores_nbest = []
            for result in nbest_results:
                tokens_nbest.append(result.tokens)
                scores_nbest.append(result.score)
            tokens.append(tokens_nbest)
            scores.append(scores_nbest)

        token_array = np.array(tokens, dtype=object).transpose((1, 0, 2))
        scores_arrray = np.array(scores, dtype=object).transpose()
        return token_array, scores_arrray





processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-russian")
model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-russian")
model.to("cuda")

vocab_dict = processor.tokenizer.get_vocab()
pad_token = processor.tokenizer.pad_token
silence_token = processor.tokenizer.word_delimiter_token
unk_token = processor.tokenizer.unk_token
kenlm = KenLMDecoder(kenlm_args, vocab_dict, blank=pad_token, silence=silence_token, unk=unk_token)



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

    logits = torch.nn.functional.log_softmax(logits.float(), dim=-1)
    # get all candidates
    lm_tokens, lm_scores = kenlm.decode(logits.cpu().detach())
    # choise the best candidate
    pred_ids = lm_tokens[0][:]

    #pred_ids = torch.argmax(logits, dim=-1)

    targets.append(row["sentence"])
    preds.append(processor.batch_decode(pred_ids)[0].lower().replace("-", " "))


for x, y in zip(preds, targets):
    print (x, "|", y)

print("WER: {:2f}".format(100 * wer.compute(predictions=preds, references=targets)))
