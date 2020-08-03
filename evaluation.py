import os
import json
import argparse

import torch
import torchaudio
import pandas as pd

from models import EncoderRNN, AttnDecoderRNN, GenderModelPL
from word2numrus.extractor import NumberExtractor

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def decode_text(seq, inv_emb):
    text = " ".join([inv_emb[k] for k in seq.numpy() if k > 1])
    return text


def evaluate(input_tensor, encoder, decoder, embedding, max_length=512, max_output_len=8):
    with torch.no_grad():
        if len(input_tensor.shape) == 2:
            input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.permute(2, 0, 1)  # 512x64x128
        encoder_hidden = encoder.initHidden(batch_size=1,
                                            input_size=2)  # 1, 512, 128

        input_length = input_tensor.size(0)
        target_length = max_output_len

        encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)  # 512, 64, 128 | hidden 1, 64, 128
        encoder_outputs = encoder_outputs[:, 0, :]

        decoder_hidden = encoder_hidden
        decoder_input = torch.eye(decoder.output_size)[embedding['<SOS>']].unsqueeze(0).unsqueeze(0)  # onehot

        output = []
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = decoder_output.unsqueeze(0)
            decoder_token = torch.eye(decoder.output_size)[topi.detach()]  # onehot
            output.append(decoder_token)
            if (decoder_token == torch.eye(len(embedding), device=device)[embedding['<EOS>']]).all():
                break

        output = torch.cat(output).max(2).indices.squeeze()
    return output

def load_recognition_models(enc_path, dec_path, input_size=128, hidden_size=256, embedding_size=44, spec_length=512):
    encoder1 = EncoderRNN(input_size, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, embedding_size, dropout_p=0.1, max_length=spec_length).to(device)

    encoder1.load_state_dict(torch.load(enc_path))
    attn_decoder1.load_state_dict(torch.load(dec_path))

    encoder1.eval()
    attn_decoder1.eval()
    return encoder1, attn_decoder1

def load_gender_model(path):
    gender_model = GenderModelPL()
    gender_model.load_state_dict(torch.load(path))
    gender_model.eval()
    return gender_model

def get_mel(path):
    waveform, sample_rate = torchaudio.load(path)
    mel = torchaudio.transforms.MelSpectrogram()(waveform).squeeze(0)
    return mel

def spec_padding(spec, target_length=512):
    """longest spec found was of length 460. Everything longer than target_len will be truncated."""
    z = torch.zeros([128, target_length])
    z[:spec.shape[0], :spec.shape[1]] = spec
    return z

def read_spec(path):
    mel = get_mel(path)
    spec_for_asr = spec_padding(mel).to(torch.float32)
    spec_for_gender = mel[:, :128]
    return spec_for_asr, spec_for_gender

def get_gender(spec, model):
    classes = ['female', 'male']
    spec = spec.unsqueeze(0)
    out = int(model(spec).round().item())
    return classes[out]

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('input_path',
                        help="Path to file to be processed")

    parser.add_argument('output_path',
                        help="Path to result", default='out.csv', required=False)

    args = parser.parse_args()
    return args

class Solution():
    def __init__(self):
        with open('embedding.json', 'r', encoding='utf-8') as fp:
            self.embedding = json.load(fp)

        self.inv_embedding = {v: k for k, v in self.embedding.items()}

        self.encoder1, self.attn_decoder1 = load_recognition_models("states/encoder.weights", 'states/decoder.weights')
        self.gender_model = load_gender_model('states/gender_model.weights')
        self.extractor = NumberExtractor()

    def process(self, path='numbers/test-example/209c6fb213.wav'):
        spec_for_asr, spec_for_gender = read_spec(path)
        seq = evaluate(spec_for_asr, self.encoder1, self.attn_decoder1, embedding=self.embedding)
        sex = get_gender(spec_for_gender, self.gender_model)

        num = self.extractor.replace_groups(decode_text(seq, self.inv_embedding))

        return sex, num

def main():
    args = parse_args()

    assert os.path.exists(args.input_path)
    solution = Solution()
    df = pd.read_csv(args.input_path)
    dirname_abs = os.path.dirname(os.path.abspath(args.input_path))

    outputs = []
    for path in df.path:
        gender, number = solution.process(os.path.join(dirname_abs, path))
        outputs.append({'path':path, 'number':number})

    pd.DataFrame(outputs).to_csv(args.output_path)

if __name__ == "__main__":
    main()


