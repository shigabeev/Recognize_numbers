import time
import random
import math
import json

import tqdm
import torch
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
import torch.optim as optim

from utils import RecognitionDataLoader, generate_embedding
from models import AttnDecoderRNN, EncoderRNN
from evaluation import evaluate

TEACHER_FORCING_RATIO = 0.5
HIDDEN_SIZE = 256
INPUT_SIZE = 128

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

# def showPlot(points):
#     plt.figure()
#     fig, ax = plt.subplots()
#     # this locator puts ticks at regular intervals
#     loc = ticker.MultipleLocator(base=0.2)
#     ax.yaxis.set_major_locator(loc)
#     plt.plot(points)


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer,
          decoder_optimizer, max_length=512, max_output_len=8):
    target_tensor = target_tensor.transpose(0, 1)
    input_tensor = input_tensor.permute(2, 0, 1)  # 512x64x128
    encoder_hidden = encoder.initHidden(batch_size=1,  # input_tensor.shape[1],
                                        input_size=2)  # 1, 512, 128

    criterion = nn.NLLLoss()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    target_length = target_tensor.size(0)
    loss = torch.zeros([], dtype=torch.double)

    encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)  # 512, 64, 128 | hidden 1, 64, 128
    encoder_outputs = encoder_outputs[:, 0, :] # be careful with processing batches here. It only works for bs=1

    decoder_input = torch.eye(decoder.output_size)[embedding['<SOS>']].unsqueeze(0).unsqueeze(0)  # onehot
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            target = target_tensor[di].reshape(1).to(torch.int64)
            loss += criterion(decoder_output, target)
            decoder_input = torch.eye(decoder.output_size)[target_tensor[di].to(torch.int64)].unsqueeze(
                0)  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            target = target_tensor[di].reshape(1).to(torch.int64)
            loss += criterion(decoder_output, target)
            decoder_input = decoder_output.unsqueeze(0)

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(encoder, decoder, dataloader, epochs=5, print_every=1000, plot_every=100, learning_rate=0.01):
    #plt.figure()
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    n_iters = len(dataloader)
    for iter, (input_tensor, target_tensor) in enumerate(dataloader, 1):
        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer)
        print_loss_total += loss
        plot_loss_total += loss

        # if iter % plot_every == 0:
        #     plot_loss_avg = plot_loss_total / plot_every
        #     plot_losses.append(plot_loss_avg)
        #     plot_loss_total = 0
        #
        #     showPlot(plot_losses)

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg), end='\r')


if __name__ == "__main__":
    df = pd.read_csv('train_norm.csv')
    embedding = generate_embedding(df['намбэр'])
    params_asr = {'batch_size': 1,
                  'num_workers': 6}

    train_X_asr, validation_X_asr, train_y_asr, validation_y_asr = train_test_split(df.path, df['намбэр'])


    train_set_asr = RecognitionDataLoader(train_X_asr, train_y_asr, embedding=embedding, extend=True, out_as_one_hot=False)
    train_loader_asr = DataLoader(train_set_asr, **params_asr, shuffle=True)

    validation_set_asr = RecognitionDataLoader(validation_X_asr, validation_y_asr,  embedding=embedding, extend=True, out_as_one_hot=False)
    validation_loader_asr = DataLoader(validation_set_asr, **params_asr, shuffle=False)

    embedding_size = len(embedding)
    encoder1 = EncoderRNN(INPUT_SIZE, HIDDEN_SIZE).to(device)
    attn_decoder1 = AttnDecoderRNN(HIDDEN_SIZE, embedding_size, dropout_p=0.1, max_length=512).to(device)

    teacher_forcing_ratio = 1.
    learning_rate = 0.01

    for i in range(10):
        trainIters(encoder1, attn_decoder1, train_loader_asr, 1000, print_every=10, plot_every=10,
                   learning_rate=learning_rate)
        torch.save(encoder1.state_dict(), f"states/encoder{i}_.weights")
        torch.save(attn_decoder1.state_dict(), f"states/decoder{i}_.weights")
        teacher_forcing_ratio *= 0.5
        learning_rate *= 0.5

    torch.save(encoder1.state_dict(), 'encoder_.weights')
    torch.save(attn_decoder1.state_dict(), 'attn_decoder_.weights')

    with open('embedding.json', 'w', encoding='utf-8') as fp:
        json.dump(embedding, fp, ensure_ascii=False)


    print('TRUE -> PREDICTED')
    for i, (x, y) in enumerate(validation_loader_asr):
        x = x
        y = y
        out = evaluate(x, encoder1, attn_decoder1)
        true = decode_text(y[0], inv_embedding)
        pred = decode_text(out, inv_embedding)
        print(f"{true} -> {pred}")
        if i == 10:
            break