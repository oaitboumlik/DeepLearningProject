import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import random
import numpy as np
from model import Attention
from test import valid
from data import dataset_to_variable, CustomDataset, collate_fn
from torch.utils.data import DataLoader

def train(train_samples,
          valid_samples,
          word2num,
          max_len_statement,
          max_len_subject, 
          max_len_speaker_pos, 
          max_len_context,
          lr = 0.001,
          epoch = 1,
          use_cuda = False,
          batch_size=20,
          batch_size_val=5):

    print('Training...')

    # Prepare training data
    print('  Preparing training data...')
    statement_word2num = word2num[0]
    subject_word2num = word2num[1]
    speaker_word2num = word2num[2]
    speaker_pos_word2num = word2num[3]
    state_word2num = word2num[4]
    party_word2num = word2num[5]
    context_word2num = word2num[6]

    # train_data = train_samples
    train_data = CustomDataset(train_samples, max_len_statement,
                               max_len_subject, max_len_speaker_pos, max_len_context)
    train_loader = DataLoader(train_data,
                            batch_size=batch_size,
                            collate_fn=collate_fn)

    # dataset_to_variable(train_data, use_cuda)
    valid_data = valid_samples
    valid_samples = CustomDataset(valid_samples, max_len_statement,
                                max_len_subject, max_len_speaker_pos, max_len_context)
    valid_loader = DataLoader(valid_samples,
                            batch_size=batch_size_val,
                            collate_fn=collate_fn)

    # dataset_to_variable(valid_data, use_cuda)

    # Construct model instance
    print('  Constructing network model...')
    model = Attention(len(statement_word2num),
                len(subject_word2num),
                len(speaker_word2num),
                len(speaker_pos_word2num),
                len(state_word2num),
                len(party_word2num),
                len(context_word2num))
    if use_cuda:
        print('using cuda')
        model.cuda()

    # Start training
    print('  Start training')

    optimizer = optim.Adam(model.parameters(), lr = lr)
    model.train()

    step = 0
    display_interval = 50

    for epoch_ in range(epoch):
        print('  ==> Epoch '+str(epoch_)+' started.')
        # random.shuffle(train_data)
        total_loss = 0
        for (inputs_statement, inputs_subject, inputs_speaker, inputs_speaker_pos, inputs_state, inputs_party, inputs_context,  target) in train_loader:

            # sample = [inputs_statement, inputs_subject, inputs_speaker, inputs_speaker_pos, inputs_state, inputs_party, inputs_context]
            optimizer.zero_grad()
            if use_cuda:
              inputs_statement.cuda()
              inputs_subject.cuda()
              inputs_speaker.cuda()
              inputs_speaker_pos.cuda()
              inputs_state.cuda()
              inputs_party.cuda()
              inputs_context.cuda()
              # sample.cuda()
              target.cuda()

            prediction = model(inputs_statement, inputs_subject, inputs_speaker, inputs_speaker_pos, inputs_state, inputs_party, inputs_context)
            # label = Variable(torch.LongTensor([sample.label]))
            # loss = F.cross_entropy(prediction, label)
            loss = F.cross_entropy(prediction, target)
            loss.backward()
            optimizer.step()

            step += 1
            if step % display_interval == 0:
                print('    ==> Iter: '+str(step)+' Loss: '+str(loss))

            total_loss += loss.data.numpy() * len(inputs_statement)

        print('  ==> Epoch '+str(epoch_)+' finished. Avg Loss: '+str(total_loss/len(train_data)))

        valid(valid_loader, word2num, model, max_len_statement, max_len_subject, max_len_speaker_pos, max_len_context, use_cuda)

    return model