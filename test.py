import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import DataSample, dataset_to_variable, CustomDataset, collate_fn
from torch.utils.data import DataLoader
import numpy as np
from model import Attention


num_to_label = ['pants-fire',
                'false',
                'barely-true',
                'half-true',
                'mostly-true',
                'true']

label_to_number = {
	'pants-fire': 0,
	'false': 1,
	'barely-true': 2,
	'half-true': 3,
	'mostly-true': 4,
	'true': 5
}

def find_word(word2num, token):
    if token in word2num:
        return word2num[token]
    else:
        return word2num['<unk>']

def test_data_prepare(test_file, word2num, phase):
    test_input = open(test_file, 'rb')
    test_data = test_input.read().decode('utf-8')
    test_input.close()

    statement_word2num = word2num[0]
    subject_word2num = word2num[1]
    speaker_word2num = word2num[2]
    speaker_pos_word2num = word2num[3]
    state_word2num = word2num[4]
    party_word2num = word2num[5]
    context_word2num = word2num[6]

    test_samples = []
    labels = []
    for line in test_data.strip().split('\n'):
        tmp = line.strip().split('\t')
        while len(tmp) < 14:
            tmp.append('')
        if phase == 'test':
            p = DataSample('test', tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7], tmp[13])
            labels.append(label_to_number.get(tmp[1], -1))
        elif phase == 'valid':
            p = DataSample(tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7], tmp[13])

        for i in range(len(p.statement)):
            p.statement[i] = find_word(statement_word2num, p.statement[i])
        for i in range(len(p.subject)):
            p.subject[i] = find_word(subject_word2num, p.subject[i])
        p.speaker = find_word(speaker_word2num, p.speaker)
        for i in range(len(p.speaker_pos)):
            p.speaker_pos[i] = find_word(speaker_pos_word2num, p.speaker_pos[i])
        p.state = find_word(state_word2num, p.state)
        p.party = find_word(party_word2num, p.party)
        for i in range(len(p.context)):
            p.context[i] = find_word(context_word2num, p.context[i])

        test_samples.append(p)

    if phase == 'test':
      return test_samples, labels

    return test_samples

def test(test_file, test_output, word2num,
         model_path, batch_size, lr,
         val_acc, sources, use_cuda = False):

    print('  Constructing network model...')
    statement_word2num = word2num[0]
    subject_word2num = word2num[1]
    speaker_word2num = word2num[2]
    speaker_pos_word2num = word2num[3]
    state_word2num = word2num[4]
    party_word2num = word2num[5]
    context_word2num = word2num[6]


    model = Attention(len(statement_word2num),
                len(subject_word2num),
                len(speaker_word2num),
                len(speaker_pos_word2num),
                len(state_word2num),
                len(party_word2num),
                len(context_word2num),
                sources = sources)

    state_dict = torch.load(os.path.join(model_path, 'model_bs_{}_lr_{}_acc_{}.pth'.format(batch_size, lr, val_acc)))
    model.load_state_dict(state_dict)
    test_samples, labels = test_data_prepare(test_file, word2num, 'test')
    dataset_to_variable(test_samples, use_cuda)
    out = open(test_output, 'w')
    acc = 0

    for (sample, label) in zip(test_samples, labels):
        statement = Variable(sample.statement).unsqueeze(0)
        subject = Variable(sample.subject).unsqueeze(0)
        speaker = Variable(sample.speaker).unsqueeze(0)
        speaker_pos = Variable(sample.speaker_pos).unsqueeze(0)
        state = Variable(sample.state).unsqueeze(0)
        party = Variable(sample.party).unsqueeze(0)
        context = Variable(sample.context).unsqueeze(0)

        prediction = model(statement, subject, speaker, speaker_pos, state, party, context)
        prediction = int(np.argmax(prediction.data.numpy()))
        if prediction == label:
          acc += 1    
        out.write(num_to_label[prediction]+'\n')

    out.close()
    acc /= len(test_samples)
    print('================================')
    print('Test accuracy :: {}'.format(acc))
    print('Val accuracy :: {}'.format(val_acc))
    print('================================')
    file_path = f'{model_path}/results.csv'
    if os.path.exists(file_path):
      f = open(file_path, 'a')
      f.write(f'{" + ".join(sources)},{val_acc}, {acc} \n ')
      f.close()
    else : 
      f = open(file_path, 'x')
      f.write('sources, validation accuracy, test accuracy \n')
      f.write(f'{" + ".join(sources)},{val_acc}, {acc} \n')
      f.close()



def valid(valid_loader, word2num, model, max_len_statement, max_len_subject, max_len_speaker_pos, max_len_context, use_cuda):
    acc = 0
    n = len(valid_loader.dataset)


    for (inputs_statement, inputs_subject, inputs_speaker, inputs_speaker_pos, inputs_state, inputs_party, inputs_context, target) in valid_loader:
        
        if use_cuda:
          inputs_statement.cuda()
          inputs_subject.cuda()
          inputs_speaker.cuda()
          inputs_speaker_pos.cuda()
          inputs_state.cuda()
          inputs_party.cuda()
          inputs_context.cuda()
          target.cuda()

        prediction = model(inputs_statement, inputs_subject, inputs_speaker, inputs_speaker_pos, inputs_state, inputs_party, inputs_context)
        prediction = prediction.max(1, keepdim=True)[1]
        acc += prediction.eq(target.data.view_as(prediction)).cpu().sum()
    
    acc = float(acc.item()) / n
    print('  Validation Accuracy: '+str(acc))
    return acc
