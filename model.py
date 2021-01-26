import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class Net(nn.Module):

    def __init__(self,

                 statement_vocab_dim,
                 subject_vocab_dim,
                 speaker_vocab_dim,
                 speaker_pos_vocab_dim,
                 state_vocab_dim,
                 party_vocab_dim,
                 context_vocab_dim,

                 statement_embed_dim = 100,
                 statement_kernel_num = 14,
                 statement_kernel_size = [3, 4, 5],

                 subject_embed_dim = 5,
                 subject_lstm_nlayers = 2,
                 subject_lstm_bidirectional = True,
                 subject_hidden_dim = 5,

                 speaker_embed_dim = 5,

                 speaker_pos_embed_dim = 10,
                 speaker_pos_lstm_nlayers = 2,
                 speaker_pos_lstm_bidirectional = True,
                 speaker_pos_hidden_dim = 5,

                 state_embed_dim = 5,

                 party_embed_dim = 5,

                 context_embed_dim = 20,
                 context_lstm_nlayers = 2,
                 context_lstm_bidirectional = True,
                 context_hidden_dim = 6,
                 dropout = 0.5,):

        # Statement CNN
        super(Net, self).__init__()

        self.statement_vocab_dim = statement_vocab_dim
        self.statement_embed_dim = statement_embed_dim
        self.statement_kernel_num = statement_kernel_num
        self.statement_kernel_size = statement_kernel_size

        self.statement_embedding = nn.Embedding(self.statement_vocab_dim, self.statement_embed_dim)
        self.statement_convs = [nn.Conv2d(1, self.statement_kernel_num, (kernel_, self.statement_embed_dim)) for kernel_ in self.statement_kernel_size]

        # Subject
        self.subject_vocab_dim = subject_vocab_dim
        self.subject_embed_dim = subject_embed_dim
        self.subject_lstm_nlayers = subject_lstm_nlayers
        self.subject_lstm_num_direction = 2 if subject_lstm_bidirectional else 1
        self.subject_hidden_dim = subject_hidden_dim

        self.subject_embedding = nn.Embedding(self.subject_vocab_dim, self.subject_embed_dim)
        self.subject_lstm = nn.LSTM(
            input_size = self.subject_embed_dim,
            hidden_size = self.subject_hidden_dim,
            num_layers = self.subject_lstm_nlayers,
            batch_first = True,
            bidirectional = subject_lstm_bidirectional
        )

        # Speaker
        self.speaker_vocab_dim = speaker_vocab_dim
        self.speaker_embed_dim = speaker_embed_dim

        self.speaker_embedding = nn.Embedding(self.speaker_vocab_dim, self.speaker_embed_dim)

        # Speaker Position
        self.speaker_pos_vocab_dim = speaker_pos_vocab_dim
        self.speaker_pos_embed_dim = speaker_pos_embed_dim
        self.speaker_pos_lstm_nlayers = speaker_pos_lstm_nlayers
        self.speaker_pos_lstm_num_direction = 2 if speaker_pos_lstm_bidirectional else 1
        self.speaker_pos_hidden_dim = speaker_pos_hidden_dim

        self.speaker_pos_embedding = nn.Embedding(self.speaker_pos_vocab_dim, self.speaker_pos_embed_dim)
        self.speaker_pos_lstm = nn.LSTM(
            input_size = self.speaker_pos_embed_dim,
            hidden_size = self.speaker_pos_hidden_dim,
            num_layers = self.speaker_pos_lstm_nlayers,
            batch_first = True,
            bidirectional = speaker_pos_lstm_bidirectional
        )

        # State
        self.state_vocab_dim = state_vocab_dim
        self.state_embed_dim = state_embed_dim

        self.state_embedding = nn.Embedding(self.state_vocab_dim, self.state_embed_dim)

        # Party
        self.party_vocab_dim = party_vocab_dim
        self.party_embed_dim = party_embed_dim

        self.party_embedding = nn.Embedding(self.party_vocab_dim, self.party_embed_dim)

        # Context
        self.context_vocab_dim = context_vocab_dim
        self.context_embed_dim = context_embed_dim
        self.context_lstm_nlayers = context_lstm_nlayers
        self.context_lstm_num_direction = 2 if context_lstm_bidirectional else 1
        self.context_hidden_dim = context_hidden_dim

        self.context_embedding = nn.Embedding(self.context_vocab_dim, self.context_embed_dim)
        self.context_lstm = nn.LSTM(
            input_size = self.context_embed_dim,
            hidden_size = self.context_hidden_dim,
            num_layers = self.context_lstm_nlayers,
            batch_first = True,
            bidirectional = context_lstm_bidirectional
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(self.statement_kernel_size) * self.statement_kernel_num
                            + self.subject_lstm_nlayers * self.subject_lstm_num_direction
                            + self.speaker_embed_dim
                            + self.speaker_pos_lstm_nlayers * self.speaker_pos_lstm_num_direction
                            + self.state_embed_dim
                            + self.party_embed_dim
                            + self.context_lstm_nlayers * self.context_lstm_num_direction,
                            6)

    def forward(self,
                statement, subject, speaker, speaker_pos, state, party, context):
  
        # Statement
        statement_ = self.statement_embedding(statement).unsqueeze(1) 
        statement_ = [F.relu(conv(statement_)).squeeze(3) for conv in self.statement_convs] 
        statement_ = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in statement_] 
        statement_ = torch.cat(statement_, 1)  
        print(f'statement_ :: {statement_.size()}')
        # Subject
        subject_ = self.subject_embedding(subject) 
        _, (subject_, _) = self.subject_lstm(subject_) 
        subject_ = F.max_pool1d(subject_, self.subject_hidden_dim).view(subject_.size(1), -1) 
        print(f'subject_ :: {subject_.size()}')
        # Speaker
        speaker_ = self.speaker_embedding(speaker).squeeze(1)
        print(f'speaker_ :: {speaker_.size()}')
        # Speaker Position
        speaker_pos_ = self.speaker_pos_embedding(speaker_pos)
        _, (speaker_pos_, _) = self.speaker_pos_lstm(speaker_pos_)
        speaker_pos_ = F.max_pool1d(speaker_pos_, self.speaker_pos_hidden_dim).view(speaker_pos_.size(1), -1)
        print(f'speaker_pos_ :: {speaker_pos_.size()}')
        # State
        state_ = self.state_embedding(state).squeeze(1)
        print(f'state :: {state_.size()}')
        # Party
        party_ = self.party_embedding(party).squeeze(1)
        print(f'party_ :: {party_.size()}')
        # Context
        context_ = self.context_embedding(context)
        _, (context_, _) = self.context_lstm(context_)
        context_ = F.max_pool1d(context_, self.context_hidden_dim).view(context_.size(1), -1)
        print(f'context_ :: {context_.size()}')
        # Concatenate
        L = (statement_, subject_, speaker_, speaker_pos_, state_, party_, context_)
        features = torch.cat((statement_, subject_, speaker_, speaker_pos_, state_, party_, context_), 1)
        print(f'features :: {features.size()}')
        features = self.dropout(features)
        print(f'features :: {features.size()}')
        features = self.fc(features)
        print(f'features :: {features.size()}')
        return features


class Attention(nn.Module):

    def __init__(self,

                 statement_vocab_dim,
                 subject_vocab_dim,
                 speaker_vocab_dim,
                 speaker_pos_vocab_dim,
                 state_vocab_dim,
                 party_vocab_dim,
                 context_vocab_dim,

                 statement_embed_dim = 100,
                 statement_kernel_num = 14,
                 statement_kernel_size = [3, 4, 5],

                 subject_embed_dim = 5,
                 subject_lstm_nlayers = 2,
                 subject_lstm_bidirectional = True,
                 subject_hidden_dim = 5,

                 speaker_embed_dim = 5,

                 speaker_pos_embed_dim = 10,
                 speaker_pos_lstm_nlayers = 2,
                 speaker_pos_lstm_bidirectional = True,
                 speaker_pos_hidden_dim = 5,

                 state_embed_dim = 5,

                 party_embed_dim = 5,

                 context_embed_dim = 20,
                 context_lstm_nlayers = 2,
                 context_lstm_bidirectional = True,
                 context_hidden_dim = 6,
                 dropout = 0.5, 
                 
                 h = 60, 
                 sources = ('statement','subject','speaker','speaker_pos','state','party','context')):
        
        
        super(Attention, self).__init__()
        self.sources = sources
        self.attention_projection_dim = h 
        self.attention_score = nn.Linear(self.attention_projection_dim, 1, bias = False)

        # Statement CNN

        self.statement_vocab_dim = statement_vocab_dim
        self.statement_embed_dim = statement_embed_dim
        self.statement_kernel_num = statement_kernel_num
        self.statement_kernel_size = statement_kernel_size

        self.statement_embedding = nn.Embedding(self.statement_vocab_dim, self.statement_embed_dim)
        self.statement_convs = [nn.Conv2d(1, self.statement_kernel_num, (kernel_, self.statement_embed_dim)) for kernel_ in self.statement_kernel_size]
        self.statement_linear_projection = nn.Linear(len(self.statement_kernel_size) * self.statement_kernel_num, self.attention_projection_dim)
        

        
        # Subject
        self.subject_vocab_dim = subject_vocab_dim
        self.subject_embed_dim = subject_embed_dim
        self.subject_lstm_nlayers = subject_lstm_nlayers
        self.subject_lstm_num_direction = 2 if subject_lstm_bidirectional else 1
        self.subject_hidden_dim = subject_hidden_dim

        self.subject_embedding = nn.Embedding(self.subject_vocab_dim, self.subject_embed_dim)
        self.subject_lstm = nn.LSTM(
            input_size = self.subject_embed_dim,
            hidden_size = self.subject_hidden_dim,
            num_layers = self.subject_lstm_nlayers,
            batch_first = True,
            bidirectional = subject_lstm_bidirectional
        )

        self.subject_linear_projection = nn.Linear(self.subject_lstm_nlayers * self.subject_lstm_num_direction, self.attention_projection_dim)

        # Speaker
        self.speaker_vocab_dim = speaker_vocab_dim
        self.speaker_embed_dim = speaker_embed_dim

        self.speaker_embedding = nn.Embedding(self.speaker_vocab_dim, self.speaker_embed_dim)
        self.speaker_linear_projection = nn.Linear(self.speaker_embed_dim, self.attention_projection_dim)

        # Speaker Position
        self.speaker_pos_vocab_dim = speaker_pos_vocab_dim
        self.speaker_pos_embed_dim = speaker_pos_embed_dim
        self.speaker_pos_lstm_nlayers = speaker_pos_lstm_nlayers
        self.speaker_pos_lstm_num_direction = 2 if speaker_pos_lstm_bidirectional else 1
        self.speaker_pos_hidden_dim = speaker_pos_hidden_dim
        

        self.speaker_pos_embedding = nn.Embedding(self.speaker_pos_vocab_dim, self.speaker_pos_embed_dim)
        self.speaker_pos_lstm = nn.LSTM(
            input_size = self.speaker_pos_embed_dim,
            hidden_size = self.speaker_pos_hidden_dim,
            num_layers = self.speaker_pos_lstm_nlayers,
            batch_first = True,
            bidirectional = speaker_pos_lstm_bidirectional
        )

        self.speaker_pos_linear_projection = nn.Linear(self.speaker_pos_lstm_nlayers * self.speaker_pos_lstm_num_direction, self.attention_projection_dim)
        # State
        self.state_vocab_dim = state_vocab_dim
        self.state_embed_dim = state_embed_dim

        self.state_embedding = nn.Embedding(self.state_vocab_dim, self.state_embed_dim)
        self.state_linear_projection = nn.Linear(self.state_embed_dim,  self.attention_projection_dim)

        # Party
        self.party_vocab_dim = party_vocab_dim
        self.party_embed_dim = party_embed_dim

        self.party_embedding = nn.Embedding(self.party_vocab_dim, self.party_embed_dim)
        self.party_linear_projection = nn.Linear(self.party_embed_dim, self.attention_projection_dim)

        # Context
        self.context_vocab_dim = context_vocab_dim
        self.context_embed_dim = context_embed_dim
        self.context_lstm_nlayers = context_lstm_nlayers
        self.context_lstm_num_direction = 2 if context_lstm_bidirectional else 1
        self.context_hidden_dim = context_hidden_dim

        self.context_embedding = nn.Embedding(self.context_vocab_dim, self.context_embed_dim)
        self.context_lstm = nn.LSTM(
            input_size = self.context_embed_dim,
            hidden_size = self.context_hidden_dim,
            num_layers = self.context_lstm_nlayers,
            batch_first = True,
            bidirectional = context_lstm_bidirectional
        )
        self.context_linear_projection = nn.Linear(self.context_lstm_nlayers * self.context_lstm_num_direction, self.attention_projection_dim)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.attention_projection_dim,6)

    def forward(self,
                statement, subject, speaker, speaker_pos, state, party, context):

        statement_ = self.statement_embedding(statement).unsqueeze(1) 
        statement_ = [F.relu(conv(statement_)).squeeze(3) for conv in self.statement_convs] 
        statement_ = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in statement_] 
        statement_ = torch.cat(statement_, 1)  
        statement_projection = torch.tanh(self.statement_linear_projection(statement_))
        statement_attention_score = self.attention_score(statement_projection)



        # Subject
        subject_ = self.subject_embedding(subject) 
        _, (subject_, _) = self.subject_lstm(subject_)
        subject_ = F.max_pool1d(subject_, self.subject_hidden_dim).view(subject_.size(1), -1)
        
        subject_projection = torch.tanh(self.subject_linear_projection(subject_))
        subject_attention_score = self.attention_score(subject_projection)

        # Speaker
        
        speaker_ = self.speaker_embedding(speaker).squeeze(1)
        speaker_projection = torch.tanh(self.speaker_linear_projection(speaker_))
        speaker_attention_score = self.attention_score(speaker_projection)

        # Speaker Position
        speaker_pos_ = self.speaker_pos_embedding(speaker_pos)
        _, (speaker_pos_, _) = self.speaker_pos_lstm(speaker_pos_)
        speaker_pos_ = F.max_pool1d(speaker_pos_, self.speaker_pos_hidden_dim).view(speaker_pos_.size(1), -1)
        
        speaker_pos_projection = torch.tanh(self.speaker_pos_linear_projection(speaker_pos_))
        speaker_pos_attention_score = self.attention_score(speaker_pos_projection)
        
        # State
        state_ = self.state_embedding(state).squeeze(1)
        state_projection = torch.tanh(self.state_linear_projection(state_))
        state_attention_score = self.attention_score(state_projection)

        # Party
        party_ = self.party_embedding(party).squeeze(1)
        party_projection = torch.tanh(self.party_linear_projection(party_))
        party_attention_score = self.attention_score(party_projection)

        # Context
        context_ = self.context_embedding(context)
        _, (context_, _) = self.context_lstm(context_)
        context_ = F.max_pool1d(context_, self.context_hidden_dim).view(context_.size(1), -1)
        context_projection = torch.tanh(self.context_linear_projection(context_))
        context_attention_score = self.attention_score(context_projection)
        
        # Concatenate
        source_to_projection = {
          'statement': statement_projection ,
          'subject': subject_projection,
          'speaker':speaker_projection,
          'speaker_pos': speaker_pos_projection,
          'state':state_projection,
          'party':party_projection,
          'context': context_projection
        }

        
        source_to_score = {
          'statement': statement_attention_score ,
          'subject': subject_attention_score,
          'speaker':speaker_attention_score,
          'speaker_pos': speaker_pos_attention_score,
          'state':state_attention_score,
          'party':party_attention_score,
          'context': context_attention_score
        }
        # get attention weights : 
        a = F.softmax(torch.cat([source_to_score[source] for source in self.sources], 1), dim = 1)


        List_of_projections = [source_to_projection[source] for source in self.sources]
        for i, projection in enumerate(List_of_projections): 
          if i == 0 : 
            features = projection * a[:, i:i+1]
          else : 
            features += projection * a[:, i:i+1]
         
        
        features = self.dropout(features)
        features = self.fc(features)

        return features
