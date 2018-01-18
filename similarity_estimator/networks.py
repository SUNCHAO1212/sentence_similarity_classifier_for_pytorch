""" An implementation of the siamese RNN for sentence similarity classification outlined in Mueller et al., 
"Siamese Recurrent Architectures for Learning Sentence Similarity." """

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm

import torch.nn.functional as F

# Inference with SVR
import pickle

from utils.parameter_initialization import xavier_normal


# ## Functions to accomplish attention
def batch_matmul_bias(seq, weight, bias, nonlinearity=''):
    s = None
    bias_dim = bias.size()
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)
        _s_bias = _s + bias.expand(bias_dim[0], _s.size()[0]).transpose(0,1)
        if(nonlinearity=='tanh'):
            _s_bias = torch.tanh(_s_bias)
        _s_bias = _s_bias.unsqueeze(0)
        if(s is None):
            s = _s_bias
        else:
            s = torch.cat((s,_s_bias),0)
    # return s.squeeze()
    return s


def batch_matmul(seq, weight, nonlinearity=''):
    s = None
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)
        if(nonlinearity=='tanh'):
            _s = torch.tanh(_s)
        _s = _s.unsqueeze(0)
        if(s is None):
            s = _s
        else:
            s = torch.cat((s, _s), 0)
    return s.squeeze().view(s.size(0), -1)
    # return s


def attention_mul(rnn_outputs, att_weights):
    attn_vectors = None
    for i in range(rnn_outputs.size(0)):
        h_i = rnn_outputs[i]
        a_i = att_weights[i].unsqueeze(1).expand_as(h_i)
        h_i = a_i * h_i
        h_i = h_i.unsqueeze(0)
        if(attn_vectors is None):
            attn_vectors = h_i
        else:
            attn_vectors = torch.cat((attn_vectors,h_i),0)
    return torch.sum(attn_vectors, 0)


class LSTMEncoder(nn.Module):
    """ Implements the network type integrated within the Siamese RNN architecture. """
    def __init__(self, vocab_size, opt, is_train=False):
        super(LSTMEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.opt = opt
        self.name = 'sim_encoder'
        # Layers
        self.embedding_table = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.opt.embedding_dims,
                                            padding_idx=0, max_norm=None, scale_grad_by_freq=False, sparse=False)
        self.lstm_rnn = nn.LSTM(input_size=self.opt.embedding_dims, hidden_size=self.opt.hidden_dims, num_layers=1)

        """"""
        # attention params
        self.weight_W_word = nn.Parameter(torch.FloatTensor(self.opt.hidden_dims, self.opt.hidden_dims))
        self.bias_word = nn.Parameter(torch.FloatTensor(self.opt.hidden_dims, 1))
        self.weight_proj_word = nn.Parameter(torch.FloatTensor(self.opt.hidden_dims, 1))
        self.softmax_word = nn.Softmax()
        self.weight_W_word.data.uniform_(-0.1, 0.1)
        self.weight_proj_word.data.uniform_(-0.1, 0.1)

    def initialize_hidden_plus_cell(self, batch_size):
        """ Re-initializes the hidden state, cell state, and the forget gate bias of the network. """
        zero_hidden = Variable(torch.randn(1, batch_size, self.opt.hidden_dims))
        zero_cell = Variable(torch.randn(1, batch_size, self.opt.hidden_dims))
        return zero_hidden, zero_cell

    def forward(self, input_len, batch_size, input_data, hidden, cell):
        """ Performs a forward pass through the network. """
        embedded = self.embedding_table(input_data).view(input_len, batch_size, -1)
        for i in range(self.opt.num_layers):    # num_layers = 1
            output, (hidden, cell) = self.lstm_rnn(embedded, (hidden, cell))
        return output, hidden, cell


class SiameseClassifier(nn.Module):
    """ Sentence similarity estimator implementing a siamese arcitecture. Uses pretrained word2vec embeddings.
    Different to the paper, the weights are untied, to avoid exploding/ vanishing gradients. """
    def __init__(self, vocab_size, opt, pretrained_embeddings=None, is_train=False):
        super(SiameseClassifier, self).__init__()
        self.opt = opt
        # Initialize constituent network
        self.encoder_a = self.encoder_b = LSTMEncoder(vocab_size, self.opt, is_train)
        # Initialize pre-trained embeddings, if given
        if pretrained_embeddings is not None:
            self.encoder_a.embedding_table.weight.data.copy_(pretrained_embeddings)
        # Initialize network parameters
        self.initialize_parameters()
        # Declare loss function
        self.loss_function = nn.MSELoss()
        # Initialize network optimizers
        self.optimizer_a = optim.Adam(self.encoder_a.parameters(), lr=self.opt.learning_rate,
                                      betas=(self.opt.beta_1, 0.999))
        self.optimizer_b = optim.Adam(self.encoder_a.parameters(), lr=self.opt.learning_rate,
                                      betas=(self.opt.beta_1, 0.999))

    def forward(self):
        """ Performs a single forward pass through the siamese architecture. """
        # Checkpoint the encoder state
        state_dict = self.encoder_a.state_dict()

        # Obtain the input length (each batch consists of padded sentences)
        input_length = self.batch_a.size(0)  # seq, batch, embedding

        # Obtain sentence encodings from each encoder
        hidden_a, cell_a = self.encoder_a.initialize_hidden_plus_cell(self.batch_size)
        net_output_a, hidden_a, cell_a = self.encoder_a(input_length, self.batch_size, self.batch_a, hidden_a, cell_a)
        # add attention
        word_squish_a = batch_matmul_bias(net_output_a, self.encoder_a.weight_W_word, self.encoder_a.bias_word,
                                          nonlinearity='tanh')
        word_attn_a = batch_matmul(word_squish_a, self.encoder_a.weight_proj_word)
        word_attn_norm_a = self.encoder_a.softmax_word(word_attn_a.transpose(1, 0))
        word_attn_vectors_a = attention_mul(net_output_a, word_attn_norm_a.transpose(1, 0))

        # Restore checkpoint to establish weight-sharing
        self.encoder_b.load_state_dict(state_dict)
        hidden_b, cell_b = self.encoder_b.initialize_hidden_plus_cell(self.batch_size)
        net_output_b, hidden_b, cell_b = self.encoder_b(input_length, self.batch_size, self.batch_b, hidden_b, cell_b)
        # add attention
        word_squish_b = batch_matmul_bias(net_output_b, self.encoder_b.weight_W_word, self.encoder_b.bias_word,
                                          nonlinearity='tanh')
        word_attn_b = batch_matmul(word_squish_b, self.encoder_b.weight_proj_word)
        word_attn_norm_b = self.encoder_b.softmax_word(word_attn_b.transpose(1, 0))
        word_attn_vectors_b = attention_mul(net_output_b, word_attn_norm_b.transpose(1, 0))

        # Format sentence encodings as 2D tensors, using attention to replace output
        self.encoding_a = word_attn_vectors_a
        self.encoding_b = word_attn_vectors_b

        # Obtain similarity score predictions by calculating the Manhattan distance between sentence encodings
        if self.batch_size == 1:
            self.prediction = torch.exp(-torch.norm((self.encoding_a - self.encoding_b), 1))
        else:
            self.prediction = torch.exp(-torch.norm((self.encoding_a - self.encoding_b), 1, 1))

    def get_loss(self):
        """ Calculates the MSE loss between the network predictions and the ground truth. """
        # Loss is the L1 norm of the difference between the obtained sentence encodings
        labels = (self.labels - 1.0) / 4.0
        prediction = self.prediction
        self.loss = self.loss_function(prediction, labels)


    def load_pretrained_parameters(self):
        """ Loads the parameters learned during the pre-training on the SemEval data. """
        pretrained_state_dict_path = os.path.join(self.opt.pretraining_dir, self.opt.pretrained_state_dict)
        self.encoder_a.load_state_dict(torch.load(pretrained_state_dict_path))
        print('Pretrained parameters have been successfully loaded into the encoder networks.')

    def initialize_parameters(self):
        """ Initializes network parameters. """
        state_dict = self.encoder_a.state_dict()
        for key in state_dict.keys():
            if '.weight' in key:
                state_dict[key] = xavier_normal(state_dict[key])
            if '.bias' in key:
                bias_length = state_dict[key].size()[0]
                start, end = bias_length // 4, bias_length // 2
                state_dict[key][start:end].fill_(2.5)
        nn.init.constant(self.encoder_a.weight_W_word, 0.1)
        nn.init.constant(self.encoder_a.bias_word, 0.1)
        nn.init.constant(self.encoder_a.weight_proj_word, 0.1)
        self.encoder_a.load_state_dict(state_dict)

    def train_step(self, train_batch_a, train_batch_b, train_labels):
        """ Optimizes the parameters of the active networks, i.e. performs a single training step. """
        # Get batches
        self.batch_a = train_batch_a
        self.batch_b = train_batch_b
        self.labels = train_labels

        # Get batch_size for current batch
        self.batch_size = self.batch_a.size(1)

        # Get gradients
        self.forward()
        self.encoder_a.zero_grad()  # encoder_a == encoder_b
        self.get_loss()
        self.loss.backward()

        # Clip gradients
        clip_grad_norm(self.encoder_a.parameters(), self.opt.clip_value)

        self.optimizer_a.step()
        # print('lstm_rnn')
        # print(self.encoder_a.lstm_rnn.all_weights)
        # print('weight_proj_word:')
        # print(self.encoder_a.weight_proj_word)
        # print(self.encoder_a.weight_proj_word.grad)
        # print('weight_proj_word')
        # print('weight_W_word:')
        # print(self.encoder_a.weight_W_word)
        # print(self.encoder_a.weight_W_word.grad)
        # print('weight_W_word')
        # print('bias_word:')
        # print(self.encoder_a.bias_word)
        # print(self.encoder_a.bias_word.grad)
        # print('bias_word')

    def test_step(self, test_batch_a, test_batch_b, test_labels):
        """ Performs a single test step. """
        # Get batches
        self.batch_a = test_batch_a
        self.batch_b = test_batch_b
        self.labels = test_labels

        # Get batch_size for current batch
        self.batch_size = self.batch_a.size(1)

        svr_path = os.path.join(self.opt.save_dir, 'sim_svr.pkl')
        if os.path.exists(svr_path):
            # Correct predictions via trained SVR
            with open(svr_path, 'rb') as f:
                sim_svr = pickle.load(f)
            self.forward()
            self.prediction = Variable(torch.FloatTensor(sim_svr.predict(self.prediction.view(-1, 1).data.numpy())))

        else:
            self.forward()

        self.get_loss()
