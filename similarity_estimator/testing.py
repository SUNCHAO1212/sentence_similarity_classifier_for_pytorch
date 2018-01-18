""" Tests the performance of the trained model by checking its predictive accuracy on n randomly sampled items. """

import os
import pickle
import torch
from similarity_estimator.networks import SiameseClassifier
from similarity_estimator.options import TestingOptions, ClusterOptions
from similarity_estimator.sim_util import load_similarity_data
from utils.data_server import DataServer
from utils.init_and_storage import load_network

# Initialize training parameters
opt = TestingOptions()
# Obtain data
extended_corpus_path = os.path.join(opt.data_dir, 'extended_sick.txt')
# corpus_data format: [[('sentence_a', 'sentence_b'),('', '')],['score','']]
_, corpus_data = load_similarity_data(opt, extended_corpus_path, 'sick_corpus')
# Load extended vocab
vocab_path = os.path.join(opt.save_dir, 'extended_vocab.pkl')
with open(vocab_path, 'rb') as f:
    _, vocab = pickle.load(f)

# Initialize the similarity classifier
classifier = SiameseClassifier(vocab.n_words, opt, is_train=False)
# Load best available configuration (or modify as needed)
load_network(classifier.encoder_a, 'sim_classifier', 'latest', opt.save_dir)

# Initialize a data loader from randomly shuffled corpus data; inspection limited to individual items, hence bs=1
# shuffled_loader format: [(('sentence_a', 'sentence_b'),('','')), ('score','')]
shuffled_loader = DataServer(corpus_data, vocab, opt, is_train=False, use_buckets=False, volatile=True)

# Keep track of performance
total_classification_divergence = 0.0
total_classification_loss = 0.0

# Test loop
prediction_list = []
label_list = []
for i, data in enumerate(shuffled_loader):
    # Upon completion
    if i >= opt.num_test_samples:
        average_classification_divergence = total_classification_divergence / opt.num_test_samples
        average_classification_loss = total_classification_loss / opt.num_test_samples
        print('=================================================\n'
              '= Testing concluded after examining %d samples. =\n'
              '= Average classification divergence is %.4f.  =\n'
              '= Average classification loss (MSE) is %.4f.  =\n'
              '=================================================' %
              (opt.num_test_samples, average_classification_divergence, average_classification_loss))
        break

    s1_var, s2_var, label_var = data
    # Get predictions and update tracking values
    classifier.test_step(s1_var, s2_var, label_var)
    prediction = classifier.prediction
    print((prediction))
    if prediction.data[0] > 0.0001:
        prediction_list.append(1.0)
    else:
        prediction_list.append(0.0)
    label_list.append(((label_var - 1.0) / 4.0).data[0][0])
    loss = classifier.loss.data[0]
    divergence = torch.abs((prediction - (label_var - 1.0) / 4.0).data[0])
    total_classification_divergence += divergence
    total_classification_loss += loss

    sentence_a = ' '.join([vocab.index_to_word[int(idx[0])] if idx[0] != 0 else '' for idx in
                           s1_var.data.numpy().tolist()])
    sentence_b = ' '.join([vocab.index_to_word[int(idx[0])] if idx[0] != 0 else '' for idx in
                           s2_var.data.numpy().tolist()])

    print('Sample: %d\n'
          'Sentence A: %s\n'
          'Sentence B: %s\n'
          'Prediction: %.4f\n'
          'Ground truth: %.4f\n'
          'Divergence: %.4f\n'
          'Loss: %.4f\n' %
          (i, sentence_a, sentence_b, prediction.data[0], (label_var.data[0][0] - 1.0) / 4.0, divergence, loss))
    with open('models/test_record.txt', 'a+') as fo:
        fo.write('\nSample: %d\n'
          'Sentence A: %s\n'
          'Sentence B: %s\n'
          'Prediction: %.4f\n'
          'Ground truth: %.4f\n'
          'Divergence: %.4f\n'
          'Loss: %.4f\n' %
          (i, sentence_a, sentence_b, prediction.data[0], (label_var.data[0][0] - 1.0) / 4.0, divergence, loss))
print(prediction_list)
print(label_list)
accuracy = 0
for i in range(len(prediction_list)):
    if prediction_list[i] == label_list[i]:
        accuracy += 1
    else:
        pass
print('Accuracy = %f' % (accuracy/len(prediction_list)))

