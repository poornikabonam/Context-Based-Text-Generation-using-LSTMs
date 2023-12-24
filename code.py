import torch
import os
import torch.nn as nn
import numpy as np 
import pandas as pd 
from torch.utils.data import TensorDataset, DataLoader
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import nltk
from nltk import tokenize
import gensim
from sklearn.cluster import KMeans
import random
import re
from torch.utils.data.dataset import random_split
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import tokenize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap

nltk.download('punkt')
        
        
with open('/kaggle/input/data-lotr-123/lotr.txt', 'r') as file:
    p=file.read()
    
lotr=sent_tokenize(p)
lotr = lotr[:10000]



f = open('/limit_data.txt', 'w')
f.write('\n'.join(lotr))
f.close()






SPECIAL_WORDS = {'PADDING': '<PAD>'}

def load_data(path):
   
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()

    return data


def preprocess_and_save_data(dataset_path, token_lookup, create_lookup_tables):
    """
    Preprocess Text Data
    """
    text = load_data(dataset_path)
    
    # Ignore notice, since we don't use it for analysing the data
    #text = text[81:]
    token_dict = token_lookup()
    for key, token in token_dict.items():
        text = text.replace(key,' ')
        text=text.replace('  ',' ')

    text = text.lower()
    text = text.split()

    vocab_to_int, int_to_vocab = create_lookup_tables(text + list(SPECIAL_WORDS.values()))
    int_text = [vocab_to_int[word] for word in text]
    pickle.dump((int_text, vocab_to_int, int_to_vocab, token_dict), open('preprocess.p', 'wb'))


def load_preprocess():
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    return pickle.load(open('preprocess.p', mode='rb'))


def save_model(filename, decoder):
    save_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'
    torch.save(decoder, save_filename)


def load_model(filename):
    save_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'
    return torch.load(save_filename)

data_dir = '/limit_data.txt'
text = load_data(data_dir)

view_line_range = (0, 10)

import numpy as np

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))

lines = text.split('\n')
print('Number of lines: {}'.format(len(lines)))
word_count_line = [len(line.split()) for line in lines]
print('Average number of words in each line: {}'.format(np.average(word_count_line)))

print()
print('The lines {} to {}:'.format(*view_line_range))
print('\n'.join(text.split('\n')[view_line_range[0]:view_line_range[1]]))
from collections import Counter


def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenized dictionary where the key is the punctuation and the value is the token
    """
    # TODO: Implement Function
    token = dict()
    token['.'] = '<PERIOD>'
    token[','] = '<COMMA>'
    token['"'] = 'QUOTATION_MARK'
    token[';'] = 'SEMICOLON'
    token['!'] = 'EXCLAIMATION_MARK'
    token['?'] = 'QUESTION_MARK'
    token['('] = 'LEFT_PAREN'
    token[')'] = 'RIGHT_PAREN'
    token['-'] = 'HYPHEN'
    token['\n'] = 'NEW_LINE'
    return token


def remove_special_characters(text):
    # Define a regular expression pattern to match special characters
    pattern = r'[^A-Za-z0-9\s]'
    
    # Use re.sub to replace special characters with an empty string
    cleaned_text = re.sub(pattern, '', text)
    
    return cleaned_text


token_dict = token_lookup()

for key, token in token_dict.items():
        text = text.replace(key,' ')
        text=text.replace('  ',' ')

text = text.lower()
text = text.split()
data=[]
'''for i in sent_tokenize(text):
    temp=[]
    #t=' '.join(i)
    for key, token in token_dict.items():
        i = i.replace(key, ' ')
    #i=i.replace('  ',' ')
    for j in word_tokenize(i):
        temp.append(j.lower())
    data.append(temp)'''
word_embeddings = gensim.models.Word2Vec([text], min_count = 1, vector_size=200,window = 5, sg=0)
#print("tst",word_embeddings.wv.index_to_key)
word_vectors_list = [word_embeddings.wv[word] for word in word_embeddings.wv.index_to_key]


#word_vectors_array = np.array(word_vectors_list).squeeze()

# Specify the number of clusters (adjust as needed)
num_clusters = 100

# Apply K-means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(word_vectors_list)

# Get cluster centers
cluster_centers_list = kmeans.cluster_centers_
cluster_centers = {f'cluster_{i}': center for i, center in enumerate(cluster_centers_list)}

svd = TruncatedSVD(n_components=60, random_state=42)
word_vectors_60d = svd.fit_transform(word_vectors_list)

# Apply t-SNE to reduce 60-dim vectors to 2-dim
tsne = TSNE(n_components=2, random_state=42,perplexity=3)
word_vectors_2d = tsne.fit_transform(word_vectors_60d)

# Get labels for each point based on cluster assignment
labels = kmeans.predict(word_vectors_list)
unique_labels = set(labels)
print(unique_labels)
colormaps = [
    'viridis', 'tab20', 'tab20b', 'tab20c',  # tableau colormaps
]

# Combine colors from selected colormaps
colors = []
for cmap_name in colormaps:
    cmap = plt.get_cmap(cmap_name)
    colors.extend([cmap(i) for i in range(cmap.N)])

# Plot the clusters in 2D
colors=set(colors)
print(len(colors))
plt.figure(figsize=(5, 5))
custom_cmap = ListedColormap(colors)
scatter = plt.scatter(word_vectors_2d[:, 0], word_vectors_2d[:, 1], c=labels, cmap=custom_cmap, alpha=0.4, s=2)

# Display legend with all labels
legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=custom_cmap(i), markersize=5) for i in unique_labels]
legend_labels = [f'Cluster {i}' for i in unique_labels]

# Display legend with all labels
#plt.legend(handles=legend_handles, labels=legend_labels)

# Add labels and title
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('Word Embeddings Clusters Visualization using t-SNE')

# Show the plot
plt.show()

def calculate_cosine_similarity(sentence, context):
    sentence_embedding = np.sum([word_embeddings.wv[word] for word in sentence if word in word_embeddings.wv.index_to_key], axis=0)
    context_word_embedding=word_embeddings.wv[context]
    cosine_similarity = np.dot(sentence_embedding, context_word_embedding) / (np.linalg.norm(sentence_embedding) * np.linalg.norm(context_word_embedding))
    return cosine_similarity

def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    # Handle the case where the denominator is zero
    if norm_a == 0 or norm_b == 0:
        return 0.0

    similarity = dot_product / (norm_a * norm_b)
    return similarity

def extract_context(sentence, cluster_centers, n=5):
    context_vectors = []

    for word in sentence.split():
        if word.lower() in word_embeddings.wv.index_to_key:
            word_vector = word_embeddings.wv[word.lower()]
            context_vectors.append(word_vector)

    if not context_vectors:
        # If no valid word vectors are found, return an empty list
        return []

    # Sum up the context vectors to obtain the sentence vector
    sentence_vector = np.sum(context_vectors, axis=0)

    # Calculate pairwise cosine similarity
    selected_cluster_center = None
    min_cosine_distance = float('inf')

    # Find the closest cluster center
    for cluster_id, cluster_center_vector in cluster_centers.items():
        cosine_distance = cosine_similarity(sentence_vector, cluster_center_vector)
        cosine_distance = abs(cosine_distance)
        if cosine_distance < min_cosine_distance:
            min_cosine_distance = cosine_distance
            selected_cluster_center = cluster_id

    # Get context vector from the selected cluster center
    context_vector = cluster_centers[selected_cluster_center][:, np.newaxis]

    # Get top n words from the selected cluster as context words
    context_words = sorted(word_embeddings.wv.index_to_key,
                           key=lambda w: np.dot(word_embeddings.wv[w].T, context_vector).flatten(),
                           reverse=True)[:n]

    return context_words
    


def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    word_count = Counter(text)
    print("text in lookup",text[:20])
    if '<SEP>' not in word_count:
        word_count['<SEP>'] = 0
    sorted_vocab = sorted(word_count, key = word_count.get, reverse=True)
    int_to_vocab = {ii:word for ii, word in enumerate(sorted_vocab)}
    #print("in to vo",int_to_vocab)
    vocab_to_int = {word:ii for ii, word in int_to_vocab.items()}
    
    # return tuple
    return (vocab_to_int, int_to_vocab)

preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)

int_text, vocab_to_int, int_to_vocab, token_dict = load_preprocess()
train_on_gpu = torch.cuda.is_available()

def batch_data_base(words, sequence_length, batch_size):
    """
    Batch the neural network data using DataLoader
    :param words: The word ids of the TV scripts
    :param sequence_length: The sequence length of each batch
    :param batch_size: The size of each batch; the number of sequences in a batch
    :return: DataLoader with batched data
    """
    n_batches = len(words)//batch_size
    x, y = [], []
    words = words[:n_batches*batch_size]
    
    for ii in range(0, len(words)-sequence_length):
        i_end = ii+sequence_length        
        batch_x = words[ii:ii+sequence_length]
        x.append(batch_x)
        batch_y = words[i_end]
        y.append(batch_y)
    
    data = TensorDataset(torch.from_numpy(np.asarray(x)), torch.from_numpy(np.asarray(y)))
    data_loader = DataLoader(data, shuffle=True, batch_size=batch_size)
        
    
    # return a dataloader
    return data_loader

def batch_data_context(words, sequence_length, batch_size):
    """
    Batch the neural network data using DataLoader
    :param words: The word ids of the TV scripts
    :param sequence_length: The sequence length of each batch
    :param batch_size: The size of each batch; the number of sequences in a batch
    :return: DataLoader with batched data
    """
    print("words",words[:20])
    n_batches = len(words)//batch_size
    x, y = [], []
    words = words[:n_batches*batch_size]
    
    for ii in range(0, len(words)-sequence_length):
        i_end = ii+sequence_length        
        batch_x = words[ii:ii+sequence_length]
        seq = [int_to_vocab[i] for i in batch_x]
        sentence=' '.join(seq)
        context = extract_context(sentence, cluster_centers)
        #print("context",context)
        for j in context:
          try:
            new=[j] + ['<SEP>']
            sentence=new+seq
            new_batch=[vocab_to_int[i] for i in sentence]
            break
          except Exception as e:
            pass
        x.append(new_batch)
        batch_y = words[i_end]
        y.append(batch_y)
    
    data = TensorDataset(torch.from_numpy(np.asarray(x)), torch.from_numpy(np.asarray(y)))
    data_loader = DataLoader(data, shuffle=True, batch_size=batch_size)
        
    
    # return a dataloader
    return data_loader

# test dataloader



class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_ii = torch.randn(hidden_size, input_size, requires_grad=True)
        self.W_hi = torch.randn(hidden_size, hidden_size, requires_grad=True)
        self.b_ii = torch.zeros(hidden_size, requires_grad=True)
        self.b_hi = torch.zeros(hidden_size, requires_grad=True)

        self.W_if = torch.randn(hidden_size, input_size, requires_grad=True)
        self.W_hf = torch.randn(hidden_size, hidden_size, requires_grad=True)
        self.b_if = torch.zeros(hidden_size, requires_grad=True)
        self.b_hf = torch.zeros(hidden_size, requires_grad=True)

        self.W_ig = torch.randn(hidden_size, input_size, requires_grad=True)
        self.W_hg = torch.randn(hidden_size, hidden_size, requires_grad=True)
        self.b_ig = torch.zeros(hidden_size, requires_grad=True)
        self.b_hg = torch.zeros(hidden_size, requires_grad=True)

        self.W_io = torch.randn(hidden_size, input_size, requires_grad=True)
        self.W_ho = torch.randn(hidden_size, hidden_size, requires_grad=True)
        self.b_io = torch.zeros(hidden_size, requires_grad=True)
        self.b_ho = torch.zeros(hidden_size, requires_grad=True)

        self.init_weights()

    def init_weights(self):
        for p in [self.W_ii, self.W_hi, self.W_if, self.W_hf, self.W_ig, self.W_hg, self.W_io, self.W_ho]:
            torch.nn.init.xavier_uniform_(p)

    def forward_pass(self, x, init_states=None):
        bs, _ = x.size()
        h_t, c_t = (torch.zeros(self.hidden_size).to(x.device),
                    torch.zeros(self.hidden_size).to(x.device)) if init_states is None else init_states

        i_t = torch.sigmoid(x @ self.W_ii.t() + self.b_ii + h_t @ self.W_hi.t() + self.b_hi)
        f_t = torch.sigmoid(x @ self.W_if.t() + self.b_if + h_t @ self.W_hf.t() + self.b_hf)
        g_t = torch.tanh(x @ self.W_ig.t() + self.b_ig + h_t @ self.W_hg.t() + self.b_hg)
        o_t = torch.sigmoid(x @ self.W_io.t() + self.b_io + h_t @ self.W_ho.t() + self.b_ho)

        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t
class RNN:
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        self.embedding_weights = torch.randn(vocab_size, embedding_dim, requires_grad=True)
        self.lstm = LSTMCell(embedding_dim, hidden_dim)
        self.dropout = dropout
        self.fc_weights = torch.randn(hidden_dim, output_size, requires_grad=True)
        
    def parameters(self):
        # Manual parameter management
        return [self.embedding_weights, self.lstm.W_ii, self.lstm.W_hi, self.lstm.b_ii, self.lstm.b_hi,
                self.lstm.W_if, self.lstm.W_hf, self.lstm.b_if, self.lstm.b_hf,
                self.lstm.W_ig, self.lstm.W_hg, self.lstm.b_ig, self.lstm.b_hg,
                self.lstm.W_io, self.lstm.W_ho, self.lstm.b_io, self.lstm.b_ho,
                self.fc_weights]

    def forward(self, x, hidden):
        #embeds = self.embedding(x)
        self.embedding_weights = self.embedding_weights.to(x.device)
        embeds  = self.embedding_weights[x]
        lstm_out = []
        for i in range(embeds.size(1)):
            h_t, c_t = self.lstm.forward_pass(embeds[:, i, :], hidden)
            hidden = (h_t, c_t)
            lstm_out.append(h_t)

        lstm_out = torch.stack(lstm_out, dim=1)
        #lstm_out = self.dropout(lstm_out)
        lstm_out = lstm_out * (torch.rand_like(lstm_out) > self.dropout) / (1 - self.dropout)
        lstm_out = lstm_out.contiguous().view(-1, self.lstm.hidden_size)
        out = torch.matmul(lstm_out, self.fc_weights)
        out = out.view(x.size(0), -1, self.fc_weights.size(1))
        out = out[:, -1]
        return out, hidden

    def init_hidden(self, batch_size):
        parameters = self.parameters()
        return (
            torch.zeros(batch_size, self.lstm.hidden_size).to(parameters[0].device),
            torch.zeros(batch_size, self.lstm.hidden_size).to(parameters[0].device)
        )
'''def forward_back_prop(rnn, optimizer, criterion, inp, target, hidden):
    if train_on_gpu:
        rnn.cuda()

    h = tuple([each.data for each in hidden])
    rnn.zero_grad()
    inputs, targets = inp.cuda(), target.cuda()
    output, h = rnn(inputs, h)
    loss = criterion(output, targets)
    loss.backward()
    nn.utils.clip_grad_norm_(rnn.parameters(), 5)
    optimizer.step()

    return loss.item(), h'''

def calculate_accuracy(output, target):
    _, predicted = torch.max(output, 1)
    correct = (predicted == target).float()
    accuracy = correct.sum() / len(target)
    return accuracy.item()

def train_rnn(rnn, batch_size, learning_rate, criterion, num_epochs, show_every_n_batches, train_loader, val_loader):
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

    epoch_cosine_similarity_values = []
    epoch_loss_values = []
    epoch_accuracy_values = []
    epoch_val_loss_values = []
    epoch_val_accuracy_values = []

    for epoch in range(num_epochs):
        total_cosine_similarity = 0.0
        total_loss = 0.0
        total_accuracy = 0.0
        total_val_loss = 0.0
        total_val_accuracy = 0.0

        # Training
        for i, (input_sequence, target_sequence) in enumerate(train_loader, 1):
            init_states = None

            output, _ = rnn.forward(input_sequence, init_states)

            loss = criterion(output, target_sequence)

            optimizer.zero_grad()
            loss.backward()
            clip_value = 5.0
            for param in rnn.parameters():
                param.grad.data = torch.clamp(param.grad.data, -clip_value, clip_value)
            optimizer.step()

            with torch.no_grad():
                batch_cosine_similarity = 0.0
                for context, target_word_id in zip(input_sequence, target_sequence):
                    sen = [int_to_vocab[i.item()] for i in context]
                    cosine_similarity = calculate_cosine_similarity(sen, int_to_vocab[target_word_id.item()])
                    batch_cosine_similarity += cosine_similarity

                average_cosine_similarity = batch_cosine_similarity / len(input_sequence)
                total_cosine_similarity += average_cosine_similarity

            total_loss += loss.item()

            # Calculate accuracy
            accuracy = calculate_accuracy(output, target_sequence)
            total_accuracy += accuracy

            if i % show_every_n_batches == 0:
                print(f'Epoch: {epoch + 1}/{num_epochs}, Batch: {i}/{len(train_loader)}, Loss: {loss.item()}, '
                      f'Average Cosine Similarity: {average_cosine_similarity}, Accuracy: {accuracy}')

        average_epoch_cosine_similarity = total_cosine_similarity / len(train_loader)
        average_epoch_loss = total_loss / len(train_loader)
        average_epoch_accuracy = total_accuracy / len(train_loader)

        epoch_cosine_similarity_values.append(average_epoch_cosine_similarity)
        epoch_loss_values.append(average_epoch_loss)
        epoch_accuracy_values.append(average_epoch_accuracy)

        print(f'Epoch: {epoch + 1}/{num_epochs}, Train - '
              f'Average Cosine Similarity: {average_epoch_cosine_similarity}, '
              f'Average Loss: {average_epoch_loss}, '
              f'Average Accuracy: {average_epoch_accuracy}')

        for val_input_sequence, val_target_sequence in val_loader:
            val_output, _ = rnn.forward(val_input_sequence,init_states)
            val_loss = criterion(val_output, val_target_sequence)
            total_val_loss += val_loss.item()

            # Calculate accuracy
            val_accuracy = calculate_accuracy(val_output, val_target_sequence)
            total_val_accuracy += val_accuracy

        average_val_loss = total_val_loss / len(val_loader)
        average_val_accuracy = total_val_accuracy / len(val_loader)

        epoch_val_loss_values.append(average_val_loss)
        epoch_val_accuracy_values.append(average_val_accuracy)

        print(f'Epoch: {epoch + 1}/{num_epochs}, Validation - '
              f'Validation Loss: {average_val_loss}, Validation Accuracy: {average_val_accuracy}')

    return rnn, epoch_cosine_similarity_values, epoch_loss_values, epoch_accuracy_values, epoch_val_loss_values, epoch_val_accuracy_values


def generate_with_context(rnn, initial_seed_words, context_words, int_to_vocab, token_dict, pad_value, predict_len=100):
    
    
    generated_sentences = []
    seed_words = initial_seed_words.copy()

    for context_word_id in context_words:
        # Combine context word, <SEP>, and seed words
        current_seq = [context_word_id, vocab_to_int['<SEP>']] + seed_words
        predicted = [int_to_vocab[word_id] for word_id in current_seq[2:]]
        
        for _ in range(predict_len):
            #print(current_seq)
            current_seq = torch.LongTensor(current_seq).view(1, -1)
            
            # initialize the hidden state
            hidden = rnn.init_hidden(current_seq.size(0))
            
            # get the output of the rnn
            output, _ = rnn.forward(current_seq, hidden)
            
            # get the next word probabilities
            p = torch.softmax(output, dim=1).data.numpy().squeeze()
            
            # use top_k sampling to get the index of the next word
            top_k = 5
            top_i = np.argsort(p)[-top_k:]
            
            # select the likely next word index with some element of randomness
            word_i = np.random.choice(top_i, p=p[top_i]/p[top_i].sum())
            
            # retrieve that word from the dictionary
            word = int_to_vocab[word_i]
            predicted.append(word)     
        
            # update the current sequence for the next step
            current_seq = [context_word_id, vocab_to_int['<SEP>']] + [vocab_to_int[i] for i in predicted[-3:]]
        
        # Extract the generated sentence from the last 3 words
        generated_sentence = ' '.join(predicted)
        for key, token in token_dict.items():
            ending = ' ' if key in ['\n', '(', '"'] else ''
            generated_sentence = generated_sentence.replace(' ' + token.lower(), key)
            generated_sentence = generated_sentence.replace('\n', '')
            generated_sentence = generated_sentence.replace('( ', '(')
            
        generated_sentences.append([generated_sentence])
        
        # Update seed words for the next iteration
        seed_words = [vocab_to_int[i] for i in predicted[-3:]]
        
    
        
    return generated_sentences


sequence_length = 10  # of words in a sequence
# Batch Size
batch_size = 64

split_ratio = 0.8  # 80% training, 20% validation
split_index = int(len(int_text) * split_ratio)
train_data, val_data = random_split(int_text, [split_index, len(int_text) - split_index])
train_data_list = list(train_data)
val_data_list = list(val_data)

train_loader_base = batch_data_base(train_data_list, sequence_length, batch_size)
val_loader_base = batch_data_base(val_data_list, sequence_length, batch_size)

train_loader_context = batch_data_context(train_data_list, sequence_length, batch_size)
val_loader_context = batch_data_context(val_data_list, sequence_length, batch_size)

print("batch done")


num_epochs = 30
learning_rate = 0.0001
vocab_size = len(vocab_to_int)
output_size = vocab_size
embedding_dim = 200
hidden_dim = 250
n_layers = 2

show_every_n_batches = 20


rnn = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5)
rnn_context = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5)


# defining loss and optimization functions for training
#optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# training the model
trained_rnn, cos, losses, acc, val_losses, val_acc = train_rnn(rnn, batch_size, learning_rate, criterion, num_epochs, show_every_n_batches, train_loader_base, val_loader_base)
trained_rnn_context, cos_con, losses_con, acc_con, val_losses_con, val_acc_con = train_rnn(rnn_context, batch_size, learning_rate, criterion, num_epochs, show_every_n_batches, train_loader_context, val_loader_context)
# Plot the graph of cosine similarity values among the epochs


plt.figure(figsize=(18, 5))

# Plotting Cosine Similarity
plt.subplot(1, 3, 1)
plt.plot(range(1, num_epochs + 1), cos, marker='v', label='Model 1')
plt.plot(range(1, num_epochs + 1), cos_con, marker='o', label='Model 2')
plt.yticks([i * 0.05 for i in range(int(max(max(cos), max(cos_con)) / 0.05) + 2)])
plt.xlabel('Epoch')
plt.ylabel('Average Cosine Similarity')
plt.title('Cosine Similarity Across Epochs - Training')
plt.legend()

# Plotting Loss
plt.subplot(1, 3, 2)
plt.plot(range(1, num_epochs + 1), losses, marker='v', label='Model 1')
plt.plot(range(1, num_epochs + 1), losses_con, marker='o', label='Model 2')
plt.plot(range(1, num_epochs + 1), val_losses, marker='s', label='Model 1 - Validation')
plt.plot(range(1, num_epochs + 1), val_losses_con, marker='^', label='Model 2 - Validation')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Loss Across Epochs - Training and Validation')
plt.legend()

# Plotting Accuracy
plt.subplot(1, 3, 3)
plt.plot(range(1, num_epochs + 1), acc, marker='v', label='Model 1')
plt.plot(range(1, num_epochs + 1), acc_con, marker='o', label='Model 2')
plt.plot(range(1, num_epochs + 1), val_acc, marker='s', label='Model 1 - Validation')
plt.plot(range(1, num_epochs + 1), val_acc_con, marker='^', label='Model 2 - Validation')
plt.xlabel('Epoch')
plt.ylabel('Average Accuracy')
plt.title('Accuracy Across Epochs - Training and Validation')
plt.legend()

plt.tight_layout()
plt.show()

# Save the trained models
save_model('./save/trained_rnn', trained_rnn)
save_model('./save/trained_rnn_context', trained_rnn_context)
print('Models Trained and Saved')

gen_length = 50  # modify the length to your preference
initial_seed_words = [vocab_to_int['there'], vocab_to_int['was'], vocab_to_int['a']]
cont=['gandalf','ring','friends','snakes','book','home','king','hobbit']
context_words = [vocab_to_int[i] for i in cont]

generated_sentences = generate_with_context(trained_rnn, initial_seed_words, context_words, int_to_vocab, token_dict, vocab_to_int['<PAD>'], gen_length)
for sentence in generated_sentences:
    print(sentence)
    

print("Generating sentences using context")
generated_sentences_context = generate_with_context(trained_rnn_context, initial_seed_words, context_words, int_to_vocab, token_dict, vocab_to_int['<PAD>'], gen_length)
for sentence in generated_sentences_context:
    print(sentence)
