import torch
import torch.nn as nn
import zipfile
import numpy as np
import torch.nn.functional as F
import io

class BaseModel(nn.Module):
    def __init__(self, args, vocab, tag_size):
        super(BaseModel, self).__init__()
        self.args = args
        self.vocab = vocab
        self.tag_size = tag_size

    def save(self, path):
        # Save model
        print(f'Saving model to {path}')
        ckpt = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(ckpt, path)

    def load(self, path):
        # Load model
        print(f'Loading model from {path}')
        ckpt = torch.load(path)
        self.vocab = ckpt['vocab']
        self.args = ckpt['args']
        self.load_state_dict(ckpt['state_dict'])


def load_embedding(vocab, emb_file, emb_size, embed_weights):
    """
    Read embeddings for words in the vocabulary from the emb_file (e.g., GloVe, FastText).
    Args:
        vocab: (Vocab), a word vocabulary
        emb_file: (string), the path to the embdding file for loading
        emb_size: (int), the embedding size (e.g., 300, 100) depending on emb_file
    Return:
        emb: (np.array), embedding matrix of size (|vocab|, emb_size) 
    """
    word2vec = {}
    with open(emb_file) as f:
        for line in f:
            values = line.split()
            word = values[0]
            vec = torch.tensor([float(x) for x in values[1:]]).float()
            word2vec[word] = vec
    embedding_matrix = torch.zeros((len(vocab), emb_size))
    for i, word in enumerate(vocab.word2id):
        if word in word2vec:
            embedding_matrix[i,:] = word2vec[word]
        else:
            embedding_matrix[i,:] = embed_weights[i];
    return embedding_matrix


class DanModel(BaseModel):
    def __init__(self, args, vocab, tag_size):
        super(DanModel, self).__init__(args, vocab, tag_size)
        self.vocab = vocab
        self.nword = len(vocab)
        self.tag_size = tag_size
        v = 0.05
        self.define_model_parameters(args)
        self.init_model_parameters(v)
        

        # Use pre-trained word embeddings if emb_file exists
        if args.emb_file is not None:
            self.copy_embedding_from_numpy(args)

    def define_model_parameters(self, args):
        """
        Define the model's parameters, e.g., embedding layer, feedforward layer.
        #0.3
        """
        self.embed = nn.Embedding(self.nword, args.emb_size)
        self.bn1 = nn.BatchNorm1d(args.emb_size)
        self.dropout1 = nn.Dropout(args.hid_drop)
        self.fc1 = nn.Linear(args.emb_size, args.emb_size)
        self.bn2 = nn.BatchNorm1d(args.emb_size)
        self.dropout2 = nn.Dropout(args.hid_drop)
        self.fc2 = nn.Linear(args.emb_size, args.emb_size)
        self.bn3 = nn.BatchNorm1d(args.emb_size)
        self.fc3 = nn.Linear(args.emb_size, self.tag_size)
        self.dropout3 = nn.Dropout(args.hid_drop)


    def init_model_parameters(self, v):
        """
        Initialize the model's parameters by uniform sampling from a range [-v, v], e.g., v=0.08
        """
        nn.init.xavier_uniform_(self.embed.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.uniform_(self.fc1.bias,a=-v, b=v)
        nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.uniform_(self.fc2.bias,a=-v, b=v)
        torch.nn.init.uniform_(self.fc3.weight,a=-v, b=v)
        torch.nn.init.uniform_(self.fc3.bias,a=-v, b=v)



    def copy_embedding_from_numpy(self, args):
        """
        Load pre-trained word embeddings from numpy.array to nn.embedding
        """
        embedding_matrix = load_embedding(self.vocab , args.emb_file, args.emb_size, self.embed.weight)
        self.embed.weight.data.copy_(embedding_matrix)

    def forward(self, x):
        """
        Compute the unnormalized scores for P(Y|X) before the softmax function.
        E.g., feature: h = f(x)
              scores: scores = w * h + b
              P(Y|X) = softmax(scores)  
        Args:
            x: (torch.LongTensor), [batch_size, seq_length]
        Return:
            scores: (torch.FloatTensor), [batch_size, ntags]
        """
        
        masks = []
        
        for tensor in torch.unbind(x):
            mask = [1 if tensor[i] !=  self.vocab['<pad>'] else 0 for i in range(tensor.size(0))]
            masks.append(mask)
        masks = torch.LongTensor(masks)
        x = self.embed(x)
        if(masks != None):
            x = (masks.unsqueeze(dim=2) * x)
        x = x.sum(dim=1)/masks.sum(dim=1).unsqueeze(dim=1)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.bn2(x)
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.bn3(x)
        x = self.dropout3(x)
        x = self.fc3(x)

        return x;

