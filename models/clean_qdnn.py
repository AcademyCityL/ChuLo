# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.autograd import Variable
import pytorch_lightning as pl
from customlayers.embedding import EmbeddingLayer
from tools.tokenizer import get_tokenizer

class CLQDNN(nn.Module):
    def __init__(self,config,params):
        super(CLQDNN, self).__init__()
        self.global_config = config
        self.params = params
        self.init_attr_from_config()
        self.init_model()

    def init_attr_from_config(self):
        model_config = self.global_config['MODEL']
        self.num_labels = model_config.get('num_labels',2)
        self.measurement_size = model_config.get('output_dim',256)
        self.dropout = model_config.get('dropout',0.1)
        self.embedding_params = model_config.get('embedding',{})
        self.mix = model_config.get('mix',False)

        data_config = self.global_config['DATA']
        self.tokenizer_type = data_config.get('tokenizer_type','non_bert')
        self.tokenizer_name = data_config.get('tokenizer_name','whitespace')
        self.tokenizer_real_name = 'files/tokenizers/{}/'.format(self.tokenizer_name)


    def init_model(self):
        # init embedding
        if self.tokenizer_type == 'non_bert':
            self.init_embedding = EmbeddingLayer(self.embedding_params['initialization'], vocab=self.params['vocab'],\
                    **self.embedding_params['kwargs'])
        elif self.tokenizer_type == 'bert':
            self.tokenizer = get_tokenizer('bert',self.tokenizer_real_name)
            self.init_embedding = EmbeddingLayer(self.embedding_params['initialization'], vocab=self.tokenizer.get_vocab(),\
                    **self.embedding_params['kwargs'])

        # feature extraction
        self.embedding_matrix = self.init_embedding.embedding.embs.weight
        # print("weight ",self.embedding_matrix.shape)
        self.embedding_dim = self.embedding_matrix.shape[1]
        self.complex_embed = ComplexEmbedding(self.embedding_matrix)
        self.l2_norm = L2Norm(dim = -1, keep_dims = True)
        self.l2_normalization = L2Normalization(dim = -1)
        self.activation = nn.Softmax(dim = 1)
        self.complex_multiply = ComplexMultiply()
        self.mixture = ComplexMixture(use_weights = True)
        self.measurement = ComplexMeasurement(self.embedding_dim, units = self.measurement_size)
    
    def forward(self, input, **kwargs):
        # print(input)
        amplitude_embedding, phase_embedding  = self.complex_embed(input)
#        phase_embedding = self.phase_embedding_layer(input_seq)
#        amplitude_embedding = self.amplitude_embedding_layer(input_seq)
        # print("amplitude_embedding  ",amplitude_embedding.shape,phase_embedding.shape)
        weights = self.l2_norm(amplitude_embedding)
        # print("after self.l2_norm",weights.shape)
        # print("weights  ",weights.shape)
        amplitude_embedding = self.l2_normalization(amplitude_embedding)
        # print("amplitude_embedding  ",amplitude_embedding.shape)
        weights = self.activation(weights)
        # print("after activation(weights)", weights.shape)
        [seq_embedding_real, seq_embedding_imag] = self.complex_multiply([phase_embedding, amplitude_embedding])
        # print("seq_embedding_real  ",seq_embedding_real.shape,seq_embedding_imag.shape)
        [sentence_embedding_real, sentence_embedding_imag] = self.mixture([seq_embedding_real, seq_embedding_imag,weights],self.mix)
        # print("sentence_embedding_real  ",sentence_embedding_real.shape,sentence_embedding_real.shape)
        # print("??????  ",sentence_embedding_real.shape,sentence_embedding_imag.shape)
        output = self.measurement([sentence_embedding_real, sentence_embedding_imag], measure_operator=None)
        return output

def PhaseEmbedding(input_dim, embedding_dim):
    embedding_layer = torch.nn.Embedding(input_dim, embedding_dim, padding_idx=0)
    torch.nn.init.uniform_(embedding_layer.weight, 0, 2*np.pi)
    return embedding_layer

def AmplitudeEmbedding(embedding_matrix, random_init=True):
    embedding_dim = embedding_matrix.shape[1]
    vocabulary_size = embedding_matrix.shape[0]
    if random_init:
        # Normal(0, 1)
        return torch.nn.Embedding(vocabulary_size,
                        embedding_dim,
#                        max_norm=1,
#                        norm_type=2,
                        padding_idx=0)
    else:
        return torch.nn.Embedding(vocabulary_size,
                        embedding_dim,
#                        max_norm=1,
#                        norm_type=2,
                        padding_idx=0,
                        _weight = embedding_matrix.clone().detach().requires_grad_(True))
#                        _weight=torch.tensor(embedding_matrix, dtype=torch.float))

class ComplexEmbedding(nn.Module):
    def __init__(self, embedding_matrix, freeze=False):
        super(ComplexEmbedding, self).__init__()
        sign_matrix = torch.sign(embedding_matrix)
        amplitude_embedding_matrix = sign_matrix * embedding_matrix
        self.amplitude_embed = nn.Embedding.from_pretrained(amplitude_embedding_matrix, freeze=freeze)
        phase_embedding_matrix = math.pi * (1 - sign_matrix) / 2 # based on [0, 2*pi]
        self.phase_embed = nn.Embedding.from_pretrained(phase_embedding_matrix, freeze=freeze)


    def forward(self, indices):
        amplitude_embed = self.amplitude_embed(indices)
        phase_embed = self.phase_embed(indices)

        return [amplitude_embed, phase_embed]

class L2Norm(nn.Module):

    def __init__(self, dim=1, keep_dims=True):
        super(L2Norm, self).__init__()
        self.dim = dim
        self.keepdim = keep_dims

    def forward(self, inputs):

        output = torch.sqrt(0.00001+ torch.sum(inputs**2, dim=self.dim, keepdim=self.keepdim))

        return output

class L2Normalization(nn.Module):

    def __init__(self, p=2, dim=1, eps=1e-12):
        super(L2Normalization, self).__init__()
        self.dim = dim
        self.p = p
        self.eps = eps

    def forward(self, inputs):
        # torch.nn.functional.normalize(input, p=2, dim=1, eps=1e-12, out=None)
        # v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}.
        output = F.normalize(inputs, p=self.p, dim=self.dim, eps=self.eps)
        return output

class ComplexMultiply(nn.Module):
    def __init__(self):
        super(ComplexMultiply, self).__init__()

    def forward(self, inputs):

        if not isinstance(inputs, list):
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.')

        if len(inputs) != 2:
            raise ValueError('This layer should be called '
                            'on a list of 2 inputs.'
                            'Got ' + str(len(inputs)) + ' inputs.')

        phase = inputs[0]
        amplitude = inputs[1]
        
        if amplitude.dim() == phase.dim()+1: # Assigning each dimension with same phase
            cos = torch.unsqueeze(torch.cos(phase), dim=-1)
            sin = torch.unsqueeze(torch.sin(phase), dim=-1)
            
        elif amplitude.dim() == phase.dim(): #Each dimension has different phases
            cos = torch.cos(phase)
            sin = torch.sin(phase)
        
       
        else:
             raise ValueError('input dimensions of phase and amplitude do not agree to each other.')

        real_part = cos*amplitude
        imag_part = sin*amplitude

        return [real_part, imag_part]

class ComplexMixture(torch.nn.Module):

    def __init__(self, use_weights=True):
        super(ComplexMixture, self).__init__()
        self.use_weights = use_weights

    def forward(self, inputs, mix = True):

        if not isinstance(inputs, list):
            raise ValueError('This layer should be called '
                             'on a list of 2/3 inputs.')

        if len(inputs) != 3 and len(inputs) != 2:
            raise ValueError('This layer should be called '
                            'on a list of 2/3 inputs.'
                            'Got ' + str(len(inputs)) + ' inputs.')

        input_real = torch.unsqueeze(inputs[0], dim=-1) 
        input_imag = torch.unsqueeze(inputs[1], dim=-1) 
        
        input_real_transpose = torch.unsqueeze(inputs[0], dim=-2) 
        input_imag_transpose = torch.unsqueeze(inputs[1], dim=-2) 


        # output = (input_real+i*input_imag)(input_real_transpose-i*input_imag_transpose)
        output_real = torch.matmul(input_real, input_real_transpose) + torch.matmul(input_imag, input_imag_transpose) #shape: (None, 60, 300, 300)
        output_imag = torch.matmul(input_imag, input_real_transpose) - torch.matmul(input_real, input_imag_transpose) #shape: (None, 60, 300, 300)
        
        embed_dim = output_real.shape[-1]
        if not self.use_weights and mix == True:
            output_r = torch.mean(output_real, dim=1)
            output_i = torch.mean(output_imag, dim=1)
        
        
        else:
            if inputs[2].dim() == inputs[1].dim()-1:
                weight = torch.unsqueeze(torch.unsqueeze(inputs[2], dim=-1), dim=-1)
            else:
                weight = torch.unsqueeze(inputs[2], dim=-1)
            
            output_real = output_real.float() * weight.float()
            output_r = output_real  #shape: (Batch, Seq, embdim, embdim)
            output_imag = output_imag.float() * weight.float()
            output_i = output_real  #shape: (Batch, Seq, embdim, embdim)
            if mix == True:
                output_r = torch.sum(output_real, dim=1)  #shape: (Batch, embdim, embdim)
                output_i = torch.sum(output_imag, dim=1)  #shape: (Batch, embdim, embdim)

        return [output_r, output_i]

class ComplexMeasurement(nn.Module):
    def __init__(self, embed_dim, units=5, ortho_init=False):
        super(ComplexMeasurement, self).__init__()
        self.units = units
        self.embed_dim = embed_dim
        if ortho_init:
            self.kernel = torch.nn.Parameter(torch.stack([torch.eye(embed_dim),torch.zeros(embed_dim, embed_dim)],dim = -1))

#            self.real_kernel = torch.nn.Parameter(torch.eye(embed_dim))
#            self.imag_kernel = torch.nn.Parameter(torch.zeros(embed_dim, embed_dim))
        else:
            rand_tensor = torch.rand(self.units, self.embed_dim, 2)
            normalized_tensor = F.normalize(rand_tensor.view(self.units, -1), p=2, dim=1, eps=1e-10).view(self.units, self.embed_dim, 2)
            self.kernel = torch.nn.Parameter(normalized_tensor)
#            self.kernel = F.normalize(self.kernel.view(self.units, -1), p=2, dim=1, eps=1e-10).view(self.units, embed_dim, 2)


#            self.real_kernel = torch.nn.Parameter(torch.Tensor(self.units, embed_dim))
#            self.imag_kernel = torch.nn.Parameter(torch.Tensor(self.units, embed_dim))

    def forward(self, inputs, measure_operator=None):
        
        input_real = inputs[0]
        input_imag = inputs[1]
        # print("input_real  ",input_real.shape,input_imag.shape)
        # print("self.kernel ",self.kernel.shape,measure_operator)
        real_kernel = self.kernel[:,:,0]
        imag_kernel = self.kernel[:,:,1]
        if measure_operator is None:
            real_kernel = real_kernel.unsqueeze(-1)
            imag_kernel = imag_kernel.unsqueeze(-1)
        else:
            real_kernel = measure_operator[0].unsqueeze(-1)
            imag_kernel = measure_operator[1].unsqueeze(-1)

        # print("real_kernel  ",real_kernel.shape,imag_kernel.shape)
        projector_real = torch.matmul(real_kernel, real_kernel.transpose(1, 2)) \
            + torch.matmul(imag_kernel, imag_kernel.transpose(1, 2))  
        projector_imag = torch.matmul(imag_kernel, real_kernel.transpose(1, 2)) \
            - torch.matmul(real_kernel, imag_kernel.transpose(1, 2))
        # print("projector_real  ",projector_real.shape,projector_imag.shape)
        # only real part is non-zero
        # input_real.shape = [batch_size, seq_len, embed_dim, embed_dim] or [batch_size, embed_dim, embed_dim]
        # projector_real.shape = [measurement_size, embed_dim, embed_dim]
        # flatten to get the tr(Pp(density))
        output_real = torch.matmul(torch.flatten(input_real, start_dim = -2, end_dim = -1),\
            torch.flatten(projector_real, start_dim = -2, end_dim = -1).t())\
            - torch.matmul(torch.flatten(input_imag, start_dim = -2, end_dim = -1), \
                torch.flatten(projector_imag, start_dim = -2, end_dim = -1).t())
    
        return output_real