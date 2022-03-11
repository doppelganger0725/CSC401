'''
This code is provided solely for the personal and private use of students
taking the CSC401H/2511H course at the University of Toronto. Copying for
purposes other than this use is expressly prohibited. All forms of
distribution of this code, including but not limited to public repositories on
GitHub, GitLab, Bitbucket, or any other online platform, whether as given or
with any changes, are expressly prohibited.

Authors: Sean Robertson, Jingcheng Niu, Zining Zhu, and Mohamed Abdall
Updated by: Raeid Saqur <raeidsaqur@cs.toronto.edu>

All of the files in this directory and all subdirectories are:
Copyright (c) 2022 University of Toronto
'''

'''Concrete implementations of abstract base classes.

You don't need anything more than what's been imported here
'''

from tkinter import E
from turtle import backward
from unittest import result
import torch
from typing import Optional, Union, Tuple, Type, Set

from a2_abcs import EncoderBase, DecoderBase, EncoderDecoderBase

# All docstrings are omitted in this file for simplicity. So please read
# a2_abcs.py carefully so that you can have a solid understanding of the
# structure of the assignment.

class Encoder(EncoderBase):


    def init_submodules(self):
        # Hints:
        # 1. You must initialize the following submodules:
        #   self.rnn, self.embedding
        # 2. You will need the following object attributes:
        #   self.source_vocab_size, self.word_embedding_size,
        #   self.pad_id, self.dropout, self.cell_type,
        #   self.hidden_state_size, self.num_hidden_layers.
        # 3. cell_type will be one of: ['lstm', 'gru', 'rnn']
        # 4. Relevant pytorch modules: torch.nn.{LSTM, GRU, RNN, Embedding}
        # assert False, "Fill me"
        self.embedding = torch.nn.Embedding(num_embeddings=self.source_vocab_size,embedding_dim = self.word_embedding_size,padding_idx = self.pad_id)
        if self.cell_type == 'lstm':
            self.rnn = torch.nn.LSTM(input_size=self.word_embedding_size,hidden_size=self.hidden_state_size,num_layers=self.num_hidden_layers,dropout=self.dropout,bidirectional = True)
        if self.cell_type == 'gru':
            self.rnn = torch.nn.GRU(input_size=self.word_embedding_size,hidden_size=self.hidden_state_size,num_layers=self.num_hidden_layers,dropout=self.dropout,bidirectional = True)
        if self.cell_type == 'rnn':
            self.rnn = torch.nn.RNN(input_size=self.word_embedding_size,hidden_size=self.hidden_state_size,num_layers=self.num_hidden_layers,dropout=self.dropout,bidirectional = True)
    
    def forward_pass(
            self,
            F: torch.LongTensor,
            F_lens: torch.LongTensor,
            h_pad: float = 0.) -> torch.FloatTensor:
        # Recall:
        #   F is shape (S, M)
        #   F_lens is of shape (M,)
        #   h_pad is a float
        #
        # Hints:
        # 1. The structure of the encoder should be:
        #   input seq -> |embedding| -> embedded seq -> |rnn| -> seq hidden
        # 2. You will need to use the following methods:
        #   self.get_all_rnn_inputs, self.get_all_hidden_states
        # assert False, "Fill me"
        embedding = self.get_all_rnn_inputs(F)
        seq_hidden = self.get_all_hidden_states(embedding,F_lens,h_pad)
        return seq_hidden

    def get_all_rnn_inputs(self, F: torch.LongTensor) -> torch.FloatTensor:
        # Recall:
        #   F is shape (S, M)
        #   x (output) is shape (S, M, I)
        # assert False, "Fill me"
        x = self.embedding(F)
        return x

    def get_all_hidden_states(
            self, 
            x: torch.FloatTensor,
            F_lens: torch.LongTensor,
            h_pad: float) -> torch.FloatTensor:
        # Recall:
        #   x is of shape (S, M, I)
        #   F_lens is of shape (M,)
        #   h_pad is a float
        #   h (output) is of shape (S, M, 2 * H)
        #
        # Hint:
        #   relevant pytorch modules:
        #   torch.nn.utils.rnn.{pad_packed,pack_padded}_sequence
        # assert False, "Fill me"
        packed_embeds = torch.nn.utils.rnn.pack_padded_sequence(input=x,lengths=F_lens,enforce_sorted=False)
        rnn_output, rnn_hidden = self.rnn(packed_embeds)
        padded_output, lengths = torch.nn.utils.rnn.pad_packed_sequence(rnn_output,padding_value=h_pad)
        return padded_output


class DecoderWithoutAttention(DecoderBase):
    '''A recurrent decoder without attention'''

    def init_submodules(self):
        # Hints:
        # 1. You must initialize the following submodules:
        #   self.embedding, self.cell, self.ff
        # 2. You will need the following object attributes:
        #   self.target_vocab_size, self.word_embedding_size, self.pad_id
        #   self.hidden_state_size, self.cell_type.
        # 3. cell_type will be one of: ['lstm', 'gru', 'rnn']
        # 4. Relevant pytorch modules:
        #   torch.nn.{Embedding, Linear, LSTMCell, RNNCell, GRUCell}
        # assert False, "Fill me"
        self.embedding = torch.nn.Embedding(num_embeddings=self.target_vocab_size,embedding_dim = self.word_embedding_size,padding_idx = self.pad_id)
        if self.cell_type == 'lstm':
            self.cell = torch.nn.LSTMCell(input_size=self.word_embedding_size,hidden_size = self.hidden_state_size)
        if self.cell_type == 'gru':
            self.cell = torch.nn.GRUCell(input_size=self.word_embedding_size,hidden_size = self.hidden_state_size)
        if self.cell_type == 'rnn':
            self.cell = torch.nn.RNNCell(input_size=self.word_embedding_size,hidden_size = self.hidden_state_size)
     
        self.ff = torch.nn.Linear(in_features=self.hidden_state_size,out_features = self.target_vocab_size )

    def forward_pass(
        self,
            E_tm1: torch.LongTensor,
            htilde_tm1: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor,
            F_lens: torch.LongTensor) -> Tuple[
                torch.FloatTensor, Union[
                    torch.FloatTensor,
                    Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # Recall:
        #   E_tm1 is of shape (M,)
        #   htilde_tm1 is of shape (M, 2 * H)
        #   h is of shape (S, M, 2 * H)
        #   F_lens is of shape (M,)
        #   logits_t (output) is of shape (M, V)
        #   htilde_t (output) is of same shape as htilde_tm1
        #
        # Hints:
        # 1. The structure of the encoder should be:
        #   encoded hidden -> |embedding| -> embedded hidden -> |rnn| ->
        #   decoded hidden -> |output layer| -> output logits
        # 2. You will need to use the following methods:
        #   self.get_current_rnn_input, self.get_current_hidden_state,
        #   self.get_current_logits
        # 3. You can assume that htilde_tm1 is not empty. I.e., the hidden state
        #   is either initialized, or t > 1.
        # 4. The output of an LSTM cell is a tuple (h, c), but a GRU cell or an
        #   RNN cell will only output h.
        # assert False, "Fill me"
        xtilde_t = self.get_current_rnn_input(E_tm1,htilde_tm1,h,F_lens)
        htilde_t = self.get_current_hidden_state(xtilde_t,htilde_tm1)
        # LSTM case h only
        logits_t = None
        if self.cell_type == "lstm":
            logits_t = self.get_current_logits(htilde_t[0])
        else:
            logits_t = self.get_current_logits(htilde_t)
        return[logits_t,htilde_t]


    def get_first_hidden_state(
            self,
            h: torch.FloatTensor,
            F_lens: torch.LongTensor) -> torch.FloatTensor:
        # Recall:
        #   h is of shape (S, M, 2 * H)
        #   F_lens is of shape (M,)
        #   htilde_tm1 (output) is of shape (M, 2 * H)
        #
        # Hint:
        # 1. Ensure it is derived from encoder hidden state that has processed
        # the entire sequence in each direction. You will need to:
        # - Populate indices [0: self.hidden_state_size // 2] with the hidden
        #   states of the encoder's forward direction at the highest index in
        #   time *before padding*
        # - Populate indices [self.hidden_state_size//2:self.hidden_state_size]
        #   with the hidden states of the encoder's backward direction at time
        #   t=0
        # 2. Relevant pytorch function: torch.cat
        # assert False, "Fill me"
        forward = h[(F_lens-1), :, 0:self.hidden_state_size // 2]
        backward = h[0,:,self.hidden_state_size//2:self.hidden_state_size]
        htilde_0 = torch.cat([forward, backward], dim=1)
        return htilde_0 

    def get_current_rnn_input(
            self,
            E_tm1: torch.LongTensor,
            htilde_tm1: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor,
            F_lens: torch.LongTensor) -> torch.FloatTensor:
        # Recall:
        #   E_tm1 is of shape (M,)
        #   htilde_tm1 is of shape (M, 2 * H) or a tuple of two of those (LSTM)
        #   h is of shape (S, M, 2 * H)
        #   F_lens is of shape (M,)
        #   xtilde_t (output) is of shape (M, Itilde)
        # assert False, "Fill me"
        xtilde_t = self.embedding(E_tm1)
        return xtilde_t

    def get_current_hidden_state(
            self,
            xtilde_t: torch.FloatTensor,
            htilde_tm1: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]]) -> Union[
                    torch.FloatTensor,
                    Tuple[torch.FloatTensor, torch.FloatTensor]]:
        # Recall:
        #   xtilde_t is of shape (M, Itilde)
        #   htilde_tm1 is of shape (M, 2 * H) or a tuple of two of those (LSTM)
        #   htilde_t (output) is of same shape as htilde_tm1
        # assert False, "Fill me" 
        if self.cell_type == "lstm":
            htilde_tm1 = (htilde_tm1[0],htilde_tm1[1])
        htilde_t = self.cell(xtilde_t,htilde_tm1)
        return htilde_t

    def get_current_logits(
            self,
            htilde_t: torch.FloatTensor) -> torch.FloatTensor:
        # Recall:
        #   htilde_t is of shape (M, 2 * H), even for LSTM (cell state discarded)
        #   logits_t (output) is of shape (M, V)
        # assert False, "Fill me"
        logits_t = self.ff(htilde_t)
        return logits_t


class DecoderWithAttention(DecoderWithoutAttention):
    '''A decoder, this time with attention

    Inherits from DecoderWithoutAttention to avoid repeated code.
    '''

    def init_submodules(self):
        # Hints:
        # 1. Same as the case without attention, you must initialize the
        #   following submodules: self.embedding, self.cell, self.ff
        # 2. You will need the following object attributes:
        #   self.target_vocab_size, self.word_embedding_size, self.pad_id
        #   self.hidden_state_size, self.cell_type.
        # 3. cell_type will be one of: ['lstm', 'gru', 'rnn']
        # 4. Relevant pytorch modules:
        #   torch.nn.{Embedding, Linear, LSTMCell, RNNCell, GRUCell}
        # 5. The implementation of this function should be different from
        #   DecoderWithoutAttention.init_submodules.
        # assert False, "Fill me"

        # input size = c_t size + embedding size = hiddenstate + embed 
        self.embedding = torch.nn.Embedding(num_embeddings=self.target_vocab_size,embedding_dim = self.word_embedding_size,padding_idx = self.pad_id)
        if self.cell_type == 'lstm':
            self.cell = torch.nn.LSTMCell(input_size=self.word_embedding_size + self.hidden_state_size,hidden_size = self.hidden_state_size)
        if self.cell_type == 'gru':
            self.cell = torch.nn.GRUCell(input_size=self.word_embedding_size + self.hidden_state_size,hidden_size = self.hidden_state_size)
        if self.cell_type == 'rnn':
            self.cell = torch.nn.RNNCell(input_size=self.word_embedding_size + self.hidden_state_size,hidden_size = self.hidden_state_size)
     
        self.ff = torch.nn.Linear(in_features=self.hidden_state_size,out_features = self.target_vocab_size )


    def get_first_hidden_state(
            self,
            h: torch.FloatTensor,
            F_lens: torch.LongTensor) -> torch.FloatTensor:
        # Hint: For this time, the hidden states should be initialized to zeros.
        # assert False, "Fill me"
        # shape(S, M, 2 * H)
        shape = h.size()
        # scale (M, 2 * H)
        scale = shape[1:]
        zero_tensor = torch.zero(scale)
        return zero_tensor

    def get_current_rnn_input(
            self,
            E_tm1: torch.LongTensor,
            htilde_tm1: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor,
            F_lens: torch.LongTensor) -> torch.FloatTensor:
        # Hint: Use attend() for c_t
        # assert False, "Fill me"
        T_e = self.embedding(E_tm1)
        c_tm1 = self.attend(htilde_tm1,h,F_lens)
        # cat c_tm1（M,T） with T_e (M,Itilde) on dim 1
        xtilde_t = torch.cat([c_tm1,T_e], dim=1)
        return xtilde_t
    
    def attend(
            self,
            htilde_t: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor,
            F_lens: torch.LongTensor) -> torch.FloatTensor:
        '''The attention mechanism. Calculate the context vector c_t.

        Parameters
        ----------
        htilde_t : torch.FloatTensor or tuple
            Like `htilde_tm1` (either a float tensor or a pair of float
            tensors), but matching the current hidden state.
        h : torch.FloatTensor
            A float tensor of shape ``(S, M, self.hidden_state_size)`` of
            hidden states of the encoder. ``h[s, m, i]`` is the
            ``i``-th index of the encoder RNN's last hidden state at time ``s``
            of the ``m``-th sequence in the batch. The states of the
            encoder have been right-padded such that ``h[F_lens[m]:, m]``
            should all be ignored.
        F_lens : torch.LongTensor
            An integer tensor of shape ``(M,)`` corresponding to the lengths
            of the encoded source sentences.

        Returns
        -------
        c_t : torch.FloatTensor
            A float tensor of shape ``(M, self.hidden_state_size)``. The
            context vector c_t is the product of weights alpha_t and h.

        Hint: Use get_attention_weights() to calculate alpha_t.
        '''
        # assert False, "Fill me"
        alpha_t = self.get_attention_weights(htilde_t,h,F_lens)
        # unsqueeze S, M into S, M, 1
        unsqueeze_alpha = alpha_t.unsqueeze(2)
        # multiple the alpha and h
        mul_alpha = torch.mul(unsqueeze_alpha, h)
        # sum up in S dim
        c_t = torch.sum(mul_alpha,dim = 0)
        return c_t

    def get_attention_weights(
            self,
            htilde_t: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor,
            F_lens: torch.LongTensor) -> torch.FloatTensor:
        # DO NOT MODIFY! Calculates attention weights, ensuring padded terms
        # in h have weight 0 and no gradient. You have to implement
        # get_attention_scores()
        # alpha_t (output) is of shape (S, M)
        e_t = self.get_attention_scores(htilde_t, h)
        pad_mask = torch.arange(h.shape[0], device=h.device)
        pad_mask = pad_mask.unsqueeze(-1) >= F_lens.to(h.device)  # (S, M)
        e_t = e_t.masked_fill(pad_mask, -float('inf'))
        return torch.nn.functional.softmax(e_t, 0)

    def get_attention_scores(
            self,
            htilde_t: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor) -> torch.FloatTensor:
        # Recall:
        #   htilde_t is of shape (M, 2 * H)
        #   h is of shape (S, M, 2 * H)
        #   e_t (output) is of shape (S, M)
        #
        # Hint:
        # Relevant pytorch function: torch.nn.functional.cosine_similarity
        # assert False, "Fill me"
        e_t = torch.nn.functional.cosine_similarity(htilde_t,h)
        return e_t

class DecoderWithMultiHeadAttention(DecoderWithAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.W is not None, 'initialize W!'
        assert self.Wtilde is not None, 'initialize Wtilde!'
        assert self.Q is not None, 'initialize Q!'

    def init_submodules(self):
        super().init_submodules()  # Do not change this line

        # Hints:
        # 1. The above line should ensure self.ff, self.embedding, self.cell are
        #    initialized
        # 2. You need to initialize the following submodules:
        #       self.W, self.Wtilde, self.Q
        # 3. You will need the following object attributes:
        #       self.hidden_state_size
        # 4. self.W, self.Wtilde, and self.Q should process all heads at once. They
        #    should not be lists!
        # 5. You do *NOT* need self.heads at this point
        # 6. Relevant pytorch module: torch.nn.Linear (note: set bias=False!)
        # assert False, "Fill me"
        self.W = torch.nn.Linear(in_features = self.hidden_state_size,out_features = self.hidden_state_size,bias=False)
        self.Wtilde = torch.nn.Linear(in_features = self.hidden_state_size,out_features = self.hidden_state_size,bias=False)
        self.Q = torch.nn.Linear(in_features = self.hidden_state_size,out_features = self.hidden_state_size,bias=False)
    def attend(
            self,
            htilde_t: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor,
            F_lens: torch.LongTensor) -> torch.FloatTensor:
        # Hints:
        # 1. You can use super().attend to call for the regular attention
        #   function.
        # 2. Relevant pytorch function:
        #   tensor().view, tensor().repeat_interleave
        # 3. Fun fact:
        #   tensor([1,2,3,4]).repeat(2) will output tensor([1,2,3,4,1,2,3,4]).
        #   tensor([1,2,3,4]).repeat_interleave(2) will output
        #   tensor([1,1,2,2,3,3,4,4]), just like numpy.repeat.
        # 4. You *WILL* need self.heads at this point
        # assert False, "Fill me"
        # shape(S, M, 2 * H)
        
        shape = h.size()
        S = shape[0]
        M = shape[1]
        Hidden = shape[2]
        h_s = self.W(h)
        h_tilde = self.Wtilde(htilde_t)
        # Change shape of htilde_t 
        trans_htilde_t = h_tilde.view(-1,Hidden//self.heads) 
        # Change shape of h 
        trans_h = h_s.view(S,-1,Hidden//self.heads)
        trans_F_len = F_lens.repeat_interleave(self.heads)
        c_t = super().attend(trans_htilde_t,trans_h,trans_F_len)
        Q_c_tm1 = self.Q(c_t.view(M,Hidden))
        return Q_c_tm1

class EncoderDecoder(EncoderDecoderBase):

    def init_submodules(
            self,
            encoder_class: Type[EncoderBase],
            decoder_class: Type[DecoderBase]):
        # Hints:
        # 1. You must initialize the following submodules:
        #   self.encoder, self.decoder
        # 2. encoder_class and decoder_class inherit from EncoderBase and
        #   DecoderBase, respectively.
        # 3. You will need the following object attributes:
        #   self.source_vocab_size, self.source_pad_id,
        #   self.word_embedding_size, self.encoder_num_hidden_layers,
        #   self.encoder_hidden_size, self.encoder_dropout, self.cell_type,
        #   self.target_vocab_size, self.target_eos, self.heads
        # 4. Recall that self.target_eos doubles as the decoder pad id since we
        #   never need an embedding for it
        # assert False, "Fill me"
        self.encoder = encoder_class(self.source_vocab_size,self.source_pad_id, self.word_embedding_size,self.encoder_num_hidden_layers,self.encoder_hidden_size,self.encoder_dropout,self.cell_type)
        #bidirection hidden size *2 
        self.decoder = decoder_class(self.target_vocab_size,self.target_eos,self.word_embedding_size,self.encoder_hidden_size * 2,self.cell_type,self.heads)

    def get_logits_for_teacher_forcing(
            self,
            h: torch.FloatTensor,
            F_lens: torch.LongTensor,
            E: torch.LongTensor) -> torch.FloatTensor:
        # Recall:
        #   h is of shape (S, M, 2 * H)
        #   F_lens is of shape (M,)
        #   E is of shape (T, M)
        #   logits (output) is of shape (T - 1, M, Vo)
        #
        # Hints:
        # 1. Relevant pytorch modules: torch.{zero_like, stack}
        # 2. Recall an LSTM's cell state is always initialized to zero.
        # 3. Note logits sequence dimension is one shorter than E (why?)
        # assert False, "Fill me"
        logits_list = []
        htilde_tm1 = None
        # dimension t - 1
        Time = E.size()[0] - 1
        # for each time stamp in E 
        for t in range(Time):
            logits_t,htilde_tm1 = self.decoder.forward(E[t],htilde_tm1,h,F_lens)
            
            logits_list.append(logits_t)
        logits = torch.stack(logits_list)
        return logits

    def update_beam(
            self,
            htilde_t: torch.FloatTensor,
            b_tm1_1: torch.LongTensor,
            logpb_tm1: torch.FloatTensor,
            logpy_t: torch.FloatTensor) -> Tuple[
                torch.FloatTensor, torch.LongTensor, torch.FloatTensor]:
        # perform the operations within the psuedo-code's loop in the
        # assignment.
        # You do not need to worry about which paths have finished, but DO NOT
        # re-normalize logpy_t.
        #
        # Recall
        #   htilde_t is of shape (M, K, 2 * H) or a tuple of two of those (LSTM)
        #   logpb_tm1 is of shape (M, K)
        #   b_tm1_1 is of shape (t, M, K)
        #   b_t_0 (first output) is of shape (M, K, 2 * H) or a tuple of two of
        #      those (LSTM)
        #   b_t_1 (second output) is of shape (t + 1, M, K)
        #   logpb_t (third output) is of shape (M, K)
        #
        # Hints:
        # 1. Relevant pytorch modules:
        #   torch.{flatten, topk, unsqueeze, expand_as, gather, cat}
        # 2. If you flatten a two-dimensional array of shape z of (A, B),
        #   then the element z[a, b] maps to z'[a*B + b]
        # assert False, "Fill me"
        
        
        M = logpy_t.size()[0]
        K = logpy_t.size()[1]
        V = logpy_t.size()[2]
        # reshape logpb_tm1 to M, K, V
        extend_logpb_tm1 = logpb_tm1.unsqueeze(-1) 
        # get full path to find top k in logpb_tm1 + logpy_t 
        path = extend_logpb_tm1 + logpy_t
        # flatten the full paths (M,K*V)
        full_path = torch.flatten(path,start_dim=1)
        # find the topk in the flatten full path
        # top_tensor and top_tensor_index with dim(M,K)
        top_value, top_tensor_index = torch.topk(full_path, k=K, dim=1)
        #update probability value
        logpb_t = top_value
        # find path k to keep
        k_path = torch.div(top_tensor_index, V,rounding_mode='floor')
        # find the best word to keep
        k_vocab = top_tensor_index % V
        # reshape k_path to b_tm1_1 (T,M,K)
        reshape_k_path = k_path.unsqueeze(0).expand_as(b_tm1_1)
        # reshape k_vocab to b_tm1_1 (T,M,K)
        reshape_k_vocab = k_vocab.unsqueeze(0)
        # gather index of those k_path in b_tm1_1 
        gather_path = b_tm1_1.gather(2,reshape_k_path)
        # Cat vocab on each time dimension
        b_t_1 = torch.cat([gather_path,reshape_k_vocab],dim=0)
        
        # udpate b_t_0
        # LSTM case
        if self.cell_type == "lstm":
            hidden = htilde_t[0]
            cell = htilde_t[1]
            hidden_index = k_path.unsqueeze(2).expand_as(hidden)
            cell_index = k_path.unsqueeze(2).expand_as(cell)
            b_t_0 = (hidden.gather(1,hidden_index),cell.gather(1,cell_index)) 
        else:
            # add dimension 2H to k_path and make it same as htilde
            index_tensor = k_path.unsqueeze(2).expand_as(htilde_t)
            b_t_0 = htilde_t.gather(1,index_tensor)
        
        return [b_t_0, b_t_1,logpb_t]
