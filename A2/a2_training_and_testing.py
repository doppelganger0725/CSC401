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

'''Functions related to training and testing.

You don't need anything more than what's been imported here.
'''

from tqdm import tqdm
import typing

import torch

import a2_bleu_score
import a2_dataloader
import a2_encoder_decoder


def train_for_epoch(
        model: a2_encoder_decoder.EncoderDecoder,
        dataloader: a2_dataloader.HansardDataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device) -> float:
    '''Train an EncoderDecoder for an epoch

    An epoch is one full loop through the training data. This function:

    1. Defines a loss function using :class:`torch.nn.CrossEntropyLoss`,
       keeping track of what id the loss considers "padding"
    2. For every iteration of the `dataloader` (which yields triples
       ``F, F_lens, E``)
       1. Sends ``F`` to the appropriate device via ``F = F.to(device)``. Same
          for ``F_lens`` and ``E``.
       2. Zeros out the model's previous gradient with ``optimizer.zero_grad()``
       3. Calls ``logits = model(F, F_lens, E)`` to determine next-token
          probabilities.
       4. Modifies ``E`` for the loss function, getting rid of a token and
          replacing excess end-of-sequence tokens with padding using
        ``model.get_target_padding_mask()`` and ``torch.masked_fill``
       5. Flattens out the sequence dimension into the batch dimension of both
          ``logits`` and ``E``
       6. Calls ``loss = loss_fn(logits, E)`` to calculate the batch loss
       7. Calls ``loss.backward()`` to backpropagate gradients through
          ``model``
       8. Calls ``optim.step()`` to update model parameters
    3. Returns the average loss over sequences

    Parameters
    ----------
    model : EncoderDecoder
        The model we're training.
    dataloader : HansardDataLoader
        Serves up batches of data.
    device : torch.device
        A torch device, like 'cpu' or 'cuda'. Where to perform computations.
    optimizer : torch.optim.Optimizer
        Implements some algorithm for updating parameters using gradient
        calculations.

    Returns
    -------
    avg_loss : float
        The total loss divided by the total numer of sequence
    '''
    # If you want, instead of looping through your dataloader as
    # for ... in dataloader: ...
    # you can wrap dataloader with "tqdm":
    # for ... in tqdm(dataloader): ...
    # This will update a progress bar on every iteration that it prints
    # to stdout. It's a good gauge for how long the rest of the epoch
    # will take. This is entirely optional - we won't grade you differently
    # either way.
    # If you are running into CUDA memory errors part way through training,
    # try "del F, F_lens, E, logits, loss" at the end of each iteration of
    # the loop.
    # assert False, "Fill me"
    total_loss = 0
    number_sequence = 0
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index = model.source_pad_id)
    for F, F_lens, E in tqdm(dataloader):
        # step1
        F = F.to(device)
        F_lens = F_lens.to(device)
        E = E.to(device)
        # step2
        optimizer.zero_grad()
        # step3
        logits = model(F, F_lens, E)
        # step4
        mask = model.get_target_padding_mask(E)
        E= torch.masked_fill(E,mask,-1)
        # step5
        #   logits is of shape (T - 1, M, Vo)
        # flatten logit to (T-1 *M, V)
        flat_logits = torch.flatten(logits, start_dim=0, end_dim=1)
        # T-1
        E = E[1:,:]
        # (T-1 *M,-1)
        E = torch.flatten(E).unsqueeze(-1)
        # step6
        loss = loss_fn(flat_logits, E)
        # step7
        loss.backward()
        # step8 
        optimizer.step()
        batch_size = F_lens.size()[0]
        number_sequence += batch_size
        total_loss += loss.item()
    del F, F_lens, E, logits, loss
    
    avg_loss = total_loss / number_sequence
    return avg_loss


def parse(list, sos, eos):
    # remove eos, sos
    result = []
    for index in list:
        if index != eos and index != sos:
            result.append(index)
    return result 


def compute_batch_total_bleu(
        E_ref: torch.LongTensor,
        E_cand: torch.LongTensor,
        target_sos: int,
        target_eos: int) -> float:
    '''Compute the total BLEU score over elements in a batch

    Parameters
    ----------
    E_ref : torch.LongTensor
        A batch of reference transcripts of shape ``(T, M)``, including
        start-of-sequence tags and right-padded with end-of-sequence tags.
    E_cand : torch.LongTensor
        A batch of candidate transcripts of shape ``(T', M)``, also including
        start-of-sequence and end-of-sequence tags.
    target_sos : int
        The ID of the start-of-sequence tag in the target vocabulary.
    target_eos : int
        The ID of the end-of-sequence tag in the target vocabulary.

    Returns
    -------
    total_bleu : float
        The sum total BLEU score for across all elements in the batch. Use
        n-gram precision 4.
    '''
    # you can use E_ref.tolist() to convert the LongTensor to a python list
    # of numbers
    # assert False, "Fill me"
    Total_score = 0
    M = E_ref.size()[1]
    # loop over batch and finds the bleu sccore for each sentence in batch 
    for m in range(M):
        E_ref_list = E_ref[:,m].tolist()
        E_candid_list = E_cand[:,m].tolist()
        parse_E_ref = parse(E_ref_list,target_sos,target_eos)
        parse_E_cand = parse(E_candid_list,target_sos,target_eos)
        Total_score += a2_bleu_score.BLEU_score(parse_E_ref,parse_E_cand,4)
    return Total_score



def compute_average_bleu_over_dataset(
        model: a2_encoder_decoder.EncoderDecoder,
        dataloader: a2_dataloader.HansardDataLoader,
        target_sos: int,
        target_eos: int,
        device: torch.device) -> float:
    '''Determine the average BLEU score across sequences

    This function computes the average BLEU score across all sequences in
    a single loop through the `dataloader`.

    1. For every iteration of the `dataloader` (which yields triples
       ``F, F_lens, E_ref``):
       1. Sends ``F`` to the appropriate device via ``F = F.to(device)``. Same
          for ``F_lens``. No need for ``E_cand``, since it will always be
          compared on the CPU.
       2. Performs a beam search by calling ``b_1 = model(F, F_lens)``
       3. Extracts the top path per beam as ``E_cand = b_1[..., 0]``
       4. Computes the total BLEU score of the batch using
          :func:`compute_batch_total_bleu`
    2. Returns the average per-sequence BLEU score

    Parameters
    ----------
    model : EncoderDecoder
        The model we're testing.
    dataloader : HansardDataLoader
        Serves up batches of data.
    target_sos : int
        The ID of the start-of-sequence tag in the target vocabulary.
    target_eos : int
        The ID of the end-of-sequence tag in the target vocabulary.

    Returns
    -------
    avg_bleu : float
        The total BLEU score summed over all sequences divided by the number of
        sequences
    '''
    # assert False, "Fill me"
    total_bleu = 0
    number_sequence = 0
    for F, F_lens, E_ref in tqdm(dataloader):
        # step1
        F = F.to(device)
        F_lens = F_lens.to(device)
        # step2
        b_1 = model(F, F_lens)
        # step3
        E_cand = b_1[:,:, 0]
        # step4
        total_bleu += compute_batch_total_bleu(E_ref,E_cand,target_sos,target_eos)
        batch_size = F_lens.size()[0]
        number_sequence += batch_size
        
    avg_bleu = total_bleu / number_sequence    
    return avg_bleu

