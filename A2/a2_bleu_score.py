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

'''Calculate BLEU score for one reference and one hypothesis

You do not need to import anything more than what is here
'''

from math import exp
from sqlite3 import SQLITE_DROP_INDEX  # exp(x) gives e^x
from typing import List, Sequence, Iterable


def grouper(seq:Sequence[str], n:int) -> List:
    '''Extract all n-grams from a sequence

    An n-gram is a contiguous sub-sequence within `seq` of length `n`. This
    function extracts them (in order) from `seq`.

    Parameters
    ----------
    seq : sequence
        A sequence of words or token ids representing a transcription.
    n : int
        The size of sub-sequence to extract.

    Returns
    -------
    ngrams : list
    '''
    s_index = 0
    e_index = len(seq) - n
    result = [] 
    while s_index <= e_index:
        group = []
        for i in range(n):
            group.append(seq[s_index + i])
        result.append(group)
        s_index += 1
    return result
    # assert False, "Fill me"


def n_gram_precision(reference:Sequence[str], candidate:Sequence[str], n:int) -> float:
    '''Calculate the precision for a given order of n-gram

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)
    n : int
        The order of n-gram precision to calculate

    Returns
    -------
    p_n : float
        The n-gram precision. In the case that the candidate has length 0,
        `p_n` is 0.
    '''
    ref_group = grouper(reference,n)
    can_group = grouper(candidate,n)
    match = 0
    for item in can_group:
        if item in ref_group:
            match += 1
    
    if len(candidate) == 0 or n > len(candidate):
        return 0
    else:
        return match / len(can_group)
    # assert False, "Fill me"


def brevity_penalty(reference:Sequence[str], candidate:Sequence[str]) -> float:
    '''Calculate the brevity penalty between a reference and candidate

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)

    Returns
    -------
    BP : float
        The brevity penalty. In the case that the candidate transcription is
        of 0 length, `BP` is 0.
    '''
    ref_len = len(reference)
    can_len = len(candidate)
    if len(candidate) == 0 :
        return 0
    else:
        brevity = ref_len / can_len
        if brevity < 1:
            return 1
        else:
            return exp(1-brevity)
    # assert False, "Fill me"


def BLEU_score(reference:Sequence[str], candidate:Sequence[str], n) -> float:
    '''Calculate the BLEU score

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)
    n : int
        The maximum order of n-gram precision to use in the calculations,
        inclusive. For example, ``n = 2`` implies both unigram and bigram
        precision will be accounted for, but not trigram.

    Returns
    -------
    bleu : float
        The BLEU score
    '''
    BP = brevity_penalty(reference,candidate)
    P_multiple = 1
    # calculate each n gram
    for i in range(1,n+1):
        value = n_gram_precision(reference,candidate,i)
        P_multiple = P_multiple * value
    
    BLEU = BP * (P_multiple ** (1/n))
    return BLEU
    # assert False, "Fill me"
