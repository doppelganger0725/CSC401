import os
from tkinter import INSERT
import numpy as np
import sys 
import re

dataDir = '/u/cs401/A3/data/'

def Levenshtein(r, h):
    """                                                                         
    Calculation of WER with Levenshtein distance.                               
                                                                                
    Works only for iterables up to 254 elements (uint8).                        
    O(nm) time ans space complexity.                                            
                                                                                
    Parameters                                                                  
    ----------                                                                  
    r : list of strings                                                                    
    h : list of strings                                                                   
                                                                                
    Returns                                                                     
    -------                                                                     
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively
                                                                                
    Examples                                                                    
    --------                                                                    
    >>> wer("who is there".split(), "is there".split())                         
    0.333 0 0 1                                                                           
    >>> wer("who is there".split(), "".split())                                 
    1.0 0 0 3                                                                           
    >>> wer("".split(), "who is there".split())                                 
    Inf 0 3 0                                                                           
    """
    n = len(r)
    m = len(h)
    if n == 0:
        return float("inf"), 0, m, 0
    if m == 0:
        return 1,0,0,n
    
    #initialize matrix
    r_len = n+1
    h_len = m+1
    R = np.zeros((r_len,h_len))
    #  back track matrix 
    # 1 = match 2 = deletion 3 = substituition, 4 = insertion 
    back_matrix = np.zeros((r_len,h_len))

    for i in range(r_len):
        R[i,0] = i
    
    for j in range(h_len):
        R[0,j] = j

    for i in range(1,r_len):
        for j in range(1,h_len):
            # if word match 
            if r[i-1] == h[j-1]:
                R[i,j] = R[i-1, j-1]
                back_matrix[i,j] = 1
            # not match
            else:
                min_value = min(R[i-1,j],R[i-1,j-1],R[i,j-1])
                R[i,j] = min_value + 1
                # deletion
                if min_value == R[i-1,j]:
                    back_matrix[i,j] = 2
                # substitution
                elif min_value == R[i-1,j-1]:
                    back_matrix[i,j] = 3
                # insertion
                else:
                    back_matrix[i,j] = 4
    wer = R[n,m]/n
    deletion = 0
    substituition = 0
    insertion = 0
    i, j = n,m
    print("backmatrix/n")
    print(back_matrix)
    while not(i == 0 and j == 0):
        # word match
        if back_matrix[i,j] == 1:
            i -= 1
            j -= 1
        elif back_matrix[i,j] == 2:
            deletion += 1
            i -= 1
        elif back_matrix[i,j] == 3:
            substituition += 1
            i -= 1
            j -= 1
        elif back_matrix[i,j] == 4:
            insertion += 1
            j -= 1
        else:
            print("error")

    
    return wer, substituition, insertion, deletion


def preprocess(line):
    cleanr = re.compile('<.*?>')
    # remove tags
    no_tag = re.sub(cleanr, '', line)
    # remove punctuation
    no_puch = re.sub(r'[^\w\s]', '', no_tag)
    
    # lwoer case and remove the leading token
    return no_puch.lower().split()[2:]
    



def readfile(path):
    # read file
    f = open(path,'r')
    line = f.readlines()
    f.close()
    return line


if __name__ == "__main__":
    google_wer = []
    kaldi_wer = []
    with open(os.path.join(sys.path[0], "asrDiscussion.txt"), "w") as f:
        for root, dirs, files in os.walk(dataDir):
            for speaker in dirs:
                Reference = readfile(os.path.join(root, speaker, "transcripts.txt"))
                Google  = readfile(os.path.join(root, speaker, "transcripts.Google.txt"))
                Kaldi = readfile(os.path.join(root, speaker, "transcripts.Kaldi.txt"))
                
                for i in range (len(Reference)):
                    R_script = preprocess(Reference[i])
                    G_script = preprocess(Google[i])
                    K_script = preprocess(Kaldi[i])
                    gwer, gnS, gnI, gnD = Levenshtein(R_script,G_script)
                    google_wer.append(gwer)

                    f.write("{} Google {} WER{} S:{}, I:{}, D:{}".format(speaker,i,gwer,gnS,gnI,gnD))
                    
                    kwer, knS, knI, knD = Levenshtein(R_script,K_script)
                    kaldi_wer.append(kwer)

                    f.write("{} Kaldi {} WER{} S:{}, I:{}, D:{}".format(speaker,i,kwer,knS,knI,knD))
        f.write("Google mean: {}, std: {}".format(np.mean(google_wer),np.std(google_wer)))
        f.write("Kaldi mean: {}, std: {}".format(np.mean(kaldi_wer),np.std(kaldi_wer)))
