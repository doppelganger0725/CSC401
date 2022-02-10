#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#  
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz

import numpy as np
import argparse
import json
import string 
import csv

# Provided wordlists.
FIRST_PERSON_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
SECOND_PERSON_PRONOUNS = {
    'you', 'your', 'yours', 'u', 'ur', 'urs'}
THIRD_PERSON_PRONOUNS = {
    'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them',
    'their', 'theirs'}
SLANG = {
    'smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff',
    'wyd', 'lylc', 'brb', 'atm', 'imao', 'sml', 'btw', 'bw', 'imho', 'fyi',
    'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys',
    'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya',
    'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml'}

FUTURE_TENSE = ["'ll","will", "gonna"]

COMMON_NOUNS = ["NN","NNS"]

PROPER_NOUNS = ["NNP","NNPS"]

ADVERBS = ["RB", "RBR", "RBS"]

WH_WORDS = ["WDT", "WP", "WP$", "WRB"]

PAST_TENSE = ["VBD"]

COORDINATING_CONJUN = ["CC"]

#Read wordlist CSV file to dict

Bristol = open("/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv")
Warriner = open("/u/cs401/Wordlists/Ratings_Warriner_et_al.csv")


reader1 = csv.reader(Bristol)
Bristol_dict = {}
for row in reader1:
    if row[1] != "WORD" and (row[1] != ""):
        Bristol_dict[row[1]] = {"AoA": float(row[3]), "IMG": float(row[4]), "FAM":float(row[5])}
reader2 = csv.reader(Warriner)
Warriner_dict = {}
for row in reader2:
    if row[1] != "Word":
        Warriner_dict[row[1]] = {"V.Mean.Sum": float(row[2]), "A.Mean.Sum": float(row[5]), "D.Mean.Sum": float(row[8])}

def all_punctuation(str):
    '''
    check if the token are punctuation entirely 
    '''
    for c in str:
        if c not in string.punctuation:
            return False
    return True



def extract1(comment):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''    
    # TODO: Extract features that rely on capitalization.
    # TODO: Lowercase the text in comment. Be careful not to lowercase the tags. (e.g. "Dog/NN" -> "dog/NN").
    # TODO: Extract features that do not rely on capitalization.

    uppercase = 0
    first_pro = 0
    second_pro = 0
    third_pro = 0
    coord_conj = 0
    past_tense = 0
    future_tense = 0
    commas = 0
    multi_pun = 0
    common_noun = 0
    proper_noun = 0
    adverb = 0
    wh_words = 0
    slang = 0

    # initialize numpy array 
    feat = np.zeros(173)
    #split the sentence by token
    word_tag_list = comment.split()
    for word_tag in word_tag_list:
        original_word = word_tag.split("/")[0]
        tag = word_tag.split("/")[1]
        #1. upper case check
        if original_word.isupper() and len(original_word) >=3:
            uppercase += 1
        word = original_word.lower()
        
        #Word feature check 
        #2-4,7,8,14
        if word in FIRST_PERSON_PRONOUNS:
            first_pro += 1
        elif word in SECOND_PERSON_PRONOUNS:
            second_pro += 1
        elif word in THIRD_PERSON_PRONOUNS:
            third_pro += 1
        elif word in FUTURE_TENSE:
            future_tense += 1
        elif word == ",":
            commas += 1
        elif word in SLANG:
            slang += 1
        
        #Tag feature check
        #5,6,10-13
        if tag in COORDINATING_CONJUN:
            coord_conj += 1
        elif tag in PAST_TENSE:
            past_tense += 1
        elif tag in COMMON_NOUNS:
            common_noun += 1
        elif tag in PROPER_NOUNS:
            proper_noun += 1
        elif tag in ADVERBS:
            adverb += 1
        elif tag in WH_WORDS:
            wh_words += 1
        
    

        #9. multi_char punc check
        if all_punctuation(word) and len(word) > 1:
            multi_pun += 1

    feat[0] = uppercase
    feat[1] = first_pro
    feat[2] = second_pro
    feat[3] = third_pro
    feat[4] = coord_conj
    feat[5] = past_tense
    feat[6] = future_tense
    feat[7] = commas
    feat[8] = multi_pun
    feat[9] = common_noun
    feat[10] = proper_noun
    feat[11] = adverb
    feat[12] = wh_words
    feat[13] = slang

    #15 Average length of sentence
    #16 Average length of token
    #17 Number of sentence
    sentence_list = comment.split("\n")[:-1]
    sentence_num = len(sentence_list)
    total_token_num = 0
    total_char = 0
    total_word_token = 0 
    for sentence in sentence_list:
        tokenlist = sentence.split()
        total_token_num += len(tokenlist)
        for token in tokenlist:
            word = token.split("/")[0]
            tag = token.split("/")[1]
            # if not punctuation. add to total char
            if not all_punctuation(word):
                total_word_token += 1
                total_char += len(word)
    sentence_avg_len = total_token_num / sentence_num
    token_avg_len = total_char / total_word_token
    feat[14] = sentence_avg_len
    feat[15] = token_avg_len
    feat[16] = sentence_num


    #18 - 29 check with wordlist
    Bris_AOA_data = []
    Bris_IMG_data = []
    Bris_FAM_data = []
    Warr_V_data = []
    Warr_A_data = [] 
    Warr_D_data = []  
    for word_tag in word_tag_list:
        o_word = word_tag.split("/")[0]
        word = o_word.lower()        
        if word in Bristol_dict.keys():
            Bris_AOA_data.append(Bristol_dict[word]["AoA"])
            Bris_IMG_data.append(Bristol_dict[word]["IMG"])
            Bris_FAM_data.append(Bristol_dict[word]["FAM"])
        if word in Warriner_dict.keys():
            Warr_V_data.append(Warriner_dict[word]["V.Mean.Sum"])
            Warr_A_data.append(Warriner_dict[word]["A.Mean.Sum"])
            Warr_D_data.append(Warriner_dict[word]["D.Mean.Sum"])
    if len(Bris_AOA_data) != 0:
        feat[17] = np.mean(Bris_AOA_data)
        feat[20] = np.std(Bris_AOA_data)
    else:
        feat[17] = 0
        feat[20] = 0
    if len(Bris_IMG_data) != 0:
        feat[18] = np.mean(Bris_IMG_data)
        feat[21] = np.std(Bris_IMG_data)
    else:
        feat[18] = 0
        feat[21] = 0   
    if len(Bris_FAM_data) != 0:    
        feat[19] = np.mean(Bris_FAM_data)
        feat[22] = np.std(Bris_FAM_data)
    else:
        feat[19] = 0
        feat[22] = 0
    if len(Warr_V_data) != 0:
        # print(Warr_V_data)
        feat[23] = np.mean(Warr_V_data)
        feat[26] = np.std(Warr_V_data)
    else:
        feat[23] = 0
        feat[26] = 0
    if len(Warr_A_data) != 0:
        # print(Warr_A_data)
        feat[24] = np.mean(Warr_A_data)
        feat[27] = np.std(Warr_A_data)
    else:
        feat[24] = 0
        feat[27] = 0
    if len(Warr_D_data) != 0:
        # print(Warr_D_data)
        feat[25] = np.mean(Warr_D_data)
        feat[28] = np.std(Warr_D_data)
    else:
        feat[25] = 0
        feat[28] = 0

    return feat
    

#Read txt file to an id: index dict
def FiletoDict(file):
    '''
    transfer ID_txt to id: index dict 
    '''
    index = 0
    dict = {}
    f = open(file,'r')
    lines = f.readlines()
    for line in lines:
        dict[line.rstrip()] = index
        index += 1
    return dict

# Read txt file

Alt_dict = FiletoDict("/u/cs401/A1/feats/Alt_IDs.txt")
Center_dict = FiletoDict("/u/cs401/A1/feats/Center_IDs.txt")
Left_dict = FiletoDict("/u/cs401/A1/feats/Left_IDs.txt")
Right_dict = FiletoDict("/u/cs401/A1/feats/Right_IDs.txt")

# load npy file
Alt_np = np.load("/u/cs401/A1/feats/Alt_feats.dat.npy")
Center_np = np.load("/u/cs401/A1/feats/Center_feats.dat.npy")
Left_np = np.load("/u/cs401/A1/feats/Left_feats.dat.npy")
Right_np = np.load("/u/cs401/A1/feats/Right_feats.dat.npy")


def extract2(feat, comment_class, comment_id):
    ''' This function adds features 30-173 for a single comment.

    Parameters:
        feat: np.array of length 173
        comment_class: str in {"Alt", "Center", "Left", "Right"}
        comment_id: int indicating the id of a comment

    Returns:
        feat : numpy Array, a 173-length vector of floating point features (this 
        function adds feature 30-173). This should be a modified version of 
        the parameter feats.
    '''
    index = 0
    if comment_class == "Alt":
        index = Alt_dict[comment_id]
        feat[29:173] = Alt_np[index]
    elif comment_class == "Center":
        index = Center_dict[comment_id]
        feat[29:173] = Center_np[index]
    elif comment_class == "Left":
        index = Left_dict[comment_id]
        feat[29:173] = Left_np[index]
    elif comment_class == "Right":
        index = Right_dict[comment_id]
        feat[29:173] = Right_np[index]
    return feat


def main(args):
    #Declare necessary global variables here. 

    #Load data
    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173+1))

    for i in range(len(data)):
        #the i th comment
        j = data[i]
        feat = extract1(j["body"])
        full_feat = extract2(feat, j["cat"], j["id"])
        if j["cat"] == "Left":
            final_feat = np.append(full_feat,0)
        elif j["cat"] == "Center":
            final_feat = np.append(full_feat,1)
        elif j["cat"] == "Right":
            final_feat = np.append(full_feat,2)
        elif j["cat"] == "Alt":
            final_feat = np.append(full_feat,3) 
        # print(final_feat)
        feats[i] = final_feat 
    # TODO: Call extract1 for each datatpoint to find the first 29 features. 
    # Add these to feats.
    # TODO: Call extract2 for each feature vector to copy LIWC features (features 30-173)
    # into feats. (Note that these rely on each data point's class,
    # which is why we can't add them in extract1).
    # print('TODO')

    np.savez_compressed(args.output, feats)

    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    parser.add_argument("-p", "--a1_dir", help="Path to csc401 A1 directory. By default it is set to the cdf directory for the assignment.", default="/u/cs401/A1/")
    args = parser.parse_args()        

    main(args)

