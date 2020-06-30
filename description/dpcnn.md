## Training DPCNNs with GULF

The GULF experiments with text classification in the Appendix of the GULF paper [Johnson & Zhang, 2020] used DPCNN [Johnson & Zhang, 2017] as a base model.  `train_yelp.py` and `train_yelp_embed.py` are provided to reproduce these experiments using the the polarized Yelp dataset from [Zhang et al.,2015].  Also note that using this code, regular training (i.e., without using GULF) of DPCNNs can also be done, and that with slight modification, other datasets can be used.  

**_Examples without embedding learning_**

* To perform GULF2 with 'ini:random' in the 'large' training data setting: `python3 train_yelp.py`
* To perform regular training in the 'large' training data setting: `python3 train_yelp.py --alpha 1 --num_stages 1`
* To get help: `python3 train_yelp.py -h`

**_Examples with embedding learning_**

DPCNNs optionally take additional features from embeddings trained on unlabeled data for a language-modeling-like objective.  To do this as in the GULF paper, 

1. To train the embedding of 3-word regions as a function of a bag of words to a 250-dim vector

        python3 train_yelp_embed.py
        
   The learned embedding is written to `emb/yelppol-n1r3-emb.pth`.

2. To train the embedding of 5-word regions as a function of a bag of {1,2,3}-grams to a 250-dim vector: 

        python3 train_yelp_embed.py  --n_max  3
        
   The learned embedding is written to `emb/yelppol-n3r5-emb.pth`.
        
3. To perform supervised training with GULF, using the embeddings obtained above

        python3 train_yelp.py  --x_emb  emb/yelppol-n3r5-emb.pth  emb/yelppol-n1r3-emb.pth
        
   To perform supervised training without GULF, using the embeddings obtained above        
   
        python3 train_yelp.py  --x_emb  emb/yelppol-n3r5-emb.pth  emb/yelppol-n1r3-emb.pth --alpha 1 --num_stages 1
        

**_Example configurations_**

code         | CPU cores     | CPU memory | GPU
------------ | ------------- | ---------- | ---
train_yelp.py       | 1  | 24GB | 1
train_yelp_embed.py | 7  | 32GB | 1

GPU device memory: 12GB

**_NOTES for DPCNN users_**

* This pyTorch version of DPCNN preserves the essence of DPCNN, but its details are not exactly the same as the DPCNN paper or the original C++ version.  This is a result of pursuing an efficient implementation in pyTorch and some simplification.  For example, the original work used the bag-of-word representation for target regions (to be predicted) and minimized squared error with negative sampling. This pyTorch version minimizes the log loss without sampling where the target probability is set by equally distributing the probability mass among the words in the target regions.  However, even after modifications, embedding learning in the pyTorch version is slower than the C++ version.  

* In the DPCNN paper, 5- and 9-word regions of uni-grams and {1,2,3}-grams were used.  Due to the above-mentioned changes in the embedding learning implementation, this choice may not be optimal for the datasets tested in the DPCNN paper.  The effective setting should be experimentally chosen newly for each dataset.  Our choice for the polarized Yelp is shown above.  

* This code downloads tokenized text (and labels, etc.) of the poloarized Yelp dataset.  To use DPCNNs on some other dataset, tokenized text (and labels, etc.) must be prepared by the user.  Please see the downloaded files at `data/` to find out the file format and file naming conventions.  You need to prepare `*.tok.txt` (tokens), `*.cat` (labels), and `*.catdic` (class label dictionary) at a minimum.  

**_Data source_**

This code downloads a tokenized version of the Yelp dataset.  The original Yelp dataset (before tokenization) was compiled by [Zhang et al., 2015].  

**_References_**

* [Johnson & Zhang, 2020] Guided Learning of Nonconvex Models through Successive Functional Gradient Optimization.  Rie Johnson and Tong Zhang.  ICML-2020.
* [Johnson & Zhang, 2017] Deep pyramid convolutional neural networks for text categorization.  Rie Johnson and Tong Zhang.  ACL-2017.  
* [Zhang et al., 2015] Character-level convolutional networks for text classification.  Xiang Zhang, Junbo Zhao, and Yann LeCun.  NIPS-2015. 
