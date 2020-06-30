## Ensemble of GULF-trained DPCNNs as in the Appendix of the GULF paper

NOTE: The training runs in each of the numbered steps below can be done simultaneously.  

**_To test the ensemble without embedding learning in the GULF paper_**
  
  1. Perform GULF training while saving models of each stage at `mod/`.  
  
            mkdir mod
            python3 train_yelp.py --ini_type iniRand --save mod/yelppol-noemb-iniRand.pth
            python3 train_yelp.py --ini_type iniBase --save mod/yelppol-noemb-iniBase.pth
        
  2. Make an ensemble of 10 models.   
  
            mods=
            for ini_type in iniRand iniBase; do
               for glc in 21 22 23 24 25; do
                  mods="$mods mod/yelp-noemb-${ini_type}-glc${glc}-slim.pth"
               done
            done

            python3 test_yelp_ensemble.py --model_paths $mods --x_emb $x_emb


**_To test the ensemble with embedding learning in the GULF paper_**

  1. Unsupervised embedding learning.  

            python3 train_yelp_embed.py            
     This generates `emb/yelppol-n1r3-emb.pth`. 
   
            python3 train_yelp_embed.py --n_max 3  
     This generates `emb/yelppol-n3r5-emb.pth`. 
  
  2. Supervised training with and without unsupervised embeddings while saving models of each stage at `mod/`.  
  
            mkdir mod
            python3 train_yelp.py --ini_type iniRand --save mod/yelppol-noemb-iniRand.pth
            python3 train_yelp.py --ini_type iniBase --save mod/yelppol-noemb-iniBase.pth
        
            x_emb="emb/yelppol-n3r5-emb.pth emb/yelppol-n1r3-emb.pth"
            python3 train_yelp.py --ini_type iniRand --save mod/yelppol-emb-iniRand.pth --x_emb $x_emb 
            python3 train_yelp.py --ini_type iniBase --save mod/yelppol-emb-iniBase.pth --x_emb $x_emb 
        
  3. Make an ensemble of 20 models.   

            #---  Assign id's "n3r5" and "n1r3" to each of the embedidngs, respectively.  
            x_emb="n3r5:emb/yelppol-n3r5-emb.pth n1r3:emb/yelppol-n1r3-emb.pth"   
            mods=
            for ini_type in iniRand iniBase; do
               for glc in 21 22 23 24 25; do
                  mods="$mods mod/yelp-noemb-${ini_type}-glc${glc}-slim.pth"
     
                  #---  Use "emb=..." to declare that this model was trained using the embeddings "n3r5" and "n1r3" in this order.  
                  mods="$mods emb=n3r5:emb=n1r3:mod/yelp-emb-${ini_type}-glc${glc}-slim.pth"   
               done
            done

            python3 test_yelp_ensemble.py --model_paths $mods --x_emb $x_emb
