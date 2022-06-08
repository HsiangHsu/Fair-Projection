# FACT: A Diagnostic for Group Fairness Trade-offs

This repository contains the code for the following [paper](https://arxiv.org/abs/2004.03424): 

> **FACT: A Diagnostic for Group Fairness Trade-offs**  
Joon Sik Kim, Jiahao Chen, Ameet Talwalkar  
To appear in *International Conference on Machine Learning (ICML)*, 2020.  

## Requirements

The code was developed and tested on Python 3.5.2. Other required libraries can be installed via running the command:

```  pip install -r requirements.txt ```

## Other papers/repositories used for comparisons

[Zafar et al. 2017](https://arxiv.org/abs/1507.05259) and its [code](https://github.com/mbilalzafar/fair-classification/tree/master/disparate_impact).

[Hardt et al. 2016](http://papers.nips.cc/paper/6373-equality-of-opportunity-in-supervised-learning) and its implementation [here](https://github.com/gpleiss/equalized_odds_and_calibration)

[Tan et al. 2020](https://arxiv.org/abs/1906.11813) and its [code](https://github.com/ztanml/fgp)

The output files from these algorithms used to plot in Jupyter notebooks are available [here](https://www.dropbox.com/sh/4xdow4fyr55jagt/AABcqkcqDUE2iTGMhyXFju79a?dl=0), along with the code portions (adpated from the above repositories) that generated them. Place all csv files in this link under the ```notebook``` directory to reproduce the plots. 
