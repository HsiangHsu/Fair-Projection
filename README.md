## FairProjection

The official code of **Beyond Adult and COMPAS: Fairness in Multi-Class Prediction via Information Projection (NeurIPS 2022 Oral)** [[arXiv]](https://arxiv.org/abs/2206.07801)

#### `data/` contains all datasets
- `UCI-Adult/`: raw data <ins> adult.data</ins> , <ins> adult.names</ins> , <ins> adult.test</ins>  [1].
- `COMPAS/`: raw data <ins> compas-scores-two-years.csv</ins>  [2]
- `HSLS/`: k-NN imputed HSLS dataset [3] (Raw data and pre-processing: https://drive.google.com/drive/folders/14Ke1fiB5RKOVlA8iU9aarAeJF0g4SdBl)
- `ENEM/`: downsampled pre-processed data downloaded from https://download.inep.gov.br/microdados/microdados_enem_2020.zip

#### `fair-projection/` contains implementations of FairProjection
- Python packages and environment are included in **fairprojection.yml**.
- command to run: *python3 <ins> run_mp.py</ins>*
  - <ins> GroupFair.py</ins> and <ins> coreMP.py</ins> contain the core ADMM implementations of FairProjection.
  - Python functions <ins>load_data</ins> loads UCI-Adult and COMPAS datasets into PANDAS DataFrames.
  - Python functions <ins>load_hsls_imputed</ins> loads the HSLS dataset into PANDAS DataFrames.
  - Python functions <ins>load_enem</ins> loads the ENEM dataset into PANDAS DataFrames with flexible numbers of samples, classes, and groups.
- `hsls/`: 
- `enem/`: experiments 
  - `acc-fairness-tradeoff/` for accuracy-fairness tradeoff curves with different fairness budgets. 
  - `runtime/` for parallel/ non-parallel runtime comparison.
  - `multi-group-multi-class/`: for multiple-group and multi-class FairProjection.

#### `baseline-methods/` contains baseline models: EqOdds [4], CalEqOdds [5], LevEqOpp [6], Reduction [7], Rejection [8], and [[FACT]](https://github.com/wnstlr/FACT) [9]
- Python packages and environment are included in **baseline.yml**.
- command to run: *python3 <ins> benchmark.py</ins>  -m [model name] -f [fair method] -c [constraint] -n [num iter] -i [inputfile] -s [seed]*
- Options for arguments:
  - [model name]: gbm, logit, rf (Default: gbm)
  - [fair method]: reduction, eqodds, roc (Default: reduction)
  - [constraint]: eo, sp, (Default: eo)
  - [num iter]: Any positive integer (Default: 10) 
  - [inputfile]: hsls, enem-20000, enem-50000, ...  (Default: hsls)
  - [seed]: Any integer (Default: 42)
- Results of this program will be saved in `results/` folder with the file name of the form `[fair method]_[model name]_s[seed]_[constraint].pkl`. You can open the result file by running the following command: *import pickle; result = pickle.load(open([filename], 'rb’))*

#### `leveraging-python/` contains Python implementation of LevEqOpp [6]. The original [[implementation was]](https://github.com/lucaoneto/NIPS2019_Fairness) in R language. 

#### `results/`: store all computation results as pickle files.

#### Citation

```
@inproceedings{alghamdi2022beyond,
  title={Beyond Adult and {\{}COMPAS{\}}: Fairness in Multi-Class Prediction via Information Projection},
  author={Alghamdi, Wael and Hsu, Hsiang and Jeong, Haewon and Wang, Hao and Michalak, P Winston and Asoodeh, Shahab and Calmon, Flavio P},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2022}
}
```

#### Reference
[1] M. Lichman. UCI machine learning repository, 2013.

[2] Julia Angwin, Jeff Larson, Surya Mattu, and Lauren Kirchner. Machine bias. ProPublica, 2016.

[3] Ingels, S. J., Pratt, D. J., Herget, D. R., Burns, L. J., Dever, J. A., Ottem, R., Rogers, J. E., Jin, Y., and Leinwand, S. (2011). High school longitudinal study of 2009 (hsls: 09): Base-year data file documentation. nces 2011-328. National Center for Education Statistics.

[4] Moritz Hardt, Eric Price, and Nati Srebro. Equality of opportunity in supervised learning. Advances in neural information processing systems, 29:3315–3323, 2016.

[5] Geoff Pleiss, Manish Raghavan, Felix Wu, Jon Kleinberg, and Kilian Q Weinberger. On fairness and calibration. arXiv preprint arXiv:1709.02012, 2017.

[6] Evgenii Chzhen, Christophe Denis, Mohamed Hebiri, Luca Oneto, and Massimiliano Pontil. Leveraging labeled and unlabeled data for consistent fair binary classification. Advances in Neural Information Processing Systems, 32, 2019.

[7] Alekh Agarwal, Alina Beygelzimer, Miroslav Dudík, John Langford, and Hanna Wallach. A reductions approach to fair classification. In International Conference on Machine Learning, pages 60–69. PMLR, 2018.

[8] F. Kamiran, A. Karim, and X. Zhang. Decision theory for discrimination-aware classification. In 2012 IEEE 12th International Conference on Data Mining, pages 924–929, Dec 2012.

[9] Joon Sik Kim, Jiahao Chen, and Ameet Talwalkar. FACT: A diagnostic for group fairness trade-offs. In International Conference on Machine Learning, pages 5264–5274. PMLR, 2020.

