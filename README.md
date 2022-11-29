# Text-Classification-Based-Approach-for-Evaluating-and-Enhancing-Machine-Interpretability-of-Building
--*author: zhengzhe*  
--*date: 2022.10.26*
 - Description: Code and dataset for the paper named "Text Classification-Based Approach for Evaluating and Enhancing Machine Interpretability of Building Codes".  
 - This is a Pytorch-based Bert Chinese text classification approach.   
 - Bert models thanks to [Bert-Chinese-Text-Classification-Pytorch](https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch)
 - Other models thanks to [Chinese-Text-Classification-Pytorch](https://github.com/649453932/Chinese-Text-Classification-Pytorch)

## environment

python 3.7  
torch 1.12.1+cu116
boto3 1.24.28  
matplotlib 3.5.3  
tqdm  
sklearn  
tensorboardX  

## Dataset 
 - Description: Chinese rule dataset including seven categories are established to classify the interpretability level of each rule in a building code  
 - The original labeled dataset can be found in [CivilRules/dataset](./CivilRules/dataset)
 - The training, validation, and test dataset can be found in [CivilRules/data](./CivilRules/data)

| Category  | Definition | Interpretability |
|-----------|------------|------------------|
| direct    | The required information is explicitly available from the BIM model      | Easy             |
| indirect  | The required information is implicitly stored in the BIM model. A set of derivations and calculations should be performed.      | Easy             |
| method    | An extended data structure and domain-specific knowledge are required.      | Medium           |
| reference | The external information, including pictures, formulas, tables, and other rules or appendices in the current code or other codes, is required.      | Medium           |
| general   | The rules provide macro design guidance.      | Hard             |
| term      | The rules define the terms used in the codes.      | Hard             |
| other     | The rules do not belong to the above six categories.      | Hard             |


## Models
| model        | Weighted F1 score |
|--------------|-------------------|
| TextCNN      | 86.3%             |
| TextRNN      | 72.2%             |
| TextRNN-Att  | 81.5%             |
| Transformers | 74.0%             |
| Bert         | 88.04%            |
| RuleBERT     | 93.68%            |

### Further pretrained domain-specific models
- The original Bert model can be found in [google drive](https://drive.google.com/drive/folders/1v_eplluVNWjBvrnzdzzw4AKcUz4GNarr?usp=share_link)  
  - Please put the original Bert model in *./bert_pretrain*
- The further pretrained domain-specific Bert model (RuleBERT) can be found in [google drive](https://drive.google.com/drive/folders/1t1MJ0DEVz6B_usqNfC2CGP-eb7oh88qr?usp=share_link)
  - Please put the RuleBERT model in *./bert_pretraindc*

### Finetune BERT models  
- The well trained BERT models (.ckpt files) can be found in [google drive](https://drive.google.com/drive/folders/1O16omrOoPgAsU8rFmjOFWkEtUj8o0aT6?usp=share_link)
- Please put these models in *./CivilRules/save_dict*

### Well-trained other models  
- The well-trained models (TextCNN, TextRNN, TextRNN-Att, Transformers) can be found in [google drive](https://drive.google.com/drive/folders/1GbfGO7m3crM3U7XMTbgRX78S_2VtUJIr?usp=share_link) 
- Reproduce the result can use the code from [Chinese-Text-Classification-Pytorch](https://github.com/649453932/Chinese-Text-Classification-Pytorch)

## How to use
### Validate the BERT model results using well fine-tuned model
 - assert the bert models and the finetune models have been put into the right place 
 - put test dataset (test.txt) in to ./CivilRules/data
```cmd
# validate the bert model weighted F1 score
python test.py --model bert
# validate the RuleBERT model weighted F1 score
python test.py --model bertDC
```
### Train your own model using grid_search to find the best model
 - prepare your own test dataset in to ./CivilRules/data
 - modify the dataset, learning_rates, batch_sizes in grid_search.py
```cmd
# to finetune bert model
python grid_search.py --model bert
# to finetune RuleBERT model
python grid_search.py --model bertDC
```
### Predict with the well-trained BERT model
 - prepare your own prediction dataset (predict.txt) and named it to dev.txt, and then put it in to ./CivilRules/data
 - modify the dataset in application.py
 - prepare well-trained bert model in to ./CivilRules/save_dict
```cmd
python application.py --model bert
python application.py --model bertDC
```
- the result will be saved in ./CivilRules/predict