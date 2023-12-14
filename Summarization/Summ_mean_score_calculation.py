import pandas as pd
import csv
from rouge_score import rouge_scorer
import evaluate
import json

rouge = []
rouge1 = []
rouge2 = []
rougeL = []
rougeLsum = []
bertprecision = []
bertrecall = []
bertf1 = []

with open('/Users/vareeshbainwala/Documents/Coursework/Codes/COMS6998/Summarization/Experiment_Results/results_indicbart_summ_hi.csv','r') as file:
    # fieldnames = ['Source','Prediction','Target','Meteor Score','Bleu Score']
    csvreader = csv.DictReader(file)
    for row in csvreader:
        # print(type(row['Rouge1']))
        rouge1.append(float(row['Rouge1']))
        rouge2.append(float(row['Rouge2']))
        rougeL.append(float(row['RougeL']))
        rougeLsum.append(float(row['RougeLsum']))
        bertprecision.append(float(row['Precision'].strip('[]')))
        bertrecall.append(float(row['Recall'].strip('[]')))
        bertf1.append(float(row['F1'].strip('[]')))

print("Calculating Means and Storing in File")

mean_r1 = sum(rouge1)/len(rouge1)
print(mean_r1)
mean_r2 = sum(rouge2)/len(rouge2)
mean_rl = sum(rougeL)/len(rougeL)
mean_rlsum = sum(rougeLsum)/len(rougeLsum)
mean_precision_bert_score = sum(bertprecision)/len(bertprecision)
mean_recall_bert_score = sum(bertrecall)/len(bertrecall)
mean_f1_bert_score = sum(bertf1)/len(bertf1)

with open('/Users/vareeshbainwala/Documents/Coursework/Codes/COMS6998/Summarization/Experiment_Results/scores_indicbart_summ_hi.csv','w') as f: 
    fieldnames = ['Mean Rouge-1','Mean Rouge-2','Mean Rouge-L','Mean Rouge-Lsum','Mean Precision','Mean Recall','Mean F1']
    csvwriter = csv.DictWriter(f,fieldnames=fieldnames)
    csvwriter.writeheader()
    csvwriter.writerow({'Mean Rouge-1': mean_r1,'Mean Rouge-2': mean_r2,'Mean Rouge-L': mean_rl,'Mean Rouge-Lsum': mean_rlsum,'Mean Precision': mean_precision_bert_score,
                        'Mean Recall': mean_recall_bert_score,'Mean F1': mean_f1_bert_score })
