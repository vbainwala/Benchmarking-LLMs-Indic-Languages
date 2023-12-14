import pandas as pd
import csv
from rouge_score import rouge_scorer
import evaluate
import json

meteor = []
bleu = []

with open('/Users/vareeshbainwala/Documents/Coursework/Codes/COMS6998/Machine_Translation/Experiment_Results/results_hi_en_backtranslation.csv','r') as file:
    # fieldnames = ['Source','Prediction','Target','Meteor Score','Bleu Score']
    csvreader = csv.DictReader(file)
    for row in csvreader:
        # print(type(row['Rouge1']))
        # print(row['Meteor Score'])
        meteor.append(float(row['Meteor Score']))
        bleu.append(float(row['Bleu Score']))

print("Calculating Means and Storing in File")

mean_meteor = sum(meteor)/len(meteor)
mean_bleu = sum(bleu)/len(bleu)
with open('/Users/vareeshbainwala/Documents/Coursework/Codes/COMS6998/Machine_Translation/Experiment_Results/scores_mbart_mt_hi_bt.csv','w') as f: 
    fieldnames = ['Mean Meteor', 'Mean Bleu']
    csvwriter = csv.DictWriter(f,fieldnames=fieldnames)
    csvwriter.writeheader()
    csvwriter.writerow({'Mean Meteor':mean_meteor, 'Mean Bleu': mean_bleu })
