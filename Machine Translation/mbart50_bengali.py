from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import evaluate
import pandas as pd
import nltk
import numpy as np
import csv
from nltk.translate import meteor_score,bleu_score
#nltk.download('punkt')

df = pd.read_parquet('/Users/vareeshbainwala/Documents/Coursework/Codes/COMS6998/Dataset/MT/0000_bengali.parquet')
source = df["src"].values.tolist()
target = df['tgt'].values.tolist()

# START: COPIED FROM https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
# END: COPIED FROM https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt

meteor = evaluate.load('meteor')
predictions = []
meteor_score_per_input = []
#bleu = evaluate.load("bleu")
bleu_score_per_input = []
for i in range(100):
    article_en = source[i]
    #Translate English to Hindi
    # START: COPIED FROM https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt
    tokenizer.src_lang = "en_XX"
    encoded_en = tokenizer(article_en, return_tensors="pt")
    generated_tokens = model.generate(**encoded_en, forced_bos_token_id=tokenizer.lang_code_to_id["bn_IN"])
    # END: COPIED FROM https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt
    predictions.append(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
    
for i in range(100):
    results = meteor.compute(predictions=predictions[i], references=[target[i]])
    #bleu_results = bleu.compute(predictions = [predictions[i][0].split(' ')],references=[[target[i].split(' ')]])
    bleu = bleu_score.sentence_bleu(references=[target[i].split(' ')],hypothesis=predictions[i][0].split(' '),weights=(0.5,0.5))
    meteor_score_per_input.append(results)
    bleu_score_per_input.append(bleu)

with open('/Users/vareeshbainwala/Documents/Coursework/Codes/COMS6998/Machine_Translation/Experiment_Results/results_en_bn.csv','w') as f: 
    fieldnames = ['Source','Prediction','Target','Meteor Score','Bleu Score']
    csvwriter = csv.DictWriter(f,fieldnames=fieldnames)
    csvwriter.writeheader()
    # csvwriter.writerow('Source[Prediction[Target[Meteor Score[Bleu Score')
    for i in range(100):  
        csvwriter.writerow({'Source':source[i],'Prediction':predictions[i],'Target': target[i], 'Meteor Score':meteor_score_per_input[i]['meteor'],'Bleu Score': bleu_score_per_input[i]})


""" Backward Translation to check if the performance in the backward is good or not and then further compare this with the IndicBart-XXEN model"""
source_backward = predictions
target_backward = source

predictions_backward = []

for i in range(len(source_backward)):
    # article_en = "The weather today is great."
    article_en = source_backward[i]
    #Translate Bengali to English
    # START: COPIED FROM https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt
    tokenizer.src_lang = "bn_IN"
    encoded_en = tokenizer(article_en, return_tensors="pt")
    generated_tokens = model.generate(**encoded_en, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
    # END: COPIED FROM https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt
    predictions_backward.append(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))


for i in range(len(source_backward)):
    results = meteor.compute(predictions=predictions_backward[i], references=[target_backward[i]])
    #bleu_results = bleu.compute(predictions = [predictions[i][0].split(' ')],references=[[target[i].split(' ')]])
    bleu = bleu_score.sentence_bleu(references=[target_backward[i].split(' ')],hypothesis=predictions_backward[i][0].split(' '),weights=(0.5,0.5))
    meteor_score_per_input.append(results)
    bleu_score_per_input.append(bleu)

with open('/Users/vareeshbainwala/Documents/Coursework/Codes/COMS6998/Machine_Translation/Experiment_Results/results_bn_en_backtranslation.csv','w') as f: 
    fieldnames = ['Source','Prediction','Target','Meteor Score','Bleu Score']
    csvwriter = csv.DictWriter(f,fieldnames=fieldnames)
    csvwriter.writeheader()
    # csvwriter.writerow('Source[Prediction[Target[Meteor Score[Bleu Score')
    for i in range(100):  
        csvwriter.writerow({'Source':source_backward[i],'Prediction':predictions_backward[i],'Target': target_backward[i], 'Meteor Score':meteor_score_per_input[i]['meteor'],'Bleu Score': bleu_score_per_input[i]})
