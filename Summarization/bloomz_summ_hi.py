from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import nltk
import evaluate
import numpy as np
import csv
from nltk.translate.meteor_score import meteor_score

df = pd.read_parquet('/Users/vareeshbainwala/Documents/Coursework/Codes/COMS6998/Dataset/Summarization/0000.parquet')
source = df["input"].values.tolist()
target = df['target'].values.tolist()
#print("Translate to Hindi: "+source[0]+".")

# START: COPIED FROM https://huggingface.co/bigscience/bloomz-560m#use
checkpoint = "bigscience/bloomz-560m"
print("Crossed Checkpoint Initiation")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)
# END: COPIED FROM https://huggingface.co/bigscience/bloomz-560m#use

predictions = []
predictions_truncated = []
rouge_score = []
bert = []
#target = ["आज का मौसम बहुत अच्छा है।"]
#target = ['तथापि, पेस, जो ऑस्ट्रेलिया के पॉल हनली के साथ भागीदारी कर रहा था']
# input = "Translate to Hindi: the weather today is great."
#print(source[0])
rouge = evaluate.load('rouge')
bertscore = evaluate.load('bertscore')
for i in range(100):
    input = "Summarize the following text:" + str(source[i])
    #वाराणसी में खाद्य सुरक्षा एवं औषधि प्रशासन की लापरवाही से व्यापारियों में नाराजगी देखी जा रही है।
    # START: COPIED FROM https://huggingface.co/bigscience/bloomz-560m#use
    inputs = tokenizer.encode(input, return_tensors="pt", max_length = 2048,truncation=True)
    outputs = model.generate(inputs,max_new_tokens = 2048)
    # END: COPIED FROM https://huggingface.co/bigscience/bloomz-560m#use
    predictions.append(tokenizer.decode(outputs[0]))
    predictions_truncated.append(predictions[i][len(input):])
    rouge_score.append(rouge.compute(references=[target[i]],predictions=[predictions_truncated[i]],tokenizer=lambda x: x.split()))
    #print(rouge_score[i]['rouge1'])
    bert.append(bertscore.compute(predictions=[predictions_truncated[i]],references=[target[i]],lang = "hi"))

print('Writing to File')

with open('/Users/vareeshbainwala/Documents/Coursework/Codes/COMS6998/Summarization/Experiment_Results/bloomz_summ_hi.csv','w') as file:
    fieldnames = ['Source','Target','Prediction','Rouge1','Rouge2','RougeL','RougeLsum','Precision','Recall','F1']
    csvwriter = csv.DictWriter(file,fieldnames=fieldnames)
    csvwriter.writeheader()
    for i in range(100):
        csvwriter.writerow({'Source':source[i],'Target':target[i],'Prediction':predictions_truncated[i],'Rouge1':rouge_score[i]['rouge1'],'Rouge2':rouge_score[i]['rouge2'],'RougeL':rouge_score[i]['rougeL'],
                            'RougeLsum':rouge_score[i]['rougeLsum'],'Precision':bert[i]['precision'],'Recall':bert[i]['recall'],'F1':bert[i]['f1']})
