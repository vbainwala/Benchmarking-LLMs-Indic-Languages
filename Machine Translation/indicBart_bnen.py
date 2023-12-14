from transformers import MBartForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers import AlbertTokenizer, AutoTokenizer
import pandas as pd
import evaluate
import csv
from nltk.translate import meteor_score, bleu_score

source = []
target = []

df = pd.read_parquet('/Users/vareeshbainwala/Documents/Coursework/Codes/COMS6998/Dataset/MT/0000_bengali.parquet')
source = df["tgt"].values.tolist()
target = df["src"].values.tolist()

# START: COPIED FROM https://huggingface.co/ai4bharat/IndicBART-XXEN
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBART-XXEN", do_lower_case=False, use_fast=False, keep_accents=True)

# Or use tokenizer = AlbertTokenizer.from_pretrained("ai4bharat/IndicBART-XXEN", do_lower_case=False, use_fast=False, keep_accents=True)

model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/IndicBART-XXEN")

# Or use model = MBartForConditionalGeneration.from_pretrained("ai4bharat/IndicBART-XXEN")

# Some initial mapping
bos_id = tokenizer._convert_token_to_id_with_added_voc("<s>")
eos_id = tokenizer._convert_token_to_id_with_added_voc("</s>")
pad_id = tokenizer._convert_token_to_id_with_added_voc("<pad>")
# To get lang_id use any of ['<2as>', '<2bn>', '<2en>', '<2gu>', '<2hi>', '<2kn>', '<2ml>', '<2mr>', '<2or>', '<2pa>', '<2ta>', '<2te>']
# END: COPIED FROM https://huggingface.co/ai4bharat/IndicBART-XXEN

"""Doing translation from the initial dataset with Bengali text as Source and English as the Target"""
meteor_results = []
decoded = []
bleu_results = []
meteor = evaluate.load('meteor')
chencherry = bleu_score.SmoothingFunction()

for i in range(100):
    input = source[i]+" </s> "+"<2bn>"
    # START: COPIED FROM https://huggingface.co/ai4bharat/IndicBART-XXEN
    # First tokenize the input and outputs. The format below is how IndicBART-XXEN was trained so the input should be "Sentence </s> <2xx>" where xx is the language code. Similarly, the output should be "<2yy> Sentence </s>". 
    inp = tokenizer(str(input), add_special_tokens=False, return_tensors="pt", padding=True).input_ids
    output = "<2en>" + target[i] + " </s>"
    out = tokenizer(str(output), add_special_tokens=False, return_tensors="pt", padding=True).input_ids
    #print(out[:,0:-1])
    model_outputs=model(input_ids=inp, decoder_input_ids=out[:,0:-1], labels=out[:,1:])
    model_output=model.generate(inp, use_cache=True, num_beams=4, max_length=1024, min_length=1, early_stopping=True, pad_token_id=pad_id, bos_token_id=bos_id, eos_token_id=eos_id, decoder_start_token_id=tokenizer._convert_token_to_id_with_added_voc("<2en>"))
    # Decode to get output strings
    decoded_output=tokenizer.decode(model_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    #print(decoded_output) # I am a boy
    # END: COPIED FROM https://huggingface.co/ai4bharat/IndicBART-XXEN
    decoded.append(decoded_output)
    results = meteor.compute(predictions=[decoded_output], references=[target[i]])
    bleu = bleu_score.sentence_bleu(references=[target[i].split(' ')],hypothesis=decoded[i].split(' '),weights=(0.5,0.5),smoothing_function=chencherry.method1)
    bleu_results.append(bleu)
    meteor_results.append(results)

print('Writing to First File')

with open('/Users/vareeshbainwala/Documents/Coursework/Codes/COMS6998/Machine_Translation/Experiment_Results/indicbart_bnen.csv','w') as file:
    fieldnames = ['Source','Target','Prediction','Meteor','Bleu']
    csvwriter = csv.DictWriter(file,fieldnames=fieldnames)
    csvwriter.writeheader()
    for i in range(100):
        csvwriter.writerow({'Source':source[i],'Target':target[i],'Prediction':decoded[i],'Meteor':meteor_results[i]['meteor'],'Bleu':bleu_results[i]})


'''Translating the text from Hindi to English using the translation mBart large gave from english to bengali'''


source_bw = []
target_bw = []


with open('/Users/vareeshbainwala/Documents/Coursework/Codes/COMS6998/Machine_Translation/Experiment_Results/results_bn_en_backtranslation.csv','r') as file:
    # fieldnames = ['Source','Prediction','Target','Meteor Score','Bleu Score']
    csvreader = csv.DictReader(file)
    for row in csvreader:
        source_bw.append(row['Source'])
        target_bw.append(row['Target'])

meteor_results_bw = []
decoded_bw = []
bleu_results_bw = []
meteor = evaluate.load('meteor')
chencherry = bleu_score.SmoothingFunction()

for i in range(len(source_bw)):
    input = source_bw[i]+" </s> "+"<2bn>"
    # First tokenize the input and outputs. The format below is how IndicBART-XXEN was trained so the input should be "Sentence </s> <2xx>" where xx is the language code. Similarly, the output should be "<2yy> Sentence </s>". 
    inp = tokenizer(str(input), add_special_tokens=False, return_tensors="pt", padding=True).input_ids
    output = "<2en>" + target_bw[i] + " </s>"
    out = tokenizer(str(output), add_special_tokens=False, return_tensors="pt", padding=True).input_ids
    #print(out[:,0:-1])
    model_outputs=model(input_ids=inp, decoder_input_ids=out[:,0:-1], labels=out[:,1:])
    model_output=model.generate(inp, use_cache=True, num_beams=4, max_length=1024, min_length=1, early_stopping=True, pad_token_id=pad_id, bos_token_id=bos_id, eos_token_id=eos_id, decoder_start_token_id=tokenizer._convert_token_to_id_with_added_voc("<2en>"))
    # Decode to get output strings
    decoded_output=tokenizer.decode(model_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    #print(decoded_output) # I am a boy
    decoded_bw.append(decoded_output)
    results = meteor.compute(predictions=[decoded_output], references=[target_bw[i]])
    bleu = bleu_score.sentence_bleu(references=[target_bw[i].split(' ')],hypothesis=decoded_bw[i].split(' '),weights=(0.5,0.5),smoothing_function=chencherry.method1)
    bleu_results_bw.append(bleu)
    meteor_results_bw.append(results)


print('Writing to Second File')

with open('/Users/vareeshbainwala/Documents/Coursework/Codes/COMS6998/Machine_Translation/Experiment_Results/indicbart_bnen_backtranslation.csv','w') as file:
    fieldnames = ['Source','Target','Prediction','Meteor','Bleu']
    csvwriter = csv.DictWriter(file,fieldnames=fieldnames)
    csvwriter.writeheader()
    for i in range(len(source_bw)):
        csvwriter.writerow({'Source':source_bw[i],'Target':target_bw[i],'Prediction':decoded_bw[i],'Meteor':meteor_results_bw[i]['meteor'],'Bleu':bleu_results_bw[i]})

