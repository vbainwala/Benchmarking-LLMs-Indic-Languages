from transformers import MBartForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers import AlbertTokenizer, AutoTokenizer
import pandas as pd
import csv
from rouge_score import rouge_scorer
import evaluate

df = pd.read_parquet('/Users/vareeshbainwala/Documents/Coursework/Codes/COMS6998/Dataset/Summarization/0000.parquet')
source = df["input"].values.tolist()
target = df['target'].values.tolist()

# START: COPIED FROM https://huggingface.co/ai4bharat/MultiIndicSentenceSummarization 
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/MultiIndicSentenceSummarization", do_lower_case=False, use_fast=False, keep_accents=True)
# Or use tokenizer = AlbertTokenizer.from_pretrained("ai4bharat/MultiIndicSentenceSummarization", do_lower_case=False, use_fast=False, keep_accents=True)
model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/MultiIndicSentenceSummarization")
# Or use model = MBartForConditionalGeneration.from_pretrained("ai4bharat/MultiIndicSentenceSummarization")

# Some initial mapping
bos_id = tokenizer._convert_token_to_id_with_added_voc("<s>")
eos_id = tokenizer._convert_token_to_id_with_added_voc("</s>")
pad_id = tokenizer._convert_token_to_id_with_added_voc("<pad>")
# END: COPIED FROM https://huggingface.co/ai4bharat/MultiIndicSentenceSummarization 

rouge = evaluate.load('rouge')
bert_score =  evaluate.load('bertscore')
bert = []
predictions = []
rouge_score = []

for i in range(100):
    # START: COPIED FROM https://huggingface.co/ai4bharat/MultiIndicSentenceSummarization 
    # To get lang_id use any of ['<2as>', '<2bn>', '<2en>', '<2gu>', '<2hi>', '<2kn>', '<2ml>', '<2mr>', '<2or>', '<2pa>', '<2ta>', '<2te>']
    # First tokenize the input. The format below is how IndicBART was trained so the input should be "Sentence </s> <2xx>" where xx is the language code. Similarly, the output should be "<2yy> Sentence </s>".
    inp = tokenizer( str(source[i]) + "</s> <2hi>", add_special_tokens=False, return_tensors="pt", padding=True).input_ids 
    # For generation. Pardon the messiness. Note the decoder_start_token_id.
    model_output=model.generate(inp, use_cache=True,no_repeat_ngram_size=3, num_beams=5, length_penalty=0.8, early_stopping=True, pad_token_id=pad_id, bos_token_id=bos_id, eos_token_id=eos_id, decoder_start_token_id=tokenizer._convert_token_to_id_with_added_voc("<2hi>"))
    # Decode to get output strings
    decoded_output=tokenizer.decode(model_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    # END: COPIED FROM https://huggingface.co/ai4bharat/MultiIndicSentenceSummarization 
    predictions.append(decoded_output)
    rouge_score.append(rouge.compute(predictions=[predictions[i]],references=[target[i]],tokenizer=lambda x: x.split()))
    bert.append(bert_score.compute(predictions=[predictions[i]],references=[target[i]],lang='hi'))

# Note that if your output language is not Hindi or Marathi, you should convert its script from Devanagari to the desired language using the Indic NLP Library.
with open('/Users/vareeshbainwala/Documents/Coursework/Codes/COMS6998/Summarization/Experiment_Results/results_indicbartsummarisation_hi.csv','w') as f: 
    fieldnames = ['Source','Target','Prediction','Rouge1','Rouge2','RougeL','RougeLsum','Precision','Recall','F1']
    csvwriter = csv.DictWriter(f,fieldnames=fieldnames)
    csvwriter.writeheader()
    for i in range(len(predictions)):
        csvwriter.writerow({'Source':source[i],'Target':target[i],'Prediction':predictions[i],'Rouge1':rouge_score[i]['rouge1'],'Rouge2':rouge_score[i]['rouge2'],'RougeL':rouge_score[i]['rougeL'],
                            'RougeLsum':rouge_score[i]['rougeLsum'],'Precision':bert[i]['precision'],'Recall':bert[i]['recall'],'F1':bert[i]['f1']})
