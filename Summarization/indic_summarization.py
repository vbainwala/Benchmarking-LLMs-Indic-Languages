from transformers import MBartForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers import AlbertTokenizer, AutoTokenizer
import csv
import pandas as pd
from rouge_score import rouge_scorer
import evaluate

df = pd.read_parquet('/Users/vareeshbainwala/Documents/Coursework/Codes/COMS6998/Dataset/Summarization/0000_summ_bn.parquet')
source = df["input"].values.tolist()
target = df['target'].values.tolist()

# START: COPIED FROM https://huggingface.co/ai4bharat/IndicBART-XLSum
tokenizer = AlbertTokenizer.from_pretrained("ai4bharat/IndicBART-XLSum", do_lower_case=False, use_fast=False, keep_accents=True)

# Or use tokenizer = AlbertTokenizer.from_pretrained("ai4bharat/IndicBART-XLSum", do_lower_case=False, use_fast=False, keep_accents=True)

model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/IndicBART-XLSum")

# Or use model = MBartForConditionalGeneration.from_pretrained("ai4bharat/IndicBART-XLSum")

# Some initial mapping
bos_id = tokenizer._convert_token_to_id_with_added_voc("<s>")
eos_id = tokenizer._convert_token_to_id_with_added_voc("</s>")
pad_id = tokenizer._convert_token_to_id_with_added_voc("<pad>")
# To get lang_id use any of ['<2bn>', '<2gu>', '<2hi>', '<2mr>', '<2pa>', '<2ta>', '<2te>']
# END: COPIED FROM https://huggingface.co/ai4bharat/IndicBART-XLSum

rouge = evaluate.load('rouge')
bert_score = evaluate.load('bertscore')
predictions = []
scores = []
bert = []

for i in range(100):
    # First tokenize the input and outputs. The format below is how IndicBART-XLSum was trained so the input should be "Sentence </s> <2xx>" where xx is the language code. Similarly, the output should be "<2yy> Sentence </s>". 
    # inp = tokenizer("टेसा जॉवल का कहना है कि मृतकों और लापता लोगों के परिजनों की मदद के लिए एक केंद्र स्थापित किया जा रहा है. उन्होंने इस हादसे के तीन के बाद भी मृतकों की सूची जारी करने में हो रही देरी के बारे में स्पष्टीकरण देते हुए कहा है शवों की ठीक पहचान होना ज़रूरी है. पुलिस के अनुसार धमाकों में मारे गए लोगों की संख्या अब 49 हो गई है और अब भी 20 से ज़्यादा लोग लापता हैं. पुलिस के अनुसार लंदन पर हमले योजनाबद्ध तरीके से हुए और भूमिगत रेलगाड़ियों में विस्फोट तो 50 सैकेंड के भीतर हुए. पहचान की प्रक्रिया किंग्स क्रॉस स्टेशन के पास सुरंग में धमाके से क्षतिग्रस्त रेल कोचों में अब भी पड़े शवों के बारे में स्थिति साफ नहीं है और पुलिस ने आगाह किया है कि हताहतों की संख्या बढ़ सकती है. पुलिस, न्यायिक अधिकारियों, चिकित्सकों और अन्य विशेषज्ञों का एक आयोग बनाया गया है जिसकी देख-रेख में शवों की पहचान की प्रक्रिया पूरी होगी. महत्वपूर्ण है कि गुरुवार को लंदन में मृतकों के सम्मान में सार्वजनिक समारोह होगा जिसमें उन्हें श्रद्धाँजलि दी जाएगी और दो मिनट का मौन रखा जाएगा. पुलिस का कहना है कि वह इस्लामी चरमपंथी संगठन अबू हफ़्स अल-मासरी ब्रिगेड्स का इन धमाकों के बारे में किए गए दावे को गंभीरता से ले रही है. 'धमाके पचास सेकेंड में हुए' पुलिस के अनुसार लंदन पर हुए हमले योजनाबद्ध तरीके से किए गए थे. पुलिस के अनुसार भूमिगत रेलों में तीन बम अलग-अलग जगहों लगभग अचानक फटे थे. इसलिए पुलिस को संदेह है कि धमाकों में टाइमिंग उपकरणों का उपयोग किया गया होगा. यह भी तथ्य सामने आया है कि धमाकों में आधुनिक किस्म के विस्फोटकों का उपयोग किया गया था. पहले माना जा रहा था कि हमलों में देसी विस्फोटकों का इस्तेमाल किया गया होगा. पुलिस मुख्यालय स्कॉटलैंड यार्ड में सहायक उपायुक्त ब्रायन पैडिक ने बताया कि भूमिगत रेलों में तीन धमाके 50 सेकेंड के अंतराल के भीतर हुए थे. धमाके गुरुवार सुबह आठ बजकर पचास मिनट पर हुए थे. लंदन अंडरग्राउंड से मिली विस्तृत तकनीकी सूचनाओं से यह तथ्य सामने आया है. इससे पहले बम धमाकों में अच्छे खासे अंतराल की बात की जा रही थी.</s> <2hi>", add_special_tokens=False, return_tensors="pt", padding=True).input_ids
    inp = tokenizer(source[i]+"</s> <2bn>", add_special_tokens=False, return_tensors="pt", padding=True).input_ids
    # START: COPIED FROM https://huggingface.co/ai4bharat/IndicBART-XLSum
    # out = tokenizer("<2hi>परिजनों की मदद की ज़िम्मेदारी मंत्री पर </s>", add_special_tokens=False, return_tensors="pt", padding=True).input_ids 
    # model_outputs=model(input_ids=inp, decoder_input_ids=out[:,0:-1], labels=out[:,1:])
    model_output=model.generate(inp, use_cache=True, num_beams=4, max_length=1024, min_length=1, early_stopping=True, pad_token_id=pad_id, bos_token_id=bos_id, eos_token_id=eos_id, decoder_start_token_id=tokenizer._convert_token_to_id_with_added_voc("<2bn>"))
    decoded_output=tokenizer.decode(model_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    # END: COPIED FROM https://huggingface.co/ai4bharat/IndicBART-XLSum
    predictions.append(decoded_output)
    scores.append(rouge.compute(predictions=[decoded_output],references=[target[i]],tokenizer= lambda x : x.split()))
    bert.append(bert_score.compute(predictions=[predictions[i]],references=[target[i]],lang='bn'))
    #print(decoded_output) # लंदन धमाकों में मारे गए लोगों की सूची जारी

with open('/Users/vareeshbainwala/Documents/Coursework/Codes/COMS6998/Summarization/Experiment_Results/results_indicbart_summ_bn.csv','w') as f: 
    fieldnames = ['Source','Prediction','Target','Rouge1','Rouge2','RougeL','RougeLsum','Precision','Recall','F1']
    csvwriter = csv.DictWriter(f,fieldnames=fieldnames)
    csvwriter.writeheader()
    # csvwriter.writerow('Source[Prediction[Target[Meteor Score[Bleu Score')
    for i in range(len(predictions)):  
        csvwriter.writerow({'Source':source[i],'Prediction':predictions[i],'Target': target[i], 'Rouge1':scores[i]['rouge1'],'Rouge2':scores[i]['rouge2'],
                            'RougeL': scores[i]['rougeL'],'RougeLsum':scores[i]['rougeLsum'],'Precision':bert[i]['precision'],'Recall':bert[i]['recall'],'F1':bert[i]['f1']})