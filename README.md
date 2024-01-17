# Benchmarking LLMs on Indic Languages
Performed as part of COMS6998 Course@Columbia University 

# Dataset
Samanantar : https://huggingface.co/datasets/ai4bharat/samanantar <br/>
IndicSentenceSummarization : https://huggingface.co/datasets/ai4bharat/IndicSentenceSummarization

# Models Used
Bloomz-560m : https://huggingface.co/bigscience/bloomz-560m <br/>
mBART-large : https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt <br/>
IndicBART-XXEN : https://huggingface.co/ai4bharat/IndicBART-XXEN

# Implementation
For each model the existing code base available at https://huggingface.co/ is recfactored to perform the following tasks: <br/>
- Machine Translation <br/>
- Summarization

Each file obtained as the output is further processed to find the mean values for the following metrics <br/>
- METEOR
- BLEU
- ROUGE
- BERTScore

