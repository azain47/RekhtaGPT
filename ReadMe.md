# RekhtaGPT v0.1

RekhtaGPT is a small Urdu language model trained on 800,000 tokens. Designed to generate Ghazals and sometimes Shers in Roman-Urdu, RekhtaGPT is built on GPT architecture with modern enhancements like rotary positional embeddings.

## Example Generations:
```
'ye sham-e-mai-KHana-e-ulfat na ho'
- Jigar Moradabadi.

jo un ke ke dil-e-KHush-ahang na ho
jo un se ishq mein ho aur kya na ho

mohabbat-o-ruKH-o-KHayal-e-yar ho
jahan-e-ishq mein kahin gham-e-hayat na ho

wo ek nazar mein rahe na us se ziyaada
na to dil mein dard-e-dil-e-be-qarar na ho

wo ham-kalam hai tere ishq mein 'jigar'
wo dil jo dard-o-alam mein gham-KHwar na ho
```
```
'ab-e-mai-KHana ko saba na kar le'
- Ahmad Faraz.

ab-e-ahl-e-dil-e-zar na kar le
ab wo dard-e-arzu ki aag na kar le

ab wo dil ki umr-e-jawedan na kar saka
ab bhi shahr mein teri mulaqat na kar le

ab to ye dil ki bazi hai ki wafa ka jawab
har taraf ke charche shajar-e-jaan na kar le

har qadam se hai ek bar wahi KHudai
ab bhi aaj kahin se tere gham-KHwar na kar le

ab tera haal-e-wafa bhi yaad nahin ki magar
ab tera KHayal-e-yar ka nam na kar le
```
```
aae-me-mohabbat mein agarche kar chale
baiThe baiThe hum teri rah mein mar kar chale

sunte hain dekh kar-gah-e-dil-e-na-rawa
ai falak us ke ru-e-nigar-e-zulf-e-yar kar chale

dekhen hum ko ta-hashr ye chale the ki idhar se
kise ai dil-e-tabassum kar tu sath chale
```
I would say the generations are not even that bad, considering the amount of data this was trained on.

## Model Configuration

The model has been configured with the following parameters:

- **Context Length:** 512
- **Vocabulary Size:** 8192
- **Number of Layers:** 12
- **Number of Attention Heads:** 8
- **Embedding Dimension:** 768
- **Total Parameters:** 97.6 Million

## Training Data

RekhtaGPT was trained on a corpus of 3800+ ghazals from various poets totalling to around 862000 tokens.

The training details are as follows:
- **Gradient Accumulation Steps:** 16
- **Learning Rate:** Total Steps 500. Warmup to *7e-4* for 20 steps, then cosine decay to 7e-5 for next 480 steps.
- **Gradient Clipping:** The gradient norm has been clipped to 1.0 . 

## Custom Tokenizer

To effectively tokenize Urdu text for RekhtaGPT, I built a custom tokenizer with a vocabulary size of 8192, built using the [SentencePiece](https://github.com/google/sentencepiece) library and the unigram model. 

## The Need For Custom Tokenizer

Since the datasets used today are mostly in English, the tokenizers are not able to learn full words as single tokens when it comes to other languages. This leads to loss of information. Hence I created a tokenizer solely for urdu, and this improved the performance when compared to tiktoken gpt2 tokenizer. 
>**NOTE** Since the data used for training the model is very less, it is important for the words to be properly tokenized when you put an initial prompt. The lack of large dataset causes generalization issues. Therefore a playground notebook is provided to check if prompt is properly tokenized, before passing to the model.

## Features

- **Language:** Urdu
- **Architecture:** GPT
- **Positional Encoding:** Rotary positional embeddings for improved contextual understanding
- **Normalization:** RMSNorm applied to stabilize training

## Inferencing
Download the model from 'Releases' and move it to the same folder.
Run the ```generate.py``` file and tweak the inference settings, such as temperature, top-p threshold.
Currently Top-P and Top-K are supported, with Beam Search WIP.


## TO-DO
The next step is to use a more general and much larger dataset so this model can truly be used for general urdu language understanding.

## Contributing
I truly welcome any contribution from the community to improve this model. It could be dataset contributions, model changes, or inferencing strategies.  
