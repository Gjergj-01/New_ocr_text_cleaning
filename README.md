# OCR Text Cleaning (of italian text)

**OCR** stands for *Optical Character Recognition* and is a technology designed to extract text from images or scanned documents. However, the output of such systems is usually very noisy and contains errors, and so here we takle the problem of cleaning and correcting it. 

## Methodology
To tackle the problem we adopt an approach based on the use of LLMs; precisely we fine-tune two base models:
- **mt5-base**: a multilingual variant of the "Text-to-Text Transfer Transformer (T5)" which is designed to map text into other text, such as in translation tasks. Therefore, it can be a strong candidate for cleaning Italian OCR-text, since also here we have a "text to text" problem ( mapping the ocr text to its cleaned version).
- **Minerva-3B-base**: an only-decoder model traned on Italian and English text. Here we adopt a generative approach to solve the problem. 


### Notes on mt5-base
A said before, *mt5* is a storng candidate for this task, since we can interpret the problem of cleaning ocr text as a translation problem, i.e. transalting the ocr-text into clean italian text. 
However, mt5 has some limitations, and the main one is surely the limited window size, which consists of only 20 tokens. So, we make a first preprocessing by slitting the original dataset into smaller samples of the kind `(orc_sample, clean_sample)`, where the criterion it's simply splitting at the end of a sentence. However, since these samples easlily exceed the windows' size, we make an additional splitting, in order to have samples of no more than 20 tokens. This however raises an additional complication, the difficulty of matching the samples, since usually we need more tokens to encode the `ocr_sample` than the ones needed to encode the tokens of the `clean_samples`. \\
In order to do so, we first encoded each sample produced in the first preprocessing step and generated the token ids. Then we used the **sequence-alignment** algorithm (used in biology to align sequences of genes) proposed by Needlman and Wunsch, to align the tokens corresponding to the ocr sample with those corresponding to the clean one. Finally, once we aligned the two sequences, we proceeded in splitting them in chunks of 20 tokens. It might seem counter-intuitive that we aligned tokens instead of the characters of the plain text, but we noticed that this leads to much better results.

#### Example
ocr:  
" \n—  Kagazzo  mio,  —  disse  la  Fata  —  quelli  che \ndicono  così.  Uniscono  quasi  sempre  o  in  carcere \no  all'ospedale."

clean:  
"— Ragazzo mio, — disse la Fata — quelli che dicono così, finiscono quasi sempre o in carcere o all’ospedale."

become:
```
ocr_tokens: ['▁—', '▁Kaga', 'zzo', '▁mio', ',', '▁—', '▁disse', '▁la', '▁Fata', '▁—', '▁quell', 'i', '▁che', '▁di', 'cono', '▁cos', 'ì', '.', '▁Un', '</s>'], 
clean_tokens: ['▁—', '▁Rag', 'azzo', '▁mio', ',', '▁—', '▁disse', '▁la', '▁Fata', '▁—', '▁quell', 'i', '▁che', '▁di', 'cono', '▁cos', 'ì', ',', '▁fin', '</s>']


ocr_tokens: ['iscono', '▁quasi', '▁sempre', '▁', 'o', '▁in', '▁', 'carcer', 'e', '▁', 'o', '▁all', "'", 'os', 'pedale', '.', '</s>', '<pad>', '<pad>', '<pad>'], 
clean_tokens: ['iscono', '▁quasi', '▁sempre', '▁', 'o', '▁in', '▁', 'carcer', 'e', '▁', 'o', '▁all', '’', 'os', 'pedale', '.', '</s>', '<pad>', '<pad>', '<pad>']
```

### Notes on Minerva3B-base
In this case, since the model is too big and we have limited computational resource, we couldn't perform a full fine-tuning. Therefore, we relied on **unsloth**, a library that provides a simple and efficient framework for fine-tuning LLMs. One of the key features of this library is the possibility to add LoRA adapters, which allow to train only a small number of parameters. More precisely, the idea behind **LoRA** (*Low Rank Adaption*) is that of freezing the pre-trained model's wheights and introduce a few new traineable parameters. In practice this means that we train only a small percentage of the model's paramters (1 to 10%) making the fine-tuning much faster and efficient while still keeping good performances.