# OCR Text Cleaning (of italian text)

**OCR** stands for *Optical Character Recognition* and is a technology designed to extract text from images or scanned documents. However, the output of such systems is usually very noisy and contains errors, and so here we takle the problem of cleaning and correcting it. 

## Methodology
To tackle the problem we adopt an approach based on the use of LLMs; precisely we fine-tune two base models:
- **mt5-base**: a multilingual varian of the "Text-to-Text Transfer Transformer (T5)" which is designed to map text into other text, such as in translation tasks. Therefore, it can be a strong candidate for cleaning Italian OCR-text, since also here we have a "text to text" problem ( mapping the ocr text to its cleaned version).
- **Minerva-3B-base**: an only-decoder model traned on Italian and English text. Here we adopt a generative approach to solve the problem. 
