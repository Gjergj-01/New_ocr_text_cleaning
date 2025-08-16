# New_ocr_text_cleaning
Fine-tuning mt5 for cleaning italian ocr-text.

SOME Data augmentation suggestions.
1. Randomly remove a pre-defined pecentage of characters
2. Randomly remove a pre-defined percentage of words (This can be too much, maybe we can set a very small percentage)
3. Manually introduce some common erros committed by OCR-systems e.g.:
    i → 1
    o → 0
    and so on.

I think we should consider using also the english dataset. All models we are using are trained also on english data, so this 
shouldn't introduce issues with the italian text, but, I think, it will only make the model more robust and lead to better 
performances.


### Italian OCR datasets

[Datasets](https://drive.google.com/drive/folders/1nbIdLEnXTd2VFUdXcLwYYQ9FpWI3npQ_?usp=drive_link)