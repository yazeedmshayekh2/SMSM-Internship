# PDFPlumber 

**What is PDFPlumber?**

Plumb a PDF for detailed information about each text character, rectangle, and line. Plus: Table extraction and visual debugging.

Works best on machine-generated, rather than scanned, PDFs. Built on [`pdfminer.six`](https://github.com/pdfminer/pdfminer.six). 


[![Version](https://img.shields.io/pypi/v/pdfplumber.svg)](https://pypi.python.org/pypi/pdfplumber) [![Github](https://img.shields.io/badge/Github_Source-gray)](https://github.com/jsvine/pdfplumber/)

**Task Description:**

Your task is to evaluate the PDFPlumber PDF text extraction tool by running it on a provided sample PDF file. Your evaluation should include a comprehensive analysis of the tool, highlighting its strengths and weaknesses.

**What to Submit:**

1. **Text Files:** Submit text files containing the extracted text from the PDF. Each text file's name should match the corresponding PDF file's name. These text files will contribute to a matrix calculation to assess the tool's performance. Optimize your code to maximize your score in the evaluation matrix.

2. **Documented Analysis:** Provide a detailed analysis documenting your observations and conclusions based on your exploration and evaluation of the PDFPlumber tool.

3. **Code:** Clone the provided GitHub repository containing the code template for text extraction. Add your code to the repository, commit the changes, and push them to the remote repository.



**[Sample Text](https://github.com/SMSM-AI/AMUN-RAG/tree/main/Sample%20Text)**

Ensure that your submission is well-organized and adheres to the provided guidelines.

- [You can find the code here in colab](https://colab.research.google.com/drive/14XjcllAE3bnoSKMdH0-ny1V0Suerl_8L?usp=sharing)

______________________________________

# Full Report about using PDFPlumber with Arabic text

# PdfPlumber - Report

# Code

> Code Found Here:
> 

[Google Colaboratory](https://colab.research.google.com/drive/14XjcllAE3bnoSKMdH0-ny1V0Suerl_8L?usp=sharing)

---

# Goal

> We want to extract the text from a PDF and compare that against a clean file of the same text for 5 pdf files, in order to benchmark - pdfplumber - with Arabic language in python.
> 

---

# Files Used

## Pdf Files

[Test (1).pdf](PdfPlumber%20-%20Report%20787ac8af1452436caf8128094fa4aae8/Test_(1).pdf)

[Test (4).pdf](PdfPlumber%20-%20Report%20787ac8af1452436caf8128094fa4aae8/Test_(4).pdf)

[Test (2).pdf](PdfPlumber%20-%20Report%20787ac8af1452436caf8128094fa4aae8/Test_(2).pdf)

[Test (5).pdf](PdfPlumber%20-%20Report%20787ac8af1452436caf8128094fa4aae8/Test_(5).pdf)

[Test (3).pdf](PdfPlumber%20-%20Report%20787ac8af1452436caf8128094fa4aae8/Test_(3).pdf)

---

## Text Files Before Processing

> Extract the text from pdf files using pdfplumber, and here is the files without post processing:
> 

[Test (1)_before.txt](PdfPlumber%20-%20Report%20787ac8af1452436caf8128094fa4aae8/Test_(1)_before.txt)

[Test (4)_before.txt](PdfPlumber%20-%20Report%20787ac8af1452436caf8128094fa4aae8/Test_(4)_before.txt)

[Test (2)_before.txt](PdfPlumber%20-%20Report%20787ac8af1452436caf8128094fa4aae8/Test_(2)_before.txt)

[Test (5)_before.txt](PdfPlumber%20-%20Report%20787ac8af1452436caf8128094fa4aae8/Test_(5)_before.txt)

[Test (3)_before.txt](PdfPlumber%20-%20Report%20787ac8af1452436caf8128094fa4aae8/Test_(3)_before.txt)

---

## Text Files After Processing

> Extract the text from pdf files using pdfplumber, and here is the files with applying some post processing techniques:
> 

[Test (1).txt](PdfPlumber%20-%20Report%20787ac8af1452436caf8128094fa4aae8/Test_(1).txt)

[Test (4).txt](PdfPlumber%20-%20Report%20787ac8af1452436caf8128094fa4aae8/Test_(4).txt)

[Test (2).txt](PdfPlumber%20-%20Report%20787ac8af1452436caf8128094fa4aae8/Test_(2).txt)

[Test (5).txt](PdfPlumber%20-%20Report%20787ac8af1452436caf8128094fa4aae8/Test_(5).txt)

[Test (3).txt](PdfPlumber%20-%20Report%20787ac8af1452436caf8128094fa4aae8/Test_(3).txt)

---

## Original Text Files to compare with

> Now I want to compare the original text with extracted one, once after processing, and once before, here is the files:
> 

[Test (1) _ baseline.txt](PdfPlumber%20-%20Report%20787ac8af1452436caf8128094fa4aae8/Test_(1)___baseline.txt)

[Test (4) _ baseline.txt](PdfPlumber%20-%20Report%20787ac8af1452436caf8128094fa4aae8/Test_(4)___baseline.txt)

[Test (2) _ baseline.txt](PdfPlumber%20-%20Report%20787ac8af1452436caf8128094fa4aae8/Test_(2)___baseline.txt)

[Test (5) _ baseline.txt](PdfPlumber%20-%20Report%20787ac8af1452436caf8128094fa4aae8/Test_(5)___baseline.txt)

[Test (3) _ baseline.txt](PdfPlumber%20-%20Report%20787ac8af1452436caf8128094fa4aae8/Test_(3)___baseline.txt)

---

## Requirements

```python
!pip install pdfplumber
!pip install python-bidi
!pip install Levenshtein
!pip install arabic-reshaper
!pip install pyarabic
```

## Dependencies

```python
import pdfplumber
import pandas as pd
from bidi import algorithm
import bidi.algorithm as bidi
import pyarabic.araby as araby
from arabic_reshaper import reshape
import datetime

import Levenshtein

import string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
```

---

# pdfplumber - Weaknesses - pdf files - Arabic Language

[Test (1).pdf](PdfPlumber%20-%20Report%20787ac8af1452436caf8128094fa4aae8/Test_(1).pdf)

> Read the pdf using pdfplumber:
> 

```python
reader = pdfplumber.open("Test (1).pdf")
```

> Save the extracted text from pages in dictionary:
> 

```python
pages = {}
for i in range(len(reader.pages)):
    page = reader.pages[i]
    pages[f'page_{i}'] = page.extract_text()

    output_text_first += pages[f'page_{i}']
```

> Print the extracted text in each page:
> 

```python
for page in range(len(reader.pages)):
    print(f'page {page} content is : \n \n', pages[f'page_{page}'], '\n')
```

> As shown below, I took the output of first two pages only.
> 

---

<aside>
💪 Main Advantages - Strengths

</aside>

- It can handle the structure of the extracted text well:

```python
Output text after processing: 
 
 اﻟﺜﻌﻠﺐ اﻟﺬي ﺧﺪع ﻧﻔﺴﮫ ....... ( ﻣﺴﺮﺣﯿﺔ ﻗﺼﯿﺮة ﻟﻸطﻔﺎل )
ﺗﺄﻟﯿﻒ :أﺣﻤﺪ اﺳﻤﺎﻋﯿﻞ اﺳﻤﺎﻋﯿﻞ
-----------------------------------------------------------------
(اﻟﻤﻨﻈﺮ ﻋﺒﺎرة ﻋﻦ ﺗﻞ ﯾﺸﺮف ﻋﻠﻰ ﺳﮭﻞ أﺧﻀﺮ رﺣﺐ , ﻓﻮق اﻟﺘﻞ ﺻﺨﺮة
ﻛﺒﯿﺮةﯾﺨﺘﺒﺊ ﺧﻠﻔﮭﺎ اﻟﺜﻌﻠﺐ وھﻮ ﯾﺴﺘﺮق اﻟﻨﻈﺮات إﻟﻰ اﻟﺴﮭﻞ ,ﻟﺤﻈﺎت ﯾﺪﺧﻞ ذﺋﺐ ھﺰﯾﻞ
اﻟﺠﺴﻢ)
اﻟﺬﺋﺐ : ﻋﻮوو..
اﻟﺜﻌﻠﺐ: (ﯾﺠﻔﻞ) أھﺬا أﻧﺖ أﯾﮭﺎ اﻟﺬﺋﺐ اﻟﻌﺠﻮز ؟
اﻟﺬﺋﺐ : ھﻞ أﺧﻔﺘﻚ ؟
اﻟﺜﻌﻠﺐ: (ﻣﻜﺎﺑﺮا ) ﻻ.ﻻ. ﻟﻢ أﺧﻒ .
اﻟﺬﺋﺐ : ﻟﻜﻨﻚ ﺟﻔﻠﺖ , و ﺷﺤﺐ ﻟﻮﻧﻚ .
اﻟﺜﻌﻠﺐ: ظﻨﻨﺘﻚ ذﺋﺒﺎً أﺧﺮ .
اﻟﺬﺋﺐ : ذﺋﺒﺎً ﺷﺎﺑﺎً وﻗﻮﯾﺎً ؟ أﻟﯿﺲ ﻛﺬاﻟﻚ؟
اﻟﺜﻌﻠﺐ: (ﺑﻀﯿﻖ ) أف أﯾﮭﺎ اﻟﻌﺠﻮز , دﻋﻨﻲ أﻧﻔﺬ اﻟﻤﮭﻤﺔ اﻟﺘﻲ أﻣﺮﻧﻲ ﺑﮭﺎ ﻣﻠﻚ
اﻟﻐﺎﺑﺔ .(ﯾﻌﻮد اﻟﺜﻌﻠﺐ إﻟﻰ ﻣﺮاﻗﺒﺔ اﻟﺴﮭﻞ )
اﻟﺬﺋﺐ : ( ﺑﺎﺳﺘﻐﺮاب ) ﻣﺎذا ﻗﻠﺖ ،ﻣﮭﻤﺔ ؟!ﻻ ﺷﻚ أﻧﮭﺎ ﻣﮭﻤﺔ ﺧﻄﺮة .
اﻟﺜﻌﻠﺐ : (ﺑﻐﺮور ) ﻧﻌﻢ . ھﻲ ﻛﺬاﻟﻚ .
اﻟﺬﺋﺐ : وﻣﺎ ھﻲ ؟
اﻟﺜﻌﻠﺐ : ھﺬا ﺳﺮ.

________________________________________________________________________________________

Original:
```

![Untitled](PdfPlumber%20-%20Report%20787ac8af1452436caf8128094fa4aae8/Untitled.png)

---

- Extract the text accurately:

```python
# As shown in the example above
```

---

<aside>
🚨 Main Problems - Weaknesses

</aside>

- It flips the text, so it appears in a wrong form, here is an example for two pages of the first file - (Note: This problem was occurred when dealing with all pdf files):

```r
page 0 content is : 
 
 ( دﻠﺑﻟا لھأ ءﺎﺿوﺿ ﻰﻠﻋرادﻟا ةدﻣﻌﻟا لﺧدﯾ )
دﻠﺑﻟا ﻰﺷﻣا فرﺎﻋ شﻣ .. ىد دﻠﺑﻟا ﻰﻓ ﺞﻧرﺧ ةدﻣﻋ ... وھو تﻧا ﻰﻧﺑا ﺎﯾ سﺑ .. ﷲ ﻻا ﮫﻟا اااﻻ : ةدﻣﻌﻟا
صﻟﺎﺧ ىد
راااﺎﻧ .. ةدﻣﻋ ﺎﯾ ﺎﯾﻓ ىرﺟﺗﺑ رااﺎﻧ ... ﻰﻧﻘﺣﻟا ... ةدﻣﻌﻟا ﺎﺑا ﺎﯾ ﻰﻧﻘﺣﻟا : ىدﯾﻌﺟ
ﻻو ﺎﯾ لﺻﺣ ﻰﻠﻟا سﺑ ﮫﯾا ... ىدﯾﻌﺟ ﺎﯾ داو ﺎﯾ رﯾﺧ : ةدﻣﻌﻟا
ﻰﺗاﺎﺗرﻣ ... ةدﻣﻌﻟا ﺎﺑا ﺎﯾ نﯾﻧﺗﻹا ﻰﺗﺎﺗارﻣ ... ةدﻣﻌﻟا ﺎﺑا ﺎﯾ قﻠطازﯾﺎﻋ ﺎﻧا : ىدﯾﻌﺟ
ىدﯾﻌﺟ ﺎﯾ داو ﺎﯾ مﮭﻟﺎﻣ : ةدﻣﻌﻟا
ﺎﯾﻓ برﺿ اوﻠﻣﻛﯾ نﯾزﯾﺎﻋ و ﺎﯾارو ىرﺟ نﯾﯾﺎﺟ : ىدﯾﻌﺟ
لﺻﺣ ﻰﻠﻟا سﺑ ﮫﯾا : ةدﻣﻌﻟا
تﺑرﺷو .. ﻰﻋﺎﺗﺑ ﻰﺗ سﯾﺎﻧﻟا تدﺧا ﺎﻣا دﻌﺑ موﻧﻟا نﻣ تﻣﻗ ﺎﻧا... ىدﯾﺳ ﺎﯾ صﺑ ... كﻠﯾﻛﺣھ ﺎﻧا : ىدﯾﻌﺟ
ﺔﻛﺳﺎﻣ ةدﺣاو ... ﺔﻧﯾﻛﺳو ﺎﯾر ىز نﯾﻧﺗﻹا ﺎﻣھ ﻰﻠﻧﯾﺎﺟ مﮭﺗﯾﻘﻟ ... موﯾ لﻛ عﺎﺗﺑ نﺑﻠﻟا صﻼﺑ
بوووھ و ... شﺎﺷر ﻊﻓدﻣ ﺔﻠﯾﺎﺷ ﺎﮭﻧﻛا ﻻو رﯾرﺳﻟا ﺔﺗوﻛﺳ ﺔﻠﯾﺎﺷ ﺔﯾﻧﺎﺗﻟا و ﺦﺑطﻣﻟا ﺔﺷﻘﻣ
ﻰﺧوﻔﻧ قوﻓ غدﻏدﺗﻣ شﻣﻟا صﻼﺑ تﯾﻘﻟ
مﮭﺗﺎﻧﺑ و مھ مﮭﯾﻔﻛﻣ شﻣ تﻧا وھ .. ﻻو ﺎﯾ سﺑ ﮫﯾﻟ ... ﷲ ﷲ ﷲ : ةدﻣﻌﻟا
ﷲ دﻣﺣﻟا .. ةدﻣﻌﻟا ﺎﺑﺎﯾ ﷲ دﻣﺣﻟا : ىدﯾﻌﺟ
هدﻛ شﻣ كﯾﻠﻋ ةدوﺳ ﺔﻠﻣﺎﻋ لﻣﺎﻋ دﯾﻛا تﻧا ﻰﻘﺑﯾ : ةدﻣﻌﻟا
ﻰﻋوا سﺑ كﻟوﻘھ ﺎﻧا.. ةدﻣﻌﻟا ﺎﺑا ﺎﯾ تﺎﺳﻧآ ﮫﺳﻟ و نﯾﻧﺗا زوﺟﺗﻣ ﺎﻧاد ... ةدﻣﻌﻟا ﺎﺑﺎﯾ ﮫﯾا ﺔﻠﻣﺎﻋ : ىدﯾﻌﺟ
مدﻗ ﻊﺿﯾ ) ﮫﻠﻛ ﮫﻌﯾﺑا ... رﺟﻧﺑﻟا لوﺻﺣﻣ مﻟا ﺎﻣا دﻌﺑ تررﻗ ﺎﻧا ... فوﺷ ... هدﻛ سﺑ تﻧا
( مدﻗ ﻰﻠﻋ
ىدﯾﻌﺟ ﺎﯾ ﻻو : ةدﻣﻌﻟا
مﻋ ﺎﯾ مﻼﻛ ىا ةدﻣﻋ تﻧاد ﮫﯾا ﺔﻠﯾﻧ و ﮫﯾا ﻻو : ىدﯾﻌﺟ
ىدﯾﻌﺟ ﺎﯾ ﻻو ﺎﯾ كﻠﺟر لزﻧ : ةدﻣﻌﻟا
ﺎﺑا ﺎﯾ سوﻠﻓ ﻰﻠﺟﯾھ ... ﮫﻌﯾﺑا و رﺟﻧﺑﻟا لوﺻﺣﻣ ﻊﯾﺑا ﺎﻣا دﻌﺑ تررﻗ ﺎﻧا .. ةدﻣﻌﻟا ﺎﺑﺎﯾ ةذﺧاؤﻣﻻ : ىدﯾﻌﺟ
زوﺟﺗا ... كﺗارﻣ تﻠﻣا و كﺗﻠﻣا لﺎﺑﻘﻋ ﻰﻘﺑ تررﻗ ةدﻣﻌﻟا
مﮭﻟﺎﺣﺑ نﯾﻧﺗا زوﺟﺗﻣ شﻣ تﻧا ... ااااﻻو ﺎﯾ ﮫﯾا زوﺟﺗﺗ ... زوﺟﺗﺗ : ةدﻣﻌﻟا 

page 1 content is : 
 
 لوﺻﺣﻣ عﺎﺗﺑ ﮫﯾﻧﺟ 200لا ﮫﻠﯾدا و ﻰﻟوﺗﻣ جﺎﺣﻟا مﻋ حورا تﻟوﻗ ﺎﻧا ... ةدﻣﻋ ﺎﯾ سﺑ ﻊﻣﺳا : ىدﯾﻌﺟ
دﻧﻋ لﻠﺣﻟا لﺳﻐﺗﺑ ﻰھو ﺎﮭﺗﻔﺷ ةدﻣﻌﻟا ﺎﺑﺎﯾ ىد ﺔﯾردﺑ ﮫﺗﻧﺑ ... ﺔﯾردﺑ ﮫﺗﻧﺑ زوﺟﺗا و ... رﺟﻧﺑﻟا
( ةدﻣﻌﻟا شوﺷوﯾ ) نﺎﻣﻛ و .. ﺔﻋرﺗﻟا
شﻣ تﻧا .. ﻻو ﺎﯾ ﮫﯾﻟ ىروﻌﺷ نﻋ ﻰﻧﺟرﺧﺗھ تﻧا .. ااﻻو ... ىد ﺔﯾردﺑ تﺑﻟا ... قﺣﻠﻟ تﯾﺟ نا : ةدﻣﻌﻟا
ﻰﻧﺎﺗ ﮫﯾا زوﺎﻋ نﯾﻧﺗا زوﺟﺗﻣ
ﻰﻠﺧ سﺑ ... عوﺿوﻣﻟا كﻟوﻘھ ﺎﻧا سﺑ صﺑ ... ةدﻣﻌﻟا ﺎﺑﺎﯾ ﺎﯾ عرﻗا تﻧاد ةدﻣﻌﻟا ﺎﺑﺎﯾ ﮫﯾا روﻌﺷ : ىدﯾﻌﺟ
وھا كﻟوﻘﺑ ﺎﻧا كﺎﻔﻗ ﻰﻠﻋ كﻠﻛﺎھ مﺟ وﻟ فرﺎﻋ تﻧا ... ةﺄﺟﻓ اوﺑطﯾ نﺳﺣا كﻟﺎﺑ
ﺔﺟﯾدﺧ لﯾﻛوﻟا مﻌﻧ و ﷲ ﻰﺑﺳﺣ ... ﺢﯾﺣﺻ تﻧﻧﺟﺗا تﻧا .. ﺔﻔﻘﻟﺎﺑ نﯾﻣﻟ ىدﺗ .. ااﻻاﺎﯾ لﺑھا تﻧا :ةدﻣﻌﻟا
طﺑظﻟﺎﺑ ﻰﯾز لﺑھا دﺣاو تﻔﻠﺧ
.. ﮫﯾﯾا نﻣ ﺎھوﻓرﻌﯾﺑ ﺔﻠﺑﮭﻟا ﻰھ ﺎﻣ : ىدﯾﻌﺟ
ﻻو ﺎﯾ ﮫﯾا نﻣ : ةدﻣﻌﻟا
ﺔﻧﯾز نوﻧﺑﻟا و لﺎﻣﻟا ... نآرﻘﻟا ﻰﻓ لوﻘﯾﺑ ﺎﻧﺑر شﻣ ... عوﺿوﻣﻟا ﻰﻓ ﻰﻘﺑ ﺎﻧﯾﻠﺧ ... ﺎﮭﺗﻔﻠﺧ نﻣ : ىدﯾﻌﺟ
نوﻧﺑﻟا لﺿﺎﻓ ﻰﻘﺑﯾ ... رﺟﻧﺑﻟا لوﺻﺣﻣ عوﺗﺑ ﮫﯾﻧﺟ 200لا ﺎﯾﺎﻌﻣ ﻰﻘﺑ ﺎﻧا ... ﺎﯾﻧدﻟا ةﺎﯾﺣﻟا
نﺎﯾﺑﺻﻟا و تﺎﻧﺑﻟا نوﻧﺑﻟﺎﺑ دﺻﻘﯾ ﺎﻧﺑر ... ﻻو ﺎﯾ كﺟازﻣ ﻰﻠﻋﺔﯾﻵا رﺳﻔﺗھ تﻧا ... ﷲ ﻻا ﮫﻟا ﻻ : ةدﻣﻌﻟا
ﻻو ﺎﯾ سﺑ نﺎﯾﺑﺻﻟا شﻣ
ﺎﺑﺎﯾ ﻰﻧﻘﺣﻟا ... لﯾﻧﻣ و دوﺳا راﺎﮭﻧ ﺎﯾ ... ( ﮫﺗﺎﺗرﻣ تاوﺻا ﻊﻣﺳﯾ ) ... ةدﻣﻌﻟا ﺎﺑﺎﯾ كﻣﮭﻔھ ﺎﻧا : ىدﯾﻌﺟ
ﻰﻧﯾﺑﺧ .... ةدﻣﻌﻟا
ﮫﻠﺗﻘﻧ و كﻠﺗﻘﻧ نﺳﺣا ةدﻣﻌﻟا ﺎﺑﺎﯾ قطﻧا ... ﮫﻧﯾﻋ ﻰﻠﻋ لﯾﻧﻣﻟا نﯾﻓ وھ .. نﯾﯾﯾﻓ وھ : ﺔﯾرﺑﺻ – ﺔﯾدﻌﺳ
ﺎﮭﯾﻟ كﻧﻣ تﺑ ﺎﯾ سﺑ : ةدﻣﻌﻟا
كﻓوﺷا ﺎﻣﻟ بط .. ﮫﻠﻛ هد رﻣﻌﻟا دﻌﺑ ﺎﻧﯾﻠﻋ زوﺟﺗﯾ زوﺎﻋ ﮫﻧﯾﻋ ﻰﻠﻋ لﯾﻧﻣﻟا .. ةدﻣﻌﻟا ﺎﺑﺎﯾ كﯾﺿرﯾ : ﺔﯾدﻌﺳ
كﺳﻔﻧ شﯾﻠﻋزﺗﻣ .. سﺑ ﻰﺗﺧا ﺎﯾ سﺑ : ﺔﯾرﺑﺻ
بﯾرﻗ نﻋ ﷲ ءﺎﺷ نا ﺎﮭﻠﺣھ و مﻛﺗﻠﻛﺷﻣ تﻓرﻋ ﻰﻧا .. صﻼﺧ سﺑ : ةدﻣﻌﻟا
ةدﻣﻋ ﺎﯾ ﮫﻧﻣ ﻰﻘﺣ ﻰﻠﺑﺟﺗھ .. ﻰﻧﻌﯾ ( ﻊﻟدﺑ ) : ﺔﯾرﺑﺻ
رﻣﻗ ﺎﯾ كﻧوﯾﻋ رطﺎﺧ نﺎﺷﻋ ﮫﺑﯾﺟھ ةوﯾا : ةدﻣﻌﻟا
قﺣ ﺎﯾﻟ نﺎﻣﻛ ﺎﻧا .. نﺎﻣﻛ ﺎﻧا و ( ظﯾﻐﺑ ) : ﺔﯾدﻌﺳ
ﮫﺳﻟ ﻻو مﺎﻣﺣﻟا ﻰﻓ تﺻﻠﺧ ... ىدﯾﻌﺟ ﺎﯾ داو ﺎﯾ تﻧا ... ﺎﻌﺑط ﮫﺑﯾﺟا مزﻻ .. ﺎﻌﺑط ﺎﻌﺑط ها : ةدﻣﻌﻟا
```

> but my colleague hazem_33 solve this problem: [https://discord.com/channels/1219261069005160468/1222950170916950129/1222973505411485726](https://discord.com/channels/1219261069005160468/1222950170916950129/1222973505411485726)
> 

```python
# process the extracted text

path = "Test (1).pdf" # path to file

# fucntion to save the extracted text in a text file
def save_extracted_text(path, output_text):
    f = open('After/' + path, 'w')
    # write the output_text in a text file
    f.write(output_text)
    f.close()

# to solve the problem of fliping the text.
# this solution took from my colleague hazem_33 https://discord.com/channels/1219261069005160468/1222950170916950129/1222973505411485726
def process(path):

    with pdfplumber.open(path) as pdf:

        now = datetime.datetime.now()

        # iterate over each page
        text=[]
        for ind, page in enumerate(pdf.pages):
            # extract and reverse text
            text.append(algorithm.get_display(
                page.extract_text(keep_blank_chars=True))
            )

        stop = datetime.datetime.now()

        time_processed = stop - now

    return "\n".join(text), time_processed

# print the output after applying read_pdf function on all files
output_first, time_first_processed = process(path)
print('Time for processing: ', time_first_processed, '\n', '\nOutput text after processing: \n \n', output_first)
save_extracted_text(path.replace('.pdf', '.txt') , output_first)
```

```r
( ﯾدﺧل اﻟﻌﻣدة اﻟدارﻋﻠﻰ ﺿوﺿﺎء أھل اﻟﺑﻠد )
اﻟﻌﻣدة : ﻻااا اﻟﮫ اﻻ ﷲ .. ﺑس ﯾﺎ اﺑﻧﻰ اﻧت وھو ... ﻋﻣدة ﺧرﻧﺞ ﻓﻰ اﻟﺑﻠد دى .. ﻣش ﻋﺎرف اﻣﺷﻰ اﻟﺑﻠد
دى ﺧﺎﻟص
ﺟﻌﯾدى : اﻟﺣﻘﻧﻰ ﯾﺎ اﺑﺎ اﻟﻌﻣدة ... اﻟﺣﻘﻧﻰ ... ﻧﺎاار ﺑﺗﺟرى ﻓﯾﺎ ﯾﺎ ﻋﻣدة .. ﻧﺎااار
اﻟﻌﻣدة : ﺧﯾر ﯾﺎ واد ﯾﺎ ﺟﻌﯾدى ... اﯾﮫ ﺑس اﻟﻠﻰ ﺣﺻل ﯾﺎ وﻻ
ﺟﻌﯾدى : اﻧﺎ ﻋﺎﯾزاطﻠق ﯾﺎ اﺑﺎ اﻟﻌﻣدة ... ﻣراﺗﺎﺗﻰ اﻹﺗﻧﯾن ﯾﺎ اﺑﺎ اﻟﻌﻣدة ... ﻣرﺗﺎاﺗﻰ
اﻟﻌﻣدة : ﻣﺎﻟﮭم ﯾﺎ واد ﯾﺎ ﺟﻌﯾدى
ﺟﻌﯾدى : ﺟﺎﯾﯾن ﺟرى وراﯾﺎ و ﻋﺎﯾزﯾن ﯾﻛﻣﻠوا ﺿرب ﻓﯾﺎ
اﻟﻌﻣدة : اﯾﮫ ﺑس اﻟﻠﻰ ﺣﺻل
ﺟﻌﯾدى : اﻧﺎ ھﺣﻛﯾﻠك ... ﺑص ﯾﺎ ﺳﯾدى ...اﻧﺎ ﻗﻣت ﻣن اﻟﻧوم ﺑﻌد اﻣﺎ اﺧدت اﻟﻧﺎﯾس ﺗﻰ ﺑﺗﺎﻋﻰ .. وﺷرﺑت
ﺑﻼص اﻟﻠﺑن ﺑﺗﺎع ﻛل ﯾوم ... ﻟﻘﯾﺗﮭم ﺟﺎﯾﻧﻠﻰ ھﻣﺎ اﻹﺗﻧﯾن زى رﯾﺎ وﺳﻛﯾﻧﺔ ... واﺣدة ﻣﺎﺳﻛﺔ
ﻣﻘﺷﺔ اﻟﻣطﺑﺦ و اﻟﺗﺎﻧﯾﺔ ﺷﺎﯾﻠﺔ ﺳﻛوﺗﺔ اﻟﺳرﯾر وﻻ اﻛﻧﮭﺎ ﺷﺎﯾﻠﺔ ﻣدﻓﻊ رﺷﺎش ... و ھوووب
ﻟﻘﯾت ﺑﻼص اﻟﻣش ﻣﺗدﻏدغ ﻓوق ﻧﻔوﺧﻰ
اﻟﻌﻣدة : ﷲ ﷲ ﷲ ... ﻟﯾﮫ ﺑس ﯾﺎ وﻻ .. ھو اﻧت ﻣش ﻣﻛﻔﯾﮭم ھم و ﺑﻧﺎﺗﮭم
ﺟﻌﯾدى : اﻟﺣﻣد ﷲ ﯾﺎﺑﺎ اﻟﻌﻣدة .. اﻟﺣﻣد ﷲ
اﻟﻌﻣدة : ﯾﺑﻘﻰ اﻧت اﻛﯾد ﻋﺎﻣل ﻋﺎﻣﻠﺔ ﺳودة ﻋﻠﯾك ﻣش ﻛده
ﺟﻌﯾدى : ﻋﺎﻣﻠﺔ اﯾﮫ ﯾﺎﺑﺎ اﻟﻌﻣدة ... داﻧﺎ ﻣﺗﺟوز اﺗﻧﯾن و ﻟﺳﮫ آﻧﺳﺎت ﯾﺎ اﺑﺎ اﻟﻌﻣدة ..اﻧﺎ ھﻘوﻟك ﺑس اوﻋﻰ
اﻧت ﺑس ﻛده ... ﺷوف ... اﻧﺎ ﻗررت ﺑﻌد اﻣﺎ اﻟم ﻣﺣﺻول اﻟﺑﻧﺟر ... اﺑﯾﻌﮫ ﻛﻠﮫ ( ﯾﺿﻊ ﻗدم
ﻋﻠﻰ ﻗدم )
اﻟﻌﻣدة : وﻻ ﯾﺎ ﺟﻌﯾدى
ﺟﻌﯾدى : وﻻ اﯾﮫ و ﻧﯾﻠﺔ اﯾﮫ داﻧت ﻋﻣدة اى ﻛﻼم ﯾﺎ ﻋم
اﻟﻌﻣدة : ﻧزل رﺟﻠك ﯾﺎ وﻻ ﯾﺎ ﺟﻌﯾدى
ﺟﻌﯾدى : ﻻﻣؤاﺧذة ﯾﺎﺑﺎ اﻟﻌﻣدة .. اﻧﺎ ﻗررت ﺑﻌد اﻣﺎ اﺑﯾﻊ ﻣﺣﺻول اﻟﺑﻧﺟر و اﺑﯾﻌﮫ ... ھﯾﺟﻠﻰ ﻓﻠوس ﯾﺎ اﺑﺎ
اﻟﻌﻣدة ﻗررت ﺑﻘﻰ ﻋﻘﺑﺎل اﻣﻠﺗك و اﻣﻠت ﻣراﺗك ... اﺗﺟوز
اﻟﻌﻣدة : ﺗﺗﺟوز ... ﺗﺗﺟوز اﯾﮫ ﯾﺎ وﻻاااا ... اﻧت ﻣش ﻣﺗﺟوز اﺗﻧﯾن ﺑﺣﺎﻟﮭم
ﺟﻌﯾدى : اﺳﻣﻊ ﺑس ﯾﺎ ﻋﻣدة ... اﻧﺎ ﻗوﻟت اروح ﻋم اﻟﺣﺎج ﻣﺗوﻟﻰ و ادﯾﻠﮫ ال200 ﺟﻧﯾﮫ ﺑﺗﺎع ﻣﺣﺻول
اﻟﺑﻧﺟر ... و اﺗﺟوز ﺑﻧﺗﮫ ﺑدرﯾﺔ ... ﺑﻧﺗﮫ ﺑدرﯾﺔ دى ﯾﺎﺑﺎ اﻟﻌﻣدة ﺷﻔﺗﮭﺎ وھﻰ ﺑﺗﻐﺳل اﻟﺣﻠل ﻋﻧد
اﻟﺗرﻋﺔ .. و ﻛﻣﺎن ( ﯾوﺷوش اﻟﻌﻣدة )
اﻟﻌﻣدة : ان ﺟﯾت ﻟﻠﺣق ... اﻟﺑت ﺑدرﯾﺔ دى ... وﻻاا .. اﻧت ھﺗﺧرﺟﻧﻰ ﻋن ﺷﻌورى ﻟﯾﮫ ﯾﺎ وﻻ .. اﻧت ﻣش
ﻣﺗﺟوز اﺗﻧﯾن ﻋﺎوز اﯾﮫ ﺗﺎﻧﻰ
ﺟﻌﯾدى : ﺷﻌور اﯾﮫ ﯾﺎﺑﺎ اﻟﻌﻣدة داﻧت اﻗرع ﯾﺎ ﯾﺎﺑﺎ اﻟﻌﻣدة ... ﺑص ﺑس اﻧﺎ ھﻘوﻟك اﻟﻣوﺿوع ... ﺑس ﺧﻠﻰ
ﺑﺎﻟك اﺣﺳن ﯾطﺑوا ﻓﺟﺄة ... اﻧت ﻋﺎرف ﻟو ﺟم ھﺎﻛﻠك ﻋﻠﻰ ﻗﻔﺎك اﻧﺎ ﺑﻘوﻟك اھو
اﻟﻌﻣدة: اﻧت اھﺑل ﯾﺎاﻻاا .. ﺗدى ﻟﻣﯾن ﺑﺎﻟﻘﻔﺔ .. اﻧت اﺗﺟﻧﻧت ﺻﺣﯾﺢ ... ﺣﺳﺑﻰ ﷲ و ﻧﻌم اﻟوﻛﯾل ﺧدﯾﺟﺔ
ﺧﻠﻔت واﺣد اھﺑل زﯾﻰ ﺑﺎﻟظﺑط
ﺟﻌﯾدى : ﻣﺎ ھﻰ اﻟﮭﺑﻠﺔ ﺑﯾﻌرﻓوھﺎ ﻣن اﯾﯾﮫ ..
اﻟﻌﻣدة : ﻣن اﯾﮫ ﯾﺎ وﻻ
ﺟﻌﯾدى : ﻣن ﺧﻠﻔﺗﮭﺎ ... ﺧﻠﯾﻧﺎ ﺑﻘﻰ ﻓﻰ اﻟﻣوﺿوع ... ﻣش رﺑﻧﺎ ﺑﯾﻘول ﻓﻰ اﻟﻘرآن ... اﻟﻣﺎل و اﻟﺑﻧون زﯾﻧﺔ
اﻟﺣﯾﺎة اﻟدﻧﯾﺎ ... اﻧﺎ ﺑﻘﻰ ﻣﻌﺎﯾﺎ ال200 ﺟﻧﯾﮫ ﺑﺗوع ﻣﺣﺻول اﻟﺑﻧﺟر ... ﯾﺑﻘﻰ ﻓﺎﺿل اﻟﺑﻧون
اﻟﻌﻣدة : ﻻ اﻟﮫ اﻻ ﷲ ... اﻧت ھﺗﻔﺳر اﻵﯾﺔﻋﻠﻰ ﻣزاﺟك ﯾﺎ وﻻ ... رﺑﻧﺎ ﯾﻘﺻد ﺑﺎﻟﺑﻧون اﻟﺑﻧﺎت و اﻟﺻﺑﯾﺎن
ﻣش اﻟﺻﺑﯾﺎن ﺑس ﯾﺎ وﻻ
ﺟﻌﯾدى : اﻧﺎ ھﻔﮭﻣك ﯾﺎﺑﺎ اﻟﻌﻣدة ... ( ﯾﺳﻣﻊ اﺻوات ﻣرﺗﺎﺗﮫ ) ... ﯾﺎ ﻧﮭﺎار اﺳود و ﻣﻧﯾل ... اﻟﺣﻘﻧﻰ ﯾﺎﺑﺎ
اﻟﻌﻣدة .... ﺧﺑﯾﻧﻰ
ﺳﻌدﯾﺔ – ﺻﺑرﯾﺔ : ھو ﻓﯾﯾﯾن .. ھو ﻓﯾن اﻟﻣﻧﯾل ﻋﻠﻰ ﻋﯾﻧﮫ ... اﻧطق ﯾﺎﺑﺎ اﻟﻌﻣدة اﺣﺳن ﻧﻘﺗﻠك و ﻧﻘﺗﻠﮫ
اﻟﻌﻣدة : ﺑس ﯾﺎ ﺑت ﻣﻧك ﻟﯾﮭﺎ
ﺳﻌدﯾﺔ : ﯾرﺿﯾك ﯾﺎﺑﺎ اﻟﻌﻣدة .. اﻟﻣﻧﯾل ﻋﻠﻰ ﻋﯾﻧﮫ ﻋﺎوز ﯾﺗﺟوز ﻋﻠﯾﻧﺎ ﺑﻌد اﻟﻌﻣر ده ﻛﻠﮫ .. طب ﻟﻣﺎ اﺷوﻓك
ﺻﺑرﯾﺔ : ﺑس ﯾﺎ اﺧﺗﻰ ﺑس .. ﻣﺗزﻋﻠﯾش ﻧﻔﺳك
اﻟﻌﻣدة : ﺑس ﺧﻼص .. اﻧﻰ ﻋرﻓت ﻣﺷﻛﻠﺗﻛم و ھﺣﻠﮭﺎ ان ﺷﺎء ﷲ ﻋن ﻗرﯾب
ﺻﺑرﯾﺔ : ( ﺑدﻟﻊ ) ﯾﻌﻧﻰ .. ھﺗﺟﺑﻠﻰ ﺣﻘﻰ ﻣﻧﮫ ﯾﺎ ﻋﻣدة
اﻟﻌﻣدة : اﯾوة ھﺟﯾﺑﮫ ﻋﺷﺎن ﺧﺎطر ﻋﯾوﻧك ﯾﺎ ﻗﻣر
ﺳﻌدﯾﺔ : ( ﺑﻐﯾظ ) و اﻧﺎ ﻛﻣﺎن .. اﻧﺎ ﻛﻣﺎن ﻟﯾﺎ ﺣق
اﻟﻌﻣدة : اه طﺑﻌﺎ طﺑﻌﺎ .. ﻻزم اﺟﯾﺑﮫ طﺑﻌﺎ ... اﻧت ﯾﺎ واد ﯾﺎ ﺟﻌﯾدى ... ﺧﻠﺻت ﻓﻰ اﻟﺣﻣﺎم وﻻ ﻟﺳﮫ
```

---

- Doesn’t have the ability to deal with some word characters, like: (like in all Test Pdf files)

```python
Wrong : األدب كامال | Original: الأدب كاملا
```

---

- there is a problem If there are brackets in the text, like:  (like in Test (2) Pdf file)

```python
# process the extracted text

path = "Test (2).pdf" # path to file

# fucntion to save the extracted text in a text file
def save_extracted_text(path, output_text):
    f = open('After/' + path, 'w')
    # write the output_text in a text file
    f.write(output_text)
    f.close()

# to solve the problem of fliping the text.
# this solution took from my colleague hazem_33 https://discord.com/channels/1219261069005160468/1222950170916950129/1222973505411485726
def process(path):

    with pdfplumber.open(path) as pdf:

        now = datetime.datetime.now()

        # iterate over each page
        text=[]
        for ind, page in enumerate(pdf.pages):
            # extract and reverse text
            text.append(algorithm.get_display(
                page.extract_text(keep_blank_chars=True))
            )

        stop = datetime.datetime.now()

        time_processed = stop - now

    return "\n".join(text), time_processed

# print the output after applying read_pdf function on all files
output_second, time_second_processed = process(path)
print('Time for processing: ', time_file[0], '\n', '\nOutput text after processing: \n \n', output_second)
save_extracted_text(path.replace('.pdf', '.txt') , output_second)
```

```python
Wrong: ) تهيأت لشوقي عوامل لم تتهيأ لغيره ( فما تلك العومل ؟ 
Original: تهيأت لشوقي عوامل لم تتهيأ لغيره)  فما تلك العومل ؟)
```

---

- If there are images in the pdf file, there are no spaces between the words in some of the output text, like: (like in Test (4) and Test (5) Pdf files)

```python
# process the extracted text

path = "Test (5).pdf" # path to file

# fucntion to save the extracted text in a text file
def save_extracted_processed(path, output_text):
    f = open('After/' + path, 'w')
    # write the output_text in a text file
    f.write(output_text)
    f.close()

# to solve the problem of fliping the text.
# this solution took from my colleague hazem_33 https://discord.com/channels/1219261069005160468/1222950170916950129/1222973505411485726
def process(path):

    with pdfplumber.open(path) as pdf:

        now = datetime.datetime.now()

        # iterate over each page
        text=[]
        for ind, page in enumerate(pdf.pages):
            # extract and reverse text
            text.append(algorithm.get_display(
                page.extract_text(keep_blank_chars=True))
            )

        stop = datetime.datetime.now()

        time_processed = stop - now

    return "\n".join(text), time_processed

# print the output after applying read_pdf function on all files
output_fifth, time_fifth_processed = process(path)
print('Time for processing: ', time_fifth_processed, '\n', '\nOutput text after processing: \n \n', output_fifth)
save_extracted_processed(path.replace('.pdf', '.txt') , output_fifth)
```

```python
Wrong: ﺑﻨﺎناﻷﺻﺒﻊ،وﺟﺴﻤﻲﺟﻤﻴﻞﻟﻮﻧﻪأﺧﴬﻻﻣﻊ،أ ﱠﻣﺎأﺟﻨﺤﺘﻲاﻷﻣﺎﻣﻴﺔﻓﻘﻮﻳﺔوﻟﻮﻧﻬﺎﺿﺎرب
Original: بنان الأصبع، وجسمي جميل لونه أخضر لامع، أما أجنحتي الأمامية فقوية ولونها ضارب
```

---

- It can’t extract text with emojis:

```python
Original: الأسد 😞 يزأر أيها الثعلب. | Wrong: األسد يزأر( أيها الثعلب .
```

---

## Metrics

**📏Levenshtein Distance📏|📐Cosine Similarity📐|📊Tf-Idf Similarity📊|⏰Time to extract text⏰**

---

## Results

> Before Processing
> 

| File | Time to extract text Before Processing | Levenshtein distance Before Processing | Cosine similarity Before Processing | Tf-idf similarity Before Processing |
| --- | --- | --- | --- | --- |
| First File | 0 days 00:00:01.610547 | 5682 | 0.000000 | 0.000000 |
| Second File | 0 days 00:00:03.747712 | 33227 | 0.332032 | 0.202314 |
| Third File | 0 days 00:00:02.557903 | 7472 | 0.000000 | 0.000000 |
| Fourth File | 0 days 00:00:00.713611 | 5423 | 0.006501 | 0.003302 |
| Fifth File | 0 days 00:00:00.532846 | 4089 | 0.000478 | 0.000242 |

---

> After Processing
> 

| File | Time to extract text While Processing | Levenshtein distance After Processing | Cosine similarity After Processing | Tf-idf similarity After Processing |
| --- | --- | --- | --- | --- |
| First File | 0 days 00:00:01.294513 | 3332 | 0.011442 | 0.005826 |
| Second File | 0 days 00:00:03.809785 | 4979 | 0.620761 | 0.580050 |
| Third File | 0 days 00:00:01.587939 | 4983 | 0.004233 | 0.002147 |
| Fourth File | 0 days 00:00:00.404708 | 4429 | 0.018897 | 0.009663 |
| Fifth File | 0 days 00:00:00.421190 | 3299 | 0.010459 | 0.005329 |
