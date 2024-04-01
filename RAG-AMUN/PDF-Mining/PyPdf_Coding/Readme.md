# PyPDF2

**What is PyPDF2?**

PyPDF2 is a free and open-source pure-Python PDF library capable of various operations such as splitting, merging, cropping, and transforming pages of PDF files. Additionally, it can add custom data, viewing options, and passwords to PDF files. PyPDF2 also offers functionalities to retrieve text and metadata from PDFs.

[![PyPI version](https://badge.fury.io/py/pypdf.svg)](https://badge.fury.io/py/pypdf) [![Documentation](https://img.shields.io/badge/-documentation-green)](https://pypdf.readthedocs.io/en/stable/) [![Github](https://img.shields.io/badge/Github_Source-gray)](https://github.com/py-pdf/pypdf)

**Task Description:**

Your task is to evaluate the PyPDF2 PDF text extraction tool by running it on a provided sample PDF file. Your evaluation should include a comprehensive analysis of the tool, highlighting its strengths and weaknesses.

**What to Submit:**

1. **Text Files:** You need to submit text files containing the extracted text from the PDF. The name of each text file should match the name of the corresponding PDF file. These text files will be used in a matrix calculation to assess the performance of the tool. Optimize your code to maximize your score in the evaluation matrix.

2. **Documented Analysis:** Provide a detailed analysis documenting your observations and conclusions based on your exploration and evaluation of the PyPDF2 tool.

3. **Code:** Clone the provided GitHub repository containing the code template for text extraction. Add your code to the repository, commit the changes, and push them to the remote repository.


**[Sample Text](https://github.com/yazeedmshayekh2/SMSM-Internship/tree/main/RAG-AMUN/PDF-Mining)**

Ensure that your submission is well-organized and follows the guidelines provided.

<a target="_blank" href="https://colab.research.google.com/drive/1qEcyERknOXANiFs7tUS01IvcxWc2DpwM#scrollTo=i2LnKNVI6VyQ">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

___________________________________________________________

# PyPdf Report About using it with Arabic Pdf Files

- You can find it [Here](https://elite-height-60d.notion.site/PyPdf2-Report-b2032324a4234d628713cf0ec952ffd3?pvs=4)

- Or:

# PyPdf2 - Report

# TODOS

- [x]  Try to solve the problems in the three pdf files.
- [ ]  Try to solve the problems in the fourth and fifth pdf files.
- [ ]  Using Metrics to find the similarity between the extracted text and the text in the original pdf file.
- [ ]  Try to use embedding for each file and then calculate the cosine similarity between the embedding of the text in the original pdf file and the extracted text.

---

> Code Here on Colab:
> 

[Google Colaboratory](https://colab.research.google.com/drive/1qEcyERknOXANiFs7tUS01IvcxWc2DpwM?usp=sharing)

| Original PDF file | Extracted File | https://github.com/yazeedmshayekh2/SMSM-Internship/tree/main/RAG-AMUN | TODO - Similarity Score  |
| --- | --- | --- | --- |
| Test (1).pdf (https://www.notion.so/Test-1-pdf-2a77c9465479486c9eb7ea88565f5e70?pvs=21)  | Test (1).txt (https://www.notion.so/Test-1-txt-c75eba82c13340e184d8a5f2546fa9ab?pvs=21)  | Problems While Using PyPdf2:

1. First, it prints every word in a separate line.

2. Problems with the 
structure of the pages, it can’t recognize the page structure, like : it prints (العمدة:/جعيدي: على نفس السطر بينما لازم يكون كل واحد عل سطر مختلف)

3. Flip “(” with “)” and vice versa. |  |
| Test (2).pdf (https://www.notion.so/Test-2-pdf-a19325b3da69470c97d3b7aa8ddbcb9a?pvs=21)  | Test (2).txt (https://www.notion.so/Test-2-txt-60c711971b234944bc00ceabebf64abb?pvs=21)  | Problems While Using PyPdf2:

1. Flip “(” with “)” and vice versa. |  |
| Test (3).pdf (https://www.notion.so/Test-3-pdf-4f5e3cbe832c4ab7ac45bf744e8fc1ce?pvs=21)  | Test (3).txt (https://www.notion.so/Test-3-txt-5f47be94aa8a4873937a26dfbd6afc8c?pvs=21)  | Problems While Using PyPdf2:

1. First, it prints every word in a separate line.

2. Problems with the 
structure of the pages, it can’t recognize the page structure, like : it prints (العمدة:/جعيدي: على نفس السطر بينما لازم يكون كل واحد عل سطر مختلف)

3. Flip “(” with “)” and vice versa. |  |
| Test (4).pdf (https://www.notion.so/Test-4-pdf-992b57d0e4da412dae329e70a7a030be?pvs=21)  | Test (4).txt (https://www.notion.so/Test-4-txt-617d69168aaf47b2893cf381ddcefb1b?pvs=21)  | Problems While Using PyPdf2:

1. It prints the sentence without spaces between Arabic words.

2. Problems with the 
structure of the pages, it can’t recognize the page structure, like : it prints (المﺸﻬﺮةﺑﺮﻗﻢ٠٧٩٥٨٥٠١ﺑﺘﺎرﻳﺦ٦٢ / ١ / ٧١٠٢ ) : (املشهرة برقم ١٠٥٨٥٩٧٠ بتاريخ ٢٦ / ١ / ٢٠١٧)
 
3. Flip “(” with “)” and vice versa. |  |
| Test (5).pdf (https://www.notion.so/Test-5-pdf-ff4daeb8cf7943ee8a3531fffd32e463?pvs=21)  | Test (5).txt (https://www.notion.so/Test-5-txt-583492ddc3904ca8928a901945fd08c6?pvs=21)  | Problems While Using PyPdf2:

1. It prints the sentence without spaces between Arabic words.

2. Problems with the 
structure of the pages, it can’t recognize the page structure, like : it prints (المﺸﻬﺮةﺑﺮﻗﻢ٠٧٩٥٨٥٠١ﺑﺘﺎرﻳﺦ٦٢ / ١ / ٧١٠٢ ) : (املشهرة برقم ١٠٥٨٥٩٧٠ بتاريخ ٢٦ / ١ / ٢٠١٧)
 
3. Flip “(” with “)” and vice versa. |  |

---

[Test (1).pdf](PyPdf2%20-%20Report%20b2032324a4234d628713cf0ec952ffd3/Test_(1).pdf)

[Test (1).txt](PyPdf2%20-%20Report%20b2032324a4234d628713cf0ec952ffd3/Test_(1).txt)

[Test (2).pdf](PyPdf2%20-%20Report%20b2032324a4234d628713cf0ec952ffd3/Test_(2).pdf)

[Test (2).txt](PyPdf2%20-%20Report%20b2032324a4234d628713cf0ec952ffd3/Test_(2).txt)

[Test (3).pdf](PyPdf2%20-%20Report%20b2032324a4234d628713cf0ec952ffd3/Test_(3).pdf)

[Test (3).txt](PyPdf2%20-%20Report%20b2032324a4234d628713cf0ec952ffd3/Test_(3).txt)

[Test (4).pdf](PyPdf2%20-%20Report%20b2032324a4234d628713cf0ec952ffd3/Test_(4).pdf)

[Test (4).txt](PyPdf2%20-%20Report%20b2032324a4234d628713cf0ec952ffd3/Test_(4).txt)

[Test (5).pdf](PyPdf2%20-%20Report%20b2032324a4234d628713cf0ec952ffd3/Test_(5).pdf)

[Test (5).txt](PyPdf2%20-%20Report%20b2032324a4234d628713cf0ec952ffd3/Test_(5).txt)

---
