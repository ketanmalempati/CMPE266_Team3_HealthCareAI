import pdfplumber
import regex

def cleanText(text):
    urlP = r'https?://\S+|www\.\S+'
    emailP = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    otherP = r'[^\x00-\x7F]+'
    footerP = r'^.*\b(Chapter|Section)\b.*$'
    citationP = r"\((((((\p{L}|-| )+((& (\p{L}|-| )+)|(,? et al.))?, )?\d{4}(, p\.\d+)?)|(p\.\d+));?)+\)"
    referencesP = r'\[\d+\]'


    text = regex.sub(referencesP, "", text)
    text = regex.sub(citationP, "", text)
    text = regex.sub(urlP, "", text)
    text = regex.sub(emailP, "", text)
    text = regex.sub(footerP, "", text, flags=regex.MULTILINE)
    text = regex.sub(otherP, "", text)

    text = regex.sub(r'\s+', ' ', text).strip()
    return text

def processPdf(input_path, output_path):
    with open(output_path, "w", encoding="utf-8") as output_file:
        with pdfplumber.open(input_path) as pdf:
            for page_number, page in enumerate(pdf.pages):
                try:
                    text = page.extract_text()
                    if text:
                        cleaned_text = cleanText(text)
                        output_file.write(cleaned_text + "\n")
                except Exception as e:
                    print(f"Error processing page {page_number}: {e}")
import glob  

for i in glob.glob("*.pdf"):
    inputPath = i
    name=i.split('.')
    outputPath = "cleaned/"+name[0]+".txt"
    processPdf(inputPath, outputPath)

    print("Text extraction of one file is complete.")