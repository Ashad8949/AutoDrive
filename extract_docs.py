import docx
import sys
import os

def extract_text(doc_path):
    print(f"Extracting: {doc_path}")
    doc = docx.Document(doc_path)
    text = []
    for para in doc.paragraphs:
        text.append(para.text)
    
    out_path = doc_path.replace('.docx', '.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(text))
    print(f"Saved to: {out_path}")

if __name__ == '__main__':
    extract_text(r"c:\Users\ashad\Desktop\Projects\Cloud\AutoDrive_Implementation_Guide.docx")
    extract_text(r"c:\Users\ashad\Desktop\Projects\Cloud\Azure_Services_Reference.docx")
