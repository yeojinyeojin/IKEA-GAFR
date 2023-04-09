from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)


if __name__ == '__main__':
    pdf_path = '../dataset/ikea_man/pdfs/Bench/applaro/0.pdf'
    images = convert_from_path(pdf_path)
    print(type(images[0]))