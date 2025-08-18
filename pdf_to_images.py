"""
pdf_to_images.py
----------------
Module to convert a PDF file into page-wise PNG images using PyMuPDF (fitz).
No Poppler dependency required.

Usage:
    python pdf_to_images.py <pdf_file>

Example:
    python pdf_to_images.py sample_data/217506282.pdf
"""

import sys
import os
import fitz  # PyMuPDF


def pdfToImages(pdf_path: str) -> list:
    """
    Convert a PDF into PNG images and save them alongside the PDF.

    Args:
        pdf_path (str): Path to input PDF file.

    Returns:
        list: List of image file paths created.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    pdf_dir = os.path.dirname(pdf_path)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    output_files = []

    # Open the PDF
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc[page_num]

        # Render page to a pixmap (image)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better resolution

        out_file = os.path.join(pdf_dir, f"{pdf_name}_{page_num + 1}.png")
        pix.save(out_file)
        output_files.append(out_file)

    doc.close()
    return output_files


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pdf_to_images.py <pdf_file>")
        sys.exit(1)

    pdf_file = sys.argv[1]
    try:
        images = pdfToImages(pdf_file)
        print("✅ Success. Created image files:")
        for img in images:
            print(img)
    except Exception as e:
        print(f"❌ Failed: {e}")
