#!/usr/bin/env python3
# PDF Data Extractor

# IMPORTANT below is the pip query for everything this project uses
# pip install numpy matplotlib pandas wordcloud pytesseract pdf2image pillow
# you also need to install tesseract separately (sudo dnf install tesseract tesseract-devel tesseract-langpack-eng if you are on fedora like me)
# arguments with their explanations are above the main() function near the bottom of the code

# using ocr is currently pretty slow because i am still new to parallelization and didn't implement it
# but i will try to do it soon just to speed everything up a bit

import os
import re
import sys
import json
import argparse
import logging
from datetime import datetime
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from wordcloud import WordCloud

# libraries for pdf processing
import PyPDF2
import pdfplumber

# setting up basic logging (it's all in one file, does not create a new one for each PDF) but that's easily done 
# if you really need it by modifying the log name below to include the name of the PDF
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pdfdataextractor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("pdfdataextractor")

# class which represents the PDF analysis instance. every different aspect has a separate function so it looks cleaner
class PDFAnalyzer:
    """main class for analyzing PDF documents."""
    
    def __init__(self, file_path: str, output_dir: str = "output", ocr_enabled: bool = False):
        """
        initialize the analyzer with file path and options
        
        args:
            file_path: path to the pdf file
            output_dir: directory to save results
            ocr_enabled: whether to use ocr for text extraction
        """
        self.file_path = os.path.abspath(file_path)
        self.filename = os.path.basename(file_path)
        self.output_dir = output_dir
        self.ocr_enabled = ocr_enabled
        
        # make sure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # data containers
        self.metadata = {}
        self.text_content = []
        self.pages = []
        self.images = []
        self.tables = []
        self.form_fields = []
        self.word_frequencies = Counter()
        self.entity_mentions = defaultdict(list)
        
        # analysis results
        self.stats = {}
        self.page_count = 0
        self.top_entities = {}
        
        logger.info(f"Initialized analyzer for {self.filename}")
        
        # check for ocr support
        if ocr_enabled:
            try:
                import pytesseract
                from PIL import Image
                from pdf2image import convert_from_path
                self.pytesseract_available = True
                logger.info("OCR support enabled.")
            except ImportError as e:
                logger.warning(f"OCR requested but dependencies not available: {str(e)}")
                logger.warning("OCR will be disabled. Install with: pip install pytesseract pdf2image pillow")
                self.pytesseract_available = False
                self.ocr_enabled = False
        else:
            self.pytesseract_available = False
    
    def extract_all(self) -> None:
        """extract all data from the pdf document"""
        try:
            self._extract_with_pypdf()
            self._extract_with_pdfplumber()
            
            if self.ocr_enabled and self.pytesseract_available:
                self._perform_ocr()
                
            self._process_text_content()
            self._calculate_statistics()
            
            logger.info(f"Completed extraction for {self.filename}")
        except Exception as e:
            logger.error(f"Error during extraction: {str(e)}")
            raise
    
    def _extract_with_pypdf(self) -> None:
        """extract basic metadata and text using pypdf"""
        logger.info("Extracting with PyPDF2...")
        
        try:
            # suppress warnings from pdfminer about missing cropbox because it generates one entry for each page and it takes up like 95% of the log lol
            logging.getLogger('pdfminer').setLevel(logging.ERROR)
            
            with open(self.file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                self.page_count = len(reader.pages)
                
                # extract metadata
                info = reader.metadata
                if info:
                    self.metadata = {
                        'title': info.get('/Title', ''),
                        'author': info.get('/Author', ''),
                        'subject': info.get('/Subject', ''),
                        'creator': info.get('/Creator', ''),
                        'producer': info.get('/Producer', ''),
                        'creation_date': info.get('/CreationDate', ''),
                        'modification_date': info.get('/ModDate', '')
                    }
                
                # extract text from each page
                for i, page in enumerate(reader.pages):
                    text = page.extract_text() or ""  # handle None case
                    self.text_content.append({
                        'page_num': i + 1,
                        'text': text,
                        'word_count': len(text.split()) if text else 0
                    })
        except Exception as e:
            logger.error(f"PyPDF2 extraction error: {str(e)}")
            raise
    
    def _extract_with_pdfplumber(self) -> None:
        """extract tables, form fields, and additional text using pdfplumber"""
        logger.info("Extracting with pdfplumber...")
        
        try:
            with pdfplumber.open(self.file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    # this is to handle potential page issues i encountered
                    try:
                        has_text = bool(page.extract_text())
                    except Exception:
                        has_text = False
                        
                    try:
                        has_images = bool(page.images)
                    except Exception:
                        has_images = False
                        
                    try:
                        has_tables = bool(page.find_tables())
                    except Exception:
                        has_tables = False
                    
                    page_data = {
                        'page_num': i + 1,
                        'width': getattr(page, 'width', 0),
                        'height': getattr(page, 'height', 0),
                        'has_text': has_text,
                        'has_images': has_images,
                        'has_tables': has_tables,
                    }
                    self.pages.append(page_data)
                    
                    # extract tables with error handling
                    try:
                        tables = page.extract_tables()
                        if tables:
                            for j, table in enumerate(tables):
                                if table:  # some tables might be empty
                                    # convert table to simpler and more usable format
                                    clean_table = [[cell or '' for cell in row] for row in table]
                                    table_data = {
                                        'page_num': i + 1,
                                        'table_num': j + 1,
                                        'data': clean_table,
                                        'rows': len(clean_table),
                                        'cols': len(clean_table[0]) if clean_table else 0
                                    }
                                    self.tables.append(table_data)
                    except Exception as e:
                        logger.warning(f"Table extraction failed on page {i+1}: {str(e)}")
                    
                    # extract images with error handling
                    try:
                        if page.images:
                            for j, img in enumerate(page.images):
                                image_data = {
                                    'page_num': i + 1,
                                    'image_num': j + 1,
                                    'width': img.get('width', 0),
                                    'height': img.get('height', 0),
                                    'x0': img.get('x0', 0),
                                    'y0': img.get('y0', 0),
                                    'x1': img.get('x1', 0),
                                    'y1': img.get('y1', 0),
                                }
                                self.images.append(image_data)
                    except Exception as e:
                        logger.warning(f"Image extraction failed on page {i+1}: {str(e)}")
        except Exception as e:
            logger.error(f"pdfplumber extraction error: {str(e)}")
            raise
    
    def _perform_ocr(self) -> None:
        """perform ocr on document pages to extract text from images. first testing if installed and available"""
        if not self.pytesseract_available:
            logger.warning("OCR requested but dependencies not available. Skipping OCR.")
            return
            
        logger.info("Performing OCR on document...")
        
        try:
            # importing here to avoid issues if not installed (later fixed but just in case)
            import pytesseract
            from PIL import Image
            from pdf2image import convert_from_path
            
            # trying to check if tesseract is properly installed for easier troubleshooting
            try:
                pytesseract.get_tesseract_version()
                logger.info(f"Tesseract version: {pytesseract.get_tesseract_version()}")
            except Exception as e:
                logger.error(f"Tesseract not properly configured: {str(e)}")
                logger.error("Make sure Tesseract is installed and in your PATH")
                logger.error("On Windows: set the tesseract_cmd path explicitly")
                return
            
            # convert pdf pages to images
            try:
                pages = convert_from_path(self.file_path, 300)  # using 300 DPI for better OCR
            except Exception as e:
                logger.error(f"Failed to convert PDF to images: {str(e)}")
                return
            
            for i, page_image in enumerate(pages):
                # use pytesseract to extract text from pages
                try:
                    text = pytesseract.image_to_string(page_image)
                except Exception as e:
                    logger.error(f"OCR failed for page {i+1}: {str(e)}")
                    continue
                
                # append to or update existing text content
                page_idx = i
                if page_idx < len(self.text_content):
                    # if we already have text for this page, append OCR text
                    existing_text = self.text_content[page_idx]['text']
                    combined_text = f"{existing_text}\n\n--- OCR TEXT ---\n{text}"
                    self.text_content[page_idx]['text'] = combined_text
                    self.text_content[page_idx]['word_count'] = len(combined_text.split())
                else:
                    # if this is a new page (shouldn't happen but just in case), add it
                    self.text_content.append({
                        'page_num': i + 1,
                        'text': text,
                        'word_count': len(text.split()) if text else 0
                    })
                
                # save the page image for reference
                img_output_path = os.path.join(self.output_dir, f"page_{i+1}.png")
                page_image.save(img_output_path, "PNG")
        except Exception as e:
            logger.error(f"OCR error: {str(e)}")
            logger.warning("OCR processing failed, continuing with other extraction methods")
    
    def _process_text_content(self) -> None:
        """process extracted text to gather statistics and patterns etc"""
        logger.info("Processing text content...")
        
        all_text = ' '.join([page['text'] for page in self.text_content if page['text']])
        
        # calculate word frequencies
        words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
        self.word_frequencies = Counter(words)
        
        # extract potential named entities (simple approach)
        # this is a basic implementation. would use nlp library if more serious
        capitalized_words = re.findall(r'\b[A-Z][a-zA-Z]{1,}\b', all_text)
        potential_entities = set()
        for word in capitalized_words:
            if len(word) > 1 and word.lower() not in ['i', 'a', 'the', 'and', 'but']:
                potential_entities.add(word)
        
        # find mentions of potential entities on each page
        for entity in potential_entities:
            pattern = re.compile(r'\b' + re.escape(entity) + r'\b')
            for i, page_data in enumerate(self.text_content):
                if page_data['text']:
                    matches = pattern.findall(page_data['text'])
                    if matches:
                        self.entity_mentions[entity].append({
                            'page_num': i + 1,
                            'count': len(matches)
                        })
    
    def _calculate_statistics(self) -> None:
        """calculate overall document statistics"""
        # basic stats
        total_words = sum(page['word_count'] for page in self.text_content)
        avg_words_per_page = total_words / self.page_count if self.page_count > 0 else 0
        
        # readability measures. very basic implementation
        all_text = ' '.join([page['text'] for page in self.text_content if page['text']])
        sentences = re.split(r'[.!?]+', all_text)
        sentence_count = len([s for s in sentences if len(s.strip()) > 0])
        avg_sentence_length = total_words / sentence_count if sentence_count > 0 else 0
        
        # calculate top entities
        self.top_entities = {
            entity: sum(mention['count'] for mention in mentions)
            for entity, mentions in sorted(
                self.entity_mentions.items(), 
                key=lambda x: sum(mention['count'] for mention in x[1]), 
                reverse=True
            )[:10]  # top 10 entities
        }
        
        # collect statistics
        self.stats = {
            'page_count': self.page_count,
            'total_words': total_words,
            'avg_words_per_page': round(avg_words_per_page, 2),
            'sentence_count': sentence_count,
            'avg_sentence_length': round(avg_sentence_length, 2),
            'image_count': len(self.images),
            'table_count': len(self.tables),
            'most_common_words': dict(self.word_frequencies.most_common(20)),
            'entity_count': len(self.entity_mentions)
        }
    
    def generate_report(self) -> Dict[str, Any]:
        """generate a comprehensive report of the analysis"""
        logger.info("Generating analysis report...")
        
        report = {
            'filename': self.filename,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metadata': self.metadata,
            'statistics': self.stats,
            'page_details': self.pages,
            'top_entities': self.top_entities
        }
        
        # save the report as json
        report_path = os.path.join(self.output_dir, f"{os.path.splitext(self.filename)[0]}_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        return report
    
    def visualize_data(self) -> None:
        """
        create visualizations of the analysis results
        (will likely add exceptions so words like "the", "and", "you", "to" and such are not counted since they will obviously always be among the top)
        """
        logger.info("Creating visualizations...")
        
        # prepare the output pdf for visualizations
        viz_path = os.path.join(self.output_dir, f"{os.path.splitext(self.filename)[0]}_visualizations.pdf")
        
        with PdfPages(viz_path) as pdf:
            # word frequency visualization
            plt.figure(figsize=(10, 6))
            words = [word for word, count in self.word_frequencies.most_common(20)]
            counts = [count for word, count in self.word_frequencies.most_common(20)]
            
            if words and counts:  # check if the words to visualize are ready
                plt.barh(words, counts, color='skyblue')
                plt.xlabel('Frequency')
                plt.ylabel('Words')
                plt.title('Top 20 Most Frequent Words')
                plt.tight_layout()
                pdf.savefig()
            plt.close()
            
            # words per page graph
            plt.figure(figsize=(10, 6))
            page_nums = [page['page_num'] for page in self.text_content]
            word_counts = [page['word_count'] for page in self.text_content]
            
            if page_nums and word_counts:  # check if data to visualize is ready
                plt.bar(page_nums, word_counts, color='lightgreen')
                plt.xlabel('Page Number')
                plt.ylabel('Word Count')
                plt.title('Words per Page')
                plt.tight_layout()
                pdf.savefig()
            plt.close()
            
            # entity mentions
            if self.top_entities:
                plt.figure(figsize=(10, 6))
                entities = list(self.top_entities.keys())[:10]  # top 10 entities
                mentions = list(self.top_entities.values())[:10]
                
                if entities and mentions:  # check if entities to visualize are ready
                    plt.barh(entities, mentions, color='salmon')
                    plt.xlabel('Mention Count')
                    plt.ylabel('Entity')
                    plt.title('Top Entity Mentions')
                    plt.tight_layout()
                    pdf.savefig()
                plt.close()
            
            # word cloud
            if self.word_frequencies:
                plt.figure(figsize=(10, 10))
                if len(self.word_frequencies) > 0:  # check if we have words for the cloud
                    wordcloud = WordCloud(
                        width=800, height=800,
                        background_color='white', # feel free to play around with this
                        min_font_size=10
                    ).generate_from_frequencies(self.word_frequencies)
                    
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    plt.title('Word Cloud')
                    plt.tight_layout()
                    pdf.savefig()
                plt.close()
            
            # document components visualization
            plt.figure(figsize=(8, 8))
            components = ['Pages', 'Images', 'Tables']
            counts = [self.stats['page_count'], self.stats['image_count'], self.stats['table_count']]
            
            # only create the pie chart if we have non-zero values
            if sum(counts) > 0:
                plt.pie(counts, labels=components, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightgreen', 'lightsalmon'])
                plt.axis('equal')
                plt.title('Document Components')
                plt.tight_layout()
                pdf.savefig()
            plt.close()
        
        logger.info(f"Visualizations saved to {viz_path}")
    
    def extract_tables_to_excel(self) -> None:
        """save extracted tables to an excel file"""
        if not self.tables:
            logger.info("No tables found to export")
            return
            
        logger.info("Exporting tables to Excel...")
        
        excel_path = os.path.join(self.output_dir, f"{os.path.splitext(self.filename)[0]}_tables.xlsx")
        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                for i, table in enumerate(self.tables):
                    try:
                        df = pd.DataFrame(table['data'])
                        # use first row as header if it looks like a header
                        if self._is_likely_header(table['data']):
                            df.columns = table['data'][0]
                            df = df.iloc[1:]
                        
                        # Ensure sheet name is valid for excel
                        sheet_name = f"Table_{table['page_num']}_{table['table_num']}"
                        if len(sheet_name) > 31:  # excel sheet name limit
                            sheet_name = sheet_name[:31]
                        
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                    except Exception as e:
                        logger.warning(f"Failed to export table {i+1}: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to create Excel file: {str(e)}")
            return
        
        logger.info(f"Tables exported to {excel_path}")
    
    def _is_likely_header(self, table_data: List[List[str]]) -> bool:
        """check if the first row of a table is likely a header row"""
        if not table_data or len(table_data) < 2:
            return False
            
        first_row = table_data[0]
        second_row = table_data[1]
        
        # check if first row has shorter text elements than second row
        first_row_len = sum(len(str(cell)) for cell in first_row)
        second_row_len = sum(len(str(cell)) for cell in second_row)
        
        # check if first row has more capitalized words
        first_row_caps = sum(1 for cell in first_row if str(cell).istitle() or str(cell).isupper())
        second_row_caps = sum(1 for cell in second_row if str(cell).istitle() or str(cell).isupper())
        
        return (first_row_len < second_row_len) or (first_row_caps > second_row_caps)
    
    def extract_text_to_file(self) -> None:
        """save extracted text content to a text file"""
        logger.info("Saving extracted text...")
        
        text_path = os.path.join(self.output_dir, f"{os.path.splitext(self.filename)[0]}_text.txt")
        try:
            with open(text_path, 'w', encoding='utf-8') as f:
                for page in self.text_content:
                    f.write(f"--- PAGE {page['page_num']} ---\n\n")
                    f.write(page['text'] or "")  # Handle None case
                    f.write("\n\n")
        except Exception as e:
            logger.error(f"Failed to write text file: {str(e)}")
            return
        
        logger.info(f"Text content saved to {text_path}")
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """run the complete analysis pipeline"""
        self.extract_all()
        report = self.generate_report()
        self.visualize_data()
        self.extract_tables_to_excel()
        self.extract_text_to_file()
        
        return report


def parse_arguments():
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description='PDF Document Analyzer')
    parser.add_argument('pdf_path', help='Path to the PDF file to analyze')
    parser.add_argument('--output', '-o', default='output', help='Output directory for analysis results')
    parser.add_argument('--ocr', action='store_true', help='Enable OCR for text extraction')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    return parser.parse_args()

# main starts here

def main():
    """main function to run the analyzer"""
    args = parse_arguments()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    if not os.path.exists(args.pdf_path):
        logger.error(f"PDF file not found: {args.pdf_path}")
        sys.exit(1)
    if (args.pdf_path.endswith (".epub")):
        logger.error(f"{args.pdf_path} is a .epub file instead of a .pdf file.")
        sys.exit(1)
    
    logger.info(f"Starting analysis of {args.pdf_path}")
    
    try:
        analyzer = PDFAnalyzer(args.pdf_path, args.output, args.ocr)
        report = analyzer.run_complete_analysis()
        
        logger.info(f"Analysis completed succesfully! Results saved to {args.output} directory.")
        logger.info(f"Document statistics: {report['statistics']['page_count']} pages, "
                   f"{report['statistics']['total_words']} words, "
                   f"{report['statistics']['image_count']} images, "
                   f"{report['statistics']['table_count']} tables.")
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()