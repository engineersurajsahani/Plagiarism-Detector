from flask import Flask, render_template, request, flash, redirect, url_for
import os
import PyPDF2
import docx
import re
import html
import chardet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from database import create_connection, add_keyword, get_all_keywords, search_keywords
import sqlite3

app = Flask(__name__)
app.secret_key = 'your_secret_key'

UPLOAD_FOLDER = "uploaded_documents"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception:
        return None
    return text.strip()

def extract_text_from_docx(docx_path):
    try:
        doc = docx.Document(docx_path)
        return "\n".join([para.text for para in doc.paragraphs]).strip()
    except Exception:
        return None

def extract_text_from_txt(file_path):
    try:
        with open(file_path, "rb") as f:
            raw_data = f.read()
            detected_encoding = chardet.detect(raw_data)['encoding']
        with open(file_path, "r", encoding=detected_encoding or "utf-8") as f:
            return f.read().strip()
    except Exception:
        return None

def preprocess_text(text):
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def highlight_text(text, matched_phrases):
    text = html.escape(text)
    for phrase, _ in sorted(matched_phrases, key=lambda x: len(x[0]), reverse=True):
        text = re.sub(r'\b' + re.escape(phrase.lower()) + r'\b', 
                      f'<mark style="background-color: yellow; font-weight: bold;">{phrase}</mark>', 
                      text, flags=re.IGNORECASE)
    return text

def check_plagiarism(texts, use_keywords=False):
    cleaned_texts = [preprocess_text(text) for text in texts]
    
    if use_keywords:
        conn = create_connection()
        keyword_weights = {}
        found_keywords = []
        
        # Search for keywords in all texts combined
        combined_text = " ".join(cleaned_texts)
        found_keywords = search_keywords(conn, combined_text)
        
        if not found_keywords:
            # If no keywords found, fall back to regular detection
            flash("⚠ No keywords from database found in texts. Using standard detection method.", "warning")
            use_keywords = False
        else:
            keyword_weights = {kw.lower(): weight for kw, weight in found_keywords}
    
    if use_keywords:
        # Create a custom vocabulary with weights
        try:
            vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), 
                                        stop_words=None, norm='l2',
                                        vocabulary=keyword_weights.keys())
            tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
            
            # Apply keyword weights to the TF-IDF matrix
            for i, feature in enumerate(vectorizer.get_feature_names_out()):
                weight = keyword_weights.get(feature.lower(), 1.0)
                tfidf_matrix[:, i] = tfidf_matrix[:, i] * weight
        except ValueError as e:
            flash(f"⚠ Error in keyword-based detection: {str(e)}. Falling back to standard detection.", "warning")
            use_keywords = False
    
    if not use_keywords:
        # Standard detection without keywords
        vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), 
                                   stop_words=None, norm='l2')
        tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
    
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    results = []
    highlighted_texts = texts.copy()

    for i in range(len(cleaned_texts)):
        for j in range(i + 1, len(cleaned_texts)):
            similarity = round(similarity_matrix[i][j] * 100, 2)
            if similarity > 50:
                alert = '<span style="color: red; font-weight: bold;">⚠ High Plagiarism Alert!</span>'
            elif similarity >= 20:
                alert = '<span style="color: orange; font-weight: bold;">⚠ Moderate Plagiarism Alert!</span>'
            else:
                alert = '<span style="color: green; font-weight: bold;">✅ No Plagiarism Detected.</span>'

            if use_keywords:
                matched_phrases = [(kw, weight) for kw, weight in found_keywords 
                                 if kw.lower() in texts[i].lower() 
                                 and kw.lower() in texts[j].lower()]
                highlighted_texts[i] = highlight_text(texts[i], matched_phrases)
                highlighted_texts[j] = highlight_text(texts[j], matched_phrases)
            else:
                feature_names = vectorizer.get_feature_names_out()
                matched_phrases = [(phrase, 1.0) for phrase in feature_names 
                                 if phrase.lower() in texts[i].lower() 
                                 and phrase.lower() in texts[j].lower()]
                highlighted_texts[i] = highlight_text(texts[i], matched_phrases)
                highlighted_texts[j] = highlight_text(texts[j], matched_phrases)

            results.append({
                "doc1": i + 1,
                "doc2": j + 1,
                "similarity": similarity,
                "alert": alert,
                "highlighted_doc1": highlighted_texts[i],
                "highlighted_doc2": highlighted_texts[j]
            })
    return results, highlighted_texts

@app.route("/", methods=["GET", "POST"])
def index():
    uploaded_files = []
    extracted_texts = []
    file_names = []
    input_mode = "files"  # Default to file upload mode

    if request.method == "POST":
        input_mode = request.form.get("input_mode", "files")
        
        if input_mode == "files":
            files = request.files.getlist("files")
            
            if len(files) < 2:
                flash("⚠ Please upload at least two files for plagiarism checking.", "danger")
                return render_template("index.html", uploaded_files=[], results=[])
            
            for file in files:
                if file.filename:
                    if not allowed_file(file.filename):
                        flash(f"⚠ Unsupported file format: '{file.filename}'. Allowed types are PDF, DOCX, TXT.", "danger")
                        continue

                    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
                    file.save(file_path)
                    file_names.append(file.filename)

                    if file.filename.endswith(".pdf"):
                        extracted_text = extract_text_from_pdf(file_path)
                    elif file.filename.endswith(".docx"):
                        extracted_text = extract_text_from_docx(file_path)
                    elif file.filename.endswith(".txt"):
                        extracted_text = extract_text_from_txt(file_path)
                    else:
                        extracted_text = None

                    if extracted_text is None:
                        flash(f"⚠ The file '{file.filename}' is corrupt or unreadable!", "danger")
                    elif not extracted_text.strip():
                        flash(f"⚠ The file '{file.filename}' is empty!", "danger")
                    else:
                        extracted_texts.append(extracted_text)
            
            if len(extracted_texts) < 2:
                flash("⚠ Not enough valid files to perform plagiarism checking.", "danger")
                return render_template("index.html", uploaded_files=file_names, results=[])
            
        elif input_mode == "text":
            text1 = request.form.get("text1", "").strip()
            text2 = request.form.get("text2", "").strip()
            
            if not text1 or not text2:
                flash("⚠ Please provide text in both input boxes.", "danger")
                return render_template("index.html", uploaded_files=[], results=[])
            
            extracted_texts = [text1, text2]
            file_names = ["Text Input 1", "Text Input 2"]
        
        use_keywords = request.form.get("use_keywords") == "on"
        results, _ = check_plagiarism(extracted_texts, use_keywords)
        flash("Analysis completed successfully!", "success")
        return render_template("index.html", 
                             uploaded_files=file_names, 
                             results=results,
                             input_mode=input_mode,
                             text1=request.form.get("text1", ""),
                             text2=request.form.get("text2", ""))
    
    return render_template("index.html", 
                         uploaded_files=[], 
                         results=[],
                         input_mode=input_mode)

@app.route("/keywords", methods=["GET", "POST"])
def manage_keywords():
    conn = create_connection()
    if request.method == "POST":
        keyword = request.form.get("keyword", "").strip()
        weight = float(request.form.get("weight", 1.0))
        if keyword:
            add_keyword(conn, keyword, weight)
            flash(f"Keyword '{keyword}' added successfully!", "success")
    
    keywords = get_all_keywords(conn)
    return render_template("keywords.html", keywords=keywords)

if __name__ == "__main__":
    app.run(debug=True)