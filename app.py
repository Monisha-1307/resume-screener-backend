import os
import urllib
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
from docx import Document
import pytesseract

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# -------------------------------
# Environment variables
# -------------------------------
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'fallbacksecret')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///default.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
jwt = JWTManager(app)

# -------------------------------
# Database models
# -------------------------------
class Resume(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)

class Job(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)

class Comparison(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    resume_id = db.Column(db.Integer, db.ForeignKey('resume.id'), nullable=False)
    job_id = db.Column(db.Integer, db.ForeignKey('job.id'), nullable=False)
    score = db.Column(db.Float, nullable=False)
    keywords = db.Column(db.Text, nullable=True)

# -------------------------------
# Health check
# -------------------------------
@app.route('/ping')
def ping():
    return jsonify({"status": "ok"})

# -------------------------------
# Root route
# -------------------------------
@app.route('/')
def home():
    return jsonify({"message": "Resume Screener Backend is running!"})

# -------------------------------
# Utility function
# -------------------------------
def calculate_similarity_with_keywords(resume_text, job_text):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_text])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    resume_words = set(resume_text.lower().split())
    job_words = set(job_text.lower().split())
    common_words = list(resume_words.intersection(job_words))

    return round(similarity * 100, 2), common_words

# -------------------------------
# Upload resume
# -------------------------------
@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    if 'resume' not in request.files:
        return jsonify({"error": "No resume file uploaded"}), 400

    file = request.files['resume']
    filename = file.filename.lower()
    text = ""

    try:
        if filename.endswith(".pdf"):
            file.seek(0)
            pdf_bytes = BytesIO(file.read())
            with pdfplumber.open(pdf_bytes) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    else:
                        pil_image = page.to_image(resolution=300).original
                        ocr_text = pytesseract.image_to_string(pil_image)
                        if ocr_text.strip():
                            text += ocr_text + "\n"

        elif filename.endswith(".docx"):
            file.seek(0)
            doc_bytes = BytesIO(file.read())
            doc = Document(doc_bytes)
            for para in doc.paragraphs:
                if para.text.strip():
                    text += para.text + "\n"
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        if cell.text.strip():
                            text += cell.text + "\n"

        else:
            file.seek(0)
            text = file.read().decode('utf-8', errors='ignore')

        if not text.strip():
            return jsonify({"error": "No text extracted from resume"}), 500

        # ✅ Save resume to DB
        new_resume = Resume(filename=filename, content=text)
        db.session.add(new_resume)
        db.session.commit()

        return jsonify({"resume_text": text, "resume_id": new_resume.id})

    except Exception as e:
        # ✅ Print error so it shows in Render logs
        print("Extraction failed:", str(e))
        return jsonify({"error": f"Failed to extract text: {str(e)}"}), 500

# -------------------------------
# Store job descriptions
# -------------------------------
@app.route('/add_job', methods=['POST'])
def add_job():
    data = request.json
    title = data.get("title", "")
    description = data.get("description", "")
    if not title or not description:
        return jsonify({"error": "Job title or description missing"}), 400

    new_job = Job(title=title, description=description)
    db.session.add(new_job)
    db.session.commit()

    return jsonify({"job_id": new_job.id, "title": title})

# -------------------------------
# Match routes with persistence
# -------------------------------
@app.route('/match', methods=['POST'])
def match_resume():
    data = request.json
    resume_text = data.get("resume", "")
    job_text = data.get("job", "")
    resume_id = data.get("resume_id")
    job_id = data.get("job_id")

    if not resume_text or not job_text:
        return jsonify({"error": "Resume or job description missing"}), 400

    score, keywords = calculate_similarity_with_keywords(resume_text, job_text)

    if resume_id and job_id:
        comparison = Comparison(resume_id=resume_id, job_id=job_id,
                                score=score, keywords=",".join(keywords))
        db.session.add(comparison)
        db.session.commit()

    return jsonify({"score": score, "keywords": keywords})

@app.route('/match_multiple', methods=['POST'])
def match_multiple():
    data = request.json
    resume_text = data.get("resume", "")
    jobs = data.get("jobs", [])
    resume_id = data.get("resume_id")

    if not resume_text or not jobs:
        return jsonify({"error": "Resume or job descriptions missing"}), 400

    results = []
    for job in jobs:
        score, keywords = calculate_similarity_with_keywords(resume_text, job["description"])
        results.append({
            "title": job["title"],
            "score": score,
            "keywords": keywords
        })

        new_job = Job(title=job["title"], description=job["description"])
        db.session.add(new_job)
        db.session.commit()

        comparison = Comparison(resume_id=resume_id, job_id=new_job.id,
                                score=score, keywords=",".join(keywords))
        db.session.add(comparison)
        db.session.commit()

    return jsonify({"results": results})

# -------------------------------
# Resume summary
# -------------------------------
@app.route('/resume_summary', methods=['POST'])
def resume_summary():
    data = request.json
    resume = data.get("resume", "")
    if not resume:
        return jsonify({"error": "Resume text missing"}), 400

    skills_list = [
        "python", "sql", "java", "c++", "tableau", "power bi", "excel",
        "kubernetes", "docker", "aws", "azure", "gcp",
        "prometheus", "grafana", "victoriametrics",
        "machine learning", "data science", "analytics",
        "digital marketing", "seo", "sem"
    ]

    resume_lower = resume.lower()
    matched_skills = [skill for skill in skills_list if skill in resume_lower]

    if matched_skills:
        summary = f"This resume highlights skills in: {', '.join(matched_skills)}"
    else:
        summary = "No specific technical skills detected in the resume."

    return jsonify({"summary": summary})

# -------------------------------
# List stored data
# -------------------------------
@app.route('/resumes', methods=['GET'])
def get_resumes():
    resumes = Resume.query.all()
    return jsonify([{"id": r.id, "filename": r.filename, "content": r.content[:200]} for r in resumes])

@app.route('/jobs', methods=['GET'])
def get_jobs():
    jobs = Job.query.all()
    return jsonify([{"id": j.id, "title": j.title, "description": j.description[:200]} for j in jobs])

@app.route('/comparisons', methods=['GET'])
def get_comparisons():
    comps = Comparison.query.all()
    return jsonify([
        {"id": c.id, "resume_id": c.resume_id, "job_id": c.job_id,
         "score": c.score, "keywords": c.keywords}
        for c in comps
    ])

# -------------------------------
# Debug route: list all routes
# -------------------------------
@app.route('/routes')
def list_routes():
    output = []
    for rule in app.url_map.iter_rules():
        methods = ','.join(rule.methods)
        line = urllib.parse.unquote(f"{rule.endpoint}: {rule.rule} [{methods}]")
        output.append(line)
    return jsonify({"routes": output})

# -------------------------------
# Run Flask app (local only)
# -------------------------------
if __name__ == '__main__':
    with app.app_context():
        db.create_all()     # ✅ ensures tables are created
    app.run(debug=True)