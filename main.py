from groq import Groq
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from collections import Counter

import io
import requests
import re
import os
import json

ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = FastAPI()

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/match")
async def match(file: UploadFile = File(...)):
    contents = await file.read()
    pdf = PdfReader(io.BytesIO(contents))

    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""

    text = re.sub(r'\s+', ' ', text).strip()

    print("📄 Resume extracted (first 300 chars):", text[:300])

    resume_data = extract_query_from_resume(text)
    print("🔎 Resume data:", resume_data)
    level = resume_data.get("level", "entry")
    print(f"🎓 Detected level: {level}")

    jobs = get_jobs(resume_data)
    print("📊 Jobs fetched:", len(jobs))

    if not jobs:
        return {"matches": [], "query_used": resume_data}

    results = score_jobs(text, jobs)
    print(f"📊 Results sample: {results[:2]}")
    return {"matches": results, "query_used": resume_data}

def extract_query_from_resume(resume_text: str) -> dict:
    if not groq_client:
        print("⚠️ No Groq key, using fallback")
        return {"titles": ["software developer"], "location": "", "keywords": ""}

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": f"""Analyze this resume carefully and extract the following. Return ONLY a JSON object, nothing else.

{{
  "titles": ["7-10 job titles suited for this person's CURRENT skill level and experience — if they are a student or entry level, only suggest entry level, junior, or internship roles. Base this on their actual experience, what they are looking for, volunteering, and skills, not just their degree"],
  "location": "city name only, where they are based",
  "keywords": "5-6 most important skills from the resume as a space separated string"
  "level": "entry, mid, or senior — based on years of experience and role history in the resume"
}}

Resume:
{resume_text[:3000]}"""}],
            max_tokens=200
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        parsed = json.loads(raw)
        print(f"✅ Groq parsed: {parsed}")
        return parsed

    except Exception as e:
        print("❌ Groq extraction failed:", e)
        return {"titles": ["software developer"], "location": "", "keywords": ""}


def get_jobs(resume_data: dict) -> list:
    if not ADZUNA_APP_ID or not ADZUNA_APP_KEY:
        print("❌ Missing Adzuna API keys")
        return []

    titles = resume_data.get("titles", ["software developer"])
    location = resume_data.get("location", "")
    keywords = resume_data.get("keywords", "")

    if isinstance(titles, str):
        titles = [titles]

    all_jobs = []
    seen = set()

    for title in titles[:10]:
        title = title.strip()
        if not title:
            continue

        query = title  # just the title, no keywords — keeps query clean

        params = {
            "app_id": ADZUNA_APP_ID,
            "app_key": ADZUNA_APP_KEY,
            "what": query,
            "results_per_page": 2
        }

        try:
            # first try with location
            if location:
                params["where"] = location
                res = requests.get("https://api.adzuna.com/v1/api/jobs/ca/search/1", params=params)
                items = res.json().get("results", [])
                print(f"🌐 [{title} | {location}] → {len(items)} results")

                # if nothing, retry without location
                if not items:
                    params.pop("where")
                    res = requests.get("https://api.adzuna.com/v1/api/jobs/ca/search/1", params=params)
                    items = res.json().get("results", [])
                    print(f"🔄 [{title} | no location] → {len(items)} results")
            else:
                res = requests.get("https://api.adzuna.com/v1/api/jobs/ca/search/1", params=params)
                items = res.json().get("results", [])
                print(f"🌐 [{title}] → {len(items)} results")

            for item in items:
                t = item.get("title")
                company = item.get("company", {}).get("display_name")
                key = f"{t}-{company}"
                if key not in seen:
                    seen.add(key)
                    all_jobs.append({
                        "title": t,
                        "company": company,
                        "description": item.get("description", ""),
                        "url": item.get("redirect_url", "")
                    })

        except Exception as e:
            print(f"❌ Adzuna fetch error for '{title}':", e)

    print(f"📊 Total jobs fetched: {len(all_jobs)}")
    return all_jobs

    
SENIOR_KEYWORDS = {"senior", "sr.", "lead", "principal", "director", "head", "vp", "vice president", "chief", "architect", "staff"}
ENTRY_KEYWORDS = {"junior", "jr.", "entry", "intern", "internship", "graduate", "trainee", "assistant"}

def score_jobs(resume_text: str, jobs: list, level: str = "entry") -> list:
    resume_words = set(resume_text.lower().split())

    results = []
    for job in jobs:
        title_lower = job["title"].lower() if job["title"] else ""

        # filter based on detected level
        if level == "entry":
            if any(kw in title_lower for kw in SENIOR_KEYWORDS):
                print(f"⛔ Skipping senior role for entry level: {job['title']}")
                continue
        elif level == "mid":
            if any(kw in title_lower for kw in SENIOR_KEYWORDS):
                print(f"⛔ Skipping senior role for mid level: {job['title']}")
                continue
        # senior level — show everything

        job_words = set((job["title"] + " " + job["description"]).lower().split())
        score = len(resume_words.intersection(job_words))
        results.append({
            "title": job["title"],
            "company": job["company"],
            "score": score,
            "url": job.get("url", "")
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)