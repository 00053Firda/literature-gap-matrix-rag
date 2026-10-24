import os
import re
import json
import time
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL_NAME = "gemini-3.8-flash"
OUTPUT_LANGUAGE = "Bahasa Indonesia"

ASPECTS = {
    "research_problem": "research problem objective aim of this study",
    "method": "proposed method approach model architecture methodology",
    "dataset": "dataset data used experiments participants sample",
    "findings": "results findings performance accuracy conclusion",
    "limitations": "limitations weaknesses constraints of this study",
    "future_work": "future work future research directions recommendations",
}


def call_llm(prompt, max_retries=3):
    """Memanggil LLM, dengan percobaan ulang jika error (misal kena rate limit)."""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
            return response.text
        except Exception as e:
            print(f"Percobaan {attempt + 1} gagal: {e}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("LLM gagal merespons setelah beberapa percobaan")


def parse_json(text):
    """Membersihkan pagar markdown (```json) lalu mengubah teks menjadi objek Python."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    return json.loads(text)


def extract_paper_info(paper_id, retriever, top_k=3):
    """Tahap RETRIEVAL + GENERATION untuk satu paper."""
    context_parts = ["### Awal paper (judul, abstrak):\n" + retriever.chunks[paper_id][0]]
    for aspect, query in ASPECTS.items():
        chunks = retriever.search(paper_id, query, top_k=top_k)
        context_parts.append(f"### Konteks untuk '{aspect}':\n" + "\n---\n".join(chunks))
    context = "\n\n".join(context_parts)

    fields = ["title"] + list(ASPECTS.keys())
    template = json.dumps({f: "..." for f in fields}, indent=2)
    prompt = (
        "Kamu adalah asisten peneliti yang teliti. Berdasarkan HANYA konteks paper di bawah, "
        "isi informasi berikut dalam format JSON. Jika informasi tidak ada di konteks, "
        "tulis 'Tidak disebutkan'. Jangan mengarang. "
        f"Tulis isi dalam {OUTPUT_LANGUAGE}, singkat (1-3 kalimat per kolom). "
        "Balas HANYA dengan JSON, tanpa teks lain.\n\n"
        f"Format JSON:\n{template}\n\n"
        f"KONTEKS PAPER:\n{context}"
    )
    return parse_json(call_llm(prompt))


def generate_gap_analysis(matrix_df):
    """Menganalisis seluruh matrix untuk menemukan celah penelitian."""
    matrix_json = matrix_df.to_json(orient="records", force_ascii=False)
    template = json.dumps(
        [{
            "gap_type": "metodologi / dataset / konteks / teori / evaluasi",
            "gap_description": "...",
            "evidence": "paper_id yang mendukung",
            "research_opportunity": "...",
        }],
        indent=2, ensure_ascii=False,
    )
    prompt = (
        "Kamu adalah peneliti senior. Di bawah ini adalah ringkasan beberapa paper (format JSON). "
        "Identifikasi 5 research gap yang paling jelas berdasarkan keterbatasan, saran penelitian "
        "lanjutan, serta pola metode dan dataset yang sering atau jarang dipakai."
        "Hanya gunakan informasi dari data ini dan sebutkan paper_id sebagai bukti. "
        f"Tulis dalam {OUTPUT_LANGUAGE}. Balas HANYA dengan JSON array berformat:\n{template}\n\n"
        f"DATA PAPER:\n{matrix_json}"
    )
    return parse_json(call_llm(prompt))
