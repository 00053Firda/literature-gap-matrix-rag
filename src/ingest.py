from pathlib import Path
from pypdf import PdfReader


def load_pdf_text(pdf_path):
    """Membaca seluruh teks dari satu file PDF."""
    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    full_text = "\n".join(pages)
    return " ".join(full_text.split()) 


def chunk_text(text, chunk_size=1000, overlap=200):
    """Memotong teks panjang menjadi potongan (chunk) yang saling tumpang tindih."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def load_all_papers(folder="/content/drive/MyDrive/Jurnal_RAG", chunk_size=1000, overlap=200):
    """Membaca semua PDF di folder Google Drive. Hasil: {nama_paper: [chunk1, chunk2, ...]}"""
    papers = {}
    path_folder = Path(folder)
    
    if not path_folder.exists():
        print(f"[PERINGATAN] Jalur folder tidak ditemukan: {folder}")
        print("Pastikan Google Drive sudah di-mount dengan benar.")
        return papers
        
    pdf_files = sorted(path_folder.glob("*.pdf"))
    if not pdf_files:
        print(f"[INFO] Tidak ada berkas berformat .pdf yang ditemukan di: {folder}")
        return papers

    for pdf_file in pdf_files:
        print(f"Membaca: {pdf_file.name}")
        try:
            text = load_pdf_text(pdf_file)
            papers[pdf_file.stem] = chunk_text(text, chunk_size, overlap)
        except Exception as e:
            print(f"Gagal membaca berkas {pdf_file.name}: {e}")
            
    return papers
