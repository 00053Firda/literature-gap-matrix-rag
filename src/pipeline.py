import pandas as pd
from src.ingest import load_all_papers
from src.retriever import PaperRetriever
from src.generator import extract_paper_info, generate_gap_analysis


def run_pipeline(papers_folder="data/papers", output_dir="outputs"):
    # 1. Baca PDF dan potong menjadi chunk
    papers = load_all_papers(papers_folder)
    print(f"\nTotal paper: {len(papers)}")

    # 2. Buat embedding dan index
    retriever = PaperRetriever()
    retriever.index(papers)
    print("Indexing selesai")

    # 3. Isi matrix per paper (RAG)
    rows = []
    for paper_id in papers:
        print(f"Mengekstrak: {paper_id}")
        info = extract_paper_info(paper_id, retriever)
        info["paper_id"] = paper_id
        rows.append(info)
    matrix_df = pd.DataFrame(rows)
    matrix_df = matrix_df[["paper_id"] + [c for c in matrix_df.columns if c != "paper_id"]]

    # 4. Analisis gap lintas paper
    print("Menganalisis research gap...")
    gap_df = pd.DataFrame(generate_gap_analysis(matrix_df))

    # 5. Simpan hasil
    matrix_df.to_csv(f"{output_dir}/paper_matrix.csv", index=False)
    with pd.ExcelWriter(f"{output_dir}/literature_gap_matrix.xlsx") as writer:
        matrix_df.to_excel(writer, sheet_name="Paper Matrix", index=False)
        gap_df.to_excel(writer, sheet_name="Gap Analysis", index=False)
    print(f"Selesai. Hasil tersimpan di folder '{output_dir}'")
    return matrix_df, gap_df
