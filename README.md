# Literature Gap Matrix RAG

Pipeline RAG untuk membuat Literature Gap Matrix dari kumpulan paper PDF.

## Cara pakai
1. Letakkan PDF paper di `data/papers/`
2. Set environment variable `GEMINI_API_KEY`
3. `pip install -r requirements.txt`
4. Jalankan:
   ```python
   from src.pipeline import run_pipeline
   matrix_df, gap_df = run_pipeline()
   ```
5. Hasil ada di folder `outputs/`

## Alur
PDF -> chunking -> embedding -> retrieval per aspek -> LLM -> Paper Matrix + Gap Analysis
