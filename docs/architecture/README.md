# Architecture Diagrams

This directory will hold editable architecture diagrams for ClearMed Study. No binary or generated diagram files are included yet.

## Planned Diagrams

1. **Overall architecture**

   ```text
   MedMCQA → preprocessing → TF-IDF / Scratch Seq2Seq / FLAN-T5 + LoRA
     → QA evaluation → answer + explanation → student follow-up
     → MiniLM / E5 / BioLORD → top-k evidence → sufficient evidence?
       YES → FLAN-T5 + LoRA → grounded answer
       NO  → fixed fallback
   ```

2. **LoRA training workflow**

   ```text
   local development → small FLAN-T5 smoke test → school GPU cluster
     → LoRA fine-tuning → save adapter + predictions + metrics → local evaluation
   ```

3. **Retrieval pipeline**

   ```text
   context-rich follow-up query → embedding model → query vector
     → cosine similarity → rank MedMCQA training passages → top-k evidence
   ```

4. **Conversational retrieval and fallback flow**

   ```text
   retrieval → sufficient evidence?
     YES → FLAN-T5
     NO  → deterministic fallback
   ```
