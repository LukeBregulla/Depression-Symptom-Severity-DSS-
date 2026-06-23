from __future__ import annotations
import csv
import re
from pathlib import Path
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Paths and Configuration
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT = str(PROJECT_ROOT / "data" / "input_de.csv")
OUTPUT = str(PROJECT_ROOT / "data" / "output_en.csv")
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-de-en"


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Helper Functions
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def clean_text(value: str) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Translation
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class FullTextTranslator:
    def __init__(self, model_name: str = TRANSLATION_MODEL):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required, but no GPU was detected.")
        
        self.device = torch.device("cuda")
        self.device_name = torch.cuda.get_device_name(0)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.tokenizer.model_max_length = int(1e9)

        model_max = getattr(self.model.config, "max_position_embeddings", 512)
        self.max_source_tokens = max(64, model_max - 8)

    def _token_len(self, text: str) -> int:
        tok = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return len(tok.get("input_ids", []))

    def _chunk_text(self, text: str) -> list[str]:
        text = clean_text(text)
        if not text:
            return []
        if self._token_len(text) <= self.max_source_tokens:
            return [text]

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not sentences:
            sentences = [text]

        chunks: list[str] = []
        current: list[str] = []

        for sentence in sentences:
            trial = " ".join(current + [sentence])
            if current and self._token_len(trial) > self.max_source_tokens:
                chunks.append(" ".join(current))
                current = [sentence]
            else:
                current.append(sentence)

            while current and self._token_len(" ".join(current)) > self.max_source_tokens:
                words = current[0].split()
                if len(words) <= 1:
                    chunks.append(current[0])
                    current = current[1:]
                    continue

                left_words: list[str] = []
                right_words: list[str] = []
                for w in words:
                    trial_left = " ".join(left_words + [w])
                    if self._token_len(trial_left) <= self.max_source_tokens:
                        left_words.append(w)
                    else:
                        right_words.append(w)

                if not left_words:
                    left_words = [words[0]]
                    right_words = words[1:]

                chunks.append(" ".join(left_words))
                current[0] = " ".join(right_words) if right_words else ""
                current = [part for part in current if part]

        if current:
            chunks.append(" ".join(current))
        
        return chunks

    def _translate_chunk_batch(self, chunks: list[str], batch_size: int = 24) -> list[str]:
        if not chunks:
            return []

        outputs: list[str] = []
        total_batches = (len(chunks) + batch_size - 1) // batch_size

        for batch_idx, start in enumerate(range(0, len(chunks), batch_size), start=1):
            batch = chunks[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_source_tokens,
            ).to(self.device)

            src_len = int(encoded["input_ids"].shape[1])
            max_new_tokens = min(512, max(32, int(src_len * 2.2)))

            with torch.no_grad():
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.1,
                )

            decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            outputs.extend(clean_text(d) for d in decoded)
            print(f"    chunk batch {batch_idx}/{total_batches}")

        return outputs

    def translate(self, text: str, batch_size: int = 24) -> str:
        text = clean_text(text)
        if not text:
            return ""
        chunks = self._chunk_text(text)
        translated_chunks = self._translate_chunk_batch(chunks, batch_size=batch_size)
        return " ".join(translated_chunks)
    
    def count_source_tokens_and_chunks(self, text: str) -> tuple[int, int]:
        """Return (token_count, chunk_count) for text."""
        text = clean_text(text)
        if not text:
            return 0, 0
        return self._token_len(text), len(self._chunk_text(text))



def run_translation(input_csv: Path, output_csv: Path, batch_size: int = 24) -> None:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    translator = FullTextTranslator()
    print(f"Model loaded on device: {translator.device} ({translator.device_name})")
    
    rows_out = []
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"session_id", "text", "hamd_sum"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError("Input CSV must contain columns: session_id,text,hamd_sum")
        rows = list(reader)
    
    total = len(rows)
    print(f"Translating {total} rows from {input_csv} ...")

    for i, row in enumerate(rows, start=1):
        src_text = clean_text(row.get("text", ""))
        session_id = clean_text(row.get("session_id", ""))
        hamd_sum = clean_text(row.get("hamd_sum", ""))
        
        token_count, chunk_count = translator.count_source_tokens_and_chunks(src_text)
        print(f"  row {i}/{total} {session_id}: tokens={token_count}, chunks={chunk_count}")
        
        translated = translator.translate(src_text, batch_size=batch_size)
        rows_out.append({
            "session_id": session_id,
            "text": translated,
            "hamd_sum": hamd_sum,
        })
        print(f"  translated row {i}/{total}: {session_id}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["session_id", "text", "hamd_sum"])
        writer.writeheader()
        writer.writerows(rows_out)
    
    print(f"Wrote translated CSV: {output_csv}")

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Main Execution
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    run_translation(Path(INPUT), Path(OUTPUT), batch_size=24)
