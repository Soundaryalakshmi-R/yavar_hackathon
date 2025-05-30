<<<<<<< HEAD
import os
import json
from pathlib import Path
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from sentence_transformers import SentenceTransformer
from PIL import Image
from tqdm import tqdm


class CaptionEvaluator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.blip_processor = Blip2Processor.from_pretrained(
            "Salesforce/blip2-flan-t5-xl"
        )
        self.blip_model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-flan-t5-xl",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.text_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.rouge = Rouge()

    def evaluate_caption(
        self, image_path: str, generated_caption: str, reference_text: str
    ) -> dict:
        image = Image.open(image_path).convert("RGB")
        inputs = self.blip_processor(images=image, return_tensors="pt").to(self.device)

        smooth = SmoothingFunction().method4
        bleu = sentence_bleu(
            [reference_text.split()],
            generated_caption.split(),
            smoothing_function=smooth,
        )

        rouge_scores = self.rouge.get_scores(generated_caption, reference_text)
        rouge_l = rouge_scores[0]["rouge-l"]["f"]

        with torch.no_grad():
            outputs = self.blip_model.generate(
                **inputs,
                num_return_sequences=3,
                num_beams=3,
                max_length=50,
                early_stopping=True,
            )
            candidate_captions = [
                self.blip_processor.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in outputs
            ]
            gen_embedding = self.text_model.encode(generated_caption)
            candidate_embeddings = self.text_model.encode(candidate_captions)
            consistency_scores = [
                torch.cosine_similarity(
                    torch.tensor(gen_embedding), torch.tensor(cand_emb), dim=0
                ).item()
                for cand_emb in candidate_embeddings
            ]
            blip_score = sum(consistency_scores) / len(consistency_scores)

        return {
            "bleu": round(bleu, 4),
            "rouge_l": round(rouge_l, 4),
            "blip_consistency": round(blip_score, 4),
            "candidate_captions": candidate_captions,
        }


def main():
    evaluator = CaptionEvaluator()

    with open("output_folder/captions.json") as f:
        captions_data = json.load(f)

    with open("preprocessed/dataset.json") as f:
        context_lookup = {
            item["image_id"]: item["context_text"] for item in json.load(f)
        }

    results = []
    count = 0
    for item in tqdm(captions_data, desc="Evaluating captions"):
        if count >= 100:
            break

        image_path = os.path.join("img_folder", item["image_id"])
        context_text = context_lookup.get(item["image_id"], "")

        try:
            concise_scores = evaluator.evaluate_caption(
                image_path, item["concise_caption"]["text"], context_text
            )

            detailed_scores = evaluator.evaluate_caption(
                image_path, item["detailed_caption"]["text"], context_text
            )

            results.append(
                {
                    "image_id": item["image_id"],
                    "concise_caption": {
                        "text": item["concise_caption"]["text"],
                        "confidence": item["concise_caption"]["confidence"],
                        **concise_scores,
                    },
                    "detailed_caption": {
                        "text": item["detailed_caption"]["text"],
                        "confidence": item["detailed_caption"]["confidence"],
                        **detailed_scores,
                    },
                    "metadata_consistency": item.get("consistent_with_metadata", True),
                }
            )
            count += 1

        except Exception as e:
            print(f"Error evaluating {item['image_id']}: {e}")

    with open("output_folder/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    avg_metrics = {
        "concise": {
            "bleu": round(
                sum(r["concise_caption"]["bleu"] for r in results) / len(results), 3
            ),
            "rouge_l": round(
                sum(r["concise_caption"]["rouge_l"] for r in results) / len(results), 3
            ),
            "blip_consistency": round(
                sum(r["concise_caption"]["blip_consistency"] for r in results)
                / len(results),
                3,
            ),
        },
        "detailed": {
            "bleu": round(
                sum(r["detailed_caption"]["bleu"] for r in results) / len(results), 3
            ),
            "rouge_l": round(
                sum(r["detailed_caption"]["rouge_l"] for r in results) / len(results), 3
            ),
            "blip_consistency": round(
                sum(r["detailed_caption"]["blip_consistency"] for r in results)
                / len(results),
                3,
            ),
        },
    }

    print("\nEvaluation Summary:")
    print(f"Average Concise Caption Scores: {avg_metrics['concise']}")
    print(f"Average Detailed Caption Scores: {avg_metrics['detailed']}")


if __name__ == "__main__":
    main()
=======
import os
import json
from pathlib import Path
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from sentence_transformers import SentenceTransformer
from PIL import Image
from tqdm import tqdm


class CaptionEvaluator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.blip_processor = Blip2Processor.from_pretrained(
            "Salesforce/blip2-flan-t5-xl"
        )
        self.blip_model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-flan-t5-xl",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.text_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.rouge = Rouge()

    def evaluate_caption(
        self, image_path: str, generated_caption: str, reference_text: str
    ) -> dict:
        image = Image.open(image_path).convert("RGB")
        inputs = self.blip_processor(images=image, return_tensors="pt").to(self.device)

        smooth = SmoothingFunction().method4
        bleu = sentence_bleu(
            [reference_text.split()],
            generated_caption.split(),
            smoothing_function=smooth,
        )

        rouge_scores = self.rouge.get_scores(generated_caption, reference_text)
        rouge_l = rouge_scores[0]["rouge-l"]["f"]

        with torch.no_grad():
            outputs = self.blip_model.generate(
                **inputs,
                num_return_sequences=3,
                num_beams=3,
                max_length=50,
                early_stopping=True,
            )
            candidate_captions = [
                self.blip_processor.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in outputs
            ]
            gen_embedding = self.text_model.encode(generated_caption)
            candidate_embeddings = self.text_model.encode(candidate_captions)
            consistency_scores = [
                torch.cosine_similarity(
                    torch.tensor(gen_embedding), torch.tensor(cand_emb), dim=0
                ).item()
                for cand_emb in candidate_embeddings
            ]
            blip_score = sum(consistency_scores) / len(consistency_scores)

        return {
            "bleu": round(bleu, 4),
            "rouge_l": round(rouge_l, 4),
            "blip_consistency": round(blip_score, 4),
            "candidate_captions": candidate_captions,
        }


def main():
    evaluator = CaptionEvaluator()

    with open("output_folder/captions.json") as f:
        captions_data = json.load(f)

    with open("preprocessed/dataset.json") as f:
        context_lookup = {
            item["image_id"]: item["context_text"] for item in json.load(f)
        }

    results = []
    count = 0
    for item in tqdm(captions_data, desc="Evaluating captions"):
        if count >= 100:
            break

        image_path = os.path.join("img_folder", item["image_id"])
        context_text = context_lookup.get(item["image_id"], "")

        try:
            concise_scores = evaluator.evaluate_caption(
                image_path, item["concise_caption"]["text"], context_text
            )

            detailed_scores = evaluator.evaluate_caption(
                image_path, item["detailed_caption"]["text"], context_text
            )

            results.append(
                {
                    "image_id": item["image_id"],
                    "concise_caption": {
                        "text": item["concise_caption"]["text"],
                        "confidence": item["concise_caption"]["confidence"],
                        **concise_scores,
                    },
                    "detailed_caption": {
                        "text": item["detailed_caption"]["text"],
                        "confidence": item["detailed_caption"]["confidence"],
                        **detailed_scores,
                    },
                    "metadata_consistency": item.get("consistent_with_metadata", True),
                }
            )
            count += 1

        except Exception as e:
            print(f"Error evaluating {item['image_id']}: {e}")

    with open("output_folder/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    avg_metrics = {
        "concise": {
            "bleu": round(
                sum(r["concise_caption"]["bleu"] for r in results) / len(results), 3
            ),
            "rouge_l": round(
                sum(r["concise_caption"]["rouge_l"] for r in results) / len(results), 3
            ),
            "blip_consistency": round(
                sum(r["concise_caption"]["blip_consistency"] for r in results)
                / len(results),
                3,
            ),
        },
        "detailed": {
            "bleu": round(
                sum(r["detailed_caption"]["bleu"] for r in results) / len(results), 3
            ),
            "rouge_l": round(
                sum(r["detailed_caption"]["rouge_l"] for r in results) / len(results), 3
            ),
            "blip_consistency": round(
                sum(r["detailed_caption"]["blip_consistency"] for r in results)
                / len(results),
                3,
            ),
        },
    }

    print("\nEvaluation Summary:")
    print(f"Average Concise Caption Scores: {avg_metrics['concise']}")
    print(f"Average Detailed Caption Scores: {avg_metrics['detailed']}")


if __name__ == "__main__":
    main()
>>>>>>> master
