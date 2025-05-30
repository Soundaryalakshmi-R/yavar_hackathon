import os
import json
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from typing import List, Dict, Tuple


class BLIPCaptionGenerator:
    def __init__(self, model_path: str = "Salesforce/blip2-flan-t5-xl"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Blip2Processor.from_pretrained(model_path)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)

        try:
            self.font = ImageFont.truetype("arial.ttf", 20)
        except:
            self.font = ImageFont.load_default()

        self.colors = {"concise": "blue", "detailed": "red", "low_conf": "yellow"}

    def generate_captions(
        self, image: Image.Image, context_text: str
    ) -> Tuple[str, str, float, float]:
        context_text = context_text.strip()
        if len(context_text) > 700:
            context_text = context_text[:700] + "..."

        prompt = f"Context: {context_text}. First, a very short caption. Second, a detailed description:"

        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(
            self.device
        )

        with torch.no_grad():
            outputs = self.model.generate(**inputs)

        full_text = self.processor.tokenizer.decode(
            outputs[0], skip_special_tokens=True
        )
        captions = [cap.strip() for cap in full_text.split("Second,")]
        concise = captions[0].replace("First,", "").strip()
        detailed = captions[1] if len(captions) > 1 else concise

        # Confidence not directly available for this model, so use placeholders
        concise_conf = detailed_conf = 1.0

        return concise, detailed, concise_conf, detailed_conf

    def annotate_image(
        self,
        image: Image.Image,
        concise: str,
        detailed: str,
        concise_conf: float,
        detailed_conf: float,
    ) -> Image.Image:
        draw = ImageDraw.Draw(image)
        margin = 10
        y_pos = 10

        def draw_text_box(text, x, y, fill, confidence):
            text_color = "black"
            if confidence < 0.5:
                fill = self.colors["low_conf"]
                text_color = "red"
            bbox = draw.textbbox((x, y), text, font=self.font)
            padding = 5
            draw.rectangle(
                [
                    bbox[0] - padding,
                    bbox[1] - padding,
                    bbox[2] + padding,
                    bbox[3] + padding,
                ],
                fill=fill,
            )
            draw.text((x, y), text, fill=text_color, font=self.font)
            return bbox[3] + margin

        y_pos = draw_text_box(
            f"Concise: {concise}", margin, y_pos, self.colors["concise"], concise_conf
        )
        y_pos = draw_text_box(
            f"Detailed: {detailed}",
            margin,
            y_pos,
            self.colors["detailed"],
            detailed_conf,
        )
        draw_text_box(
            f"Confidence: Concise={concise_conf:.2f}, Detailed={detailed_conf:.2f}",
            margin,
            y_pos,
            "white",
            1.0,
        )
        return image

    def process_folder(
        self,
        dataset_json_path: str,
        img_folder: str,
        output_folder: str,
        max_samples: int = 2000,
    ):
        os.makedirs(output_folder, exist_ok=True)
        captions_data = []

        with open(dataset_json_path, "r", encoding="utf-8") as f:
            entries = json.load(f)

        for entry in entries[:max_samples]:
            img_path = os.path.join(img_folder, entry["image_id"])
            if not os.path.exists(img_path):
                print(f"Image not found: {entry['image_id']}")
                continue

            try:
                image = Image.open(img_path).convert("RGB")
                context_text = entry.get("context_text", "")

                concise, detailed, concise_conf, detailed_conf = self.generate_captions(
                    image, context_text
                )
                annotated_img = self.annotate_image(
                    image.copy(), concise, detailed, concise_conf, detailed_conf
                )
                annotated_img.save(
                    os.path.join(output_folder, f"annotated_{entry['image_id']}")
                )

                captions_data.append(
                    {
                        "image_id": entry["image_id"],
                        "concise_caption": {
                            "text": concise,
                            "confidence": concise_conf,
                            "consistent_with_metadata": self.check_consistency(
                                concise, {"context_text": context_text}
                            ),
                        },
                        "detailed_caption": {
                            "text": detailed,
                            "confidence": detailed_conf,
                            "consistent_with_metadata": self.check_consistency(
                                detailed, {"context_text": context_text}
                            ),
                        },
                        "warnings": self.generate_warnings(
                            concise, detailed, concise_conf, detailed_conf
                        ),
                    }
                )

            except Exception as e:
                print(f"Error processing {entry['image_id']}: {e}")

        with open(
            os.path.join(output_folder, "captions.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(captions_data, f, indent=2)

    def check_consistency(self, caption: str, metadata: dict) -> bool:
        context = metadata.get("context_text", "").lower()
        caption = caption.lower()
        contradictions = [
            ("increase", "decrease"),
            ("left", "right"),
            ("before", "after"),
        ]
        for term1, term2 in contradictions:
            if (term1 in context and term2 in caption) or (
                term2 in context and term1 in caption
            ):
                return False
        return True

    def generate_warnings(
        self, concise: str, detailed: str, concise_conf: float, detailed_conf: float
    ) -> List[str]:
        warnings = []
        if concise_conf < 0.5:
            warnings.append(f"Low confidence concise caption ({concise_conf:.2f})")
        if detailed_conf < 0.5:
            warnings.append(f"Low confidence detailed caption ({detailed_conf:.2f})")
        if concise.lower() not in detailed.lower():
            warnings.append("Concise and detailed captions may be inconsistent")
        return warnings


if __name__ == "__main__":
    generator = BLIPCaptionGenerator(model_path="Salesforce/blip2-flan-t5-xl")
    generator.process_folder(
        img_folder="img_folder",
        dataset_json_path="preprocessed/dataset.json",
        output_folder="output_folder",
        max_samples=1000,
    )
