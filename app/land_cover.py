from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch


class LandCoverAnalyzer:

    def __init__(self):
        print("Loading land-cover model...")

        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        self.labels = [
            "Urban / Built-up",
            "Vegetation",
            "Water",
            "Agriculture",
            "Roads"
        ]

        print("Land-cover model loaded!")

    def analyze(self, image):

        image = image.convert("RGB")

        prompts = [
            "a satellite image of urban built-up areas",
            "a satellite image of vegetation and forests",
            "a satellite image of water bodies",
            "a satellite image of agricultural land",
            "a satellite image showing roads"
        ]

        inputs = self.processor(
            text=prompts,
            images=image,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        probabilities = outputs.logits_per_image.softmax(dim=1)[0]

        results = []

        for label, probability in zip(
            self.labels,
            probabilities
        ):
            confidence = float(probability.item() * 100)

            results.append({
                "label": label,
                "confidence": confidence
            })

        results.sort(
            key=lambda x: x["confidence"],
            reverse=True
        )

        return results