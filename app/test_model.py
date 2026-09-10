from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image


class SatelliteAI:

    def __init__(self):

        print("Loading AI model...")

        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-vqa-base"
        )

        self.model = BlipForQuestionAnswering.from_pretrained(
            "Salesforce/blip-vqa-base"
        )

        print("AI model loaded successfully!")

    def answer_question(self, image, question):

        image = image.convert("RGB")

        inputs = self.processor(
            images=image,
            text=question,
            return_tensors="pt"
        )

        output = self.model.generate(
            **inputs,
            max_new_tokens=30
        )

        answer = self.processor.decode(
            output[0],
            skip_special_tokens=True
        )

        return answer