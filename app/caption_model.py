from transformers import BlipProcessor, BlipForConditionalGeneration


class SatelliteCaptioner:

    def __init__(self):
        print("Loading captioning model...")

        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )

        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )

        print("Captioning model loaded successfully!")

    def generate_caption(self, image):

        image = image.convert("RGB")

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        output = self.model.generate(
            **inputs,
            max_new_tokens=40
        )

        caption = self.processor.decode(
            output[0],
            skip_special_tokens=True
        )

        return caption