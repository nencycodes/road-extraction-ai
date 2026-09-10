import os

from PIL import Image

from src.inference.predict import predict


class ImagePipeline:

    def process(self, file):

        upload_dir = "data/uploads"
        output_dir = "data/processed"

        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Save uploaded file
        file_path = os.path.join(
            upload_dir,
            file.filename
        )

        with open(file_path, "wb") as buffer:
            buffer.write(file.file.read())

        # Run trained U-Net
        mask, original_size = predict(file_path)

        # Save prediction mask
        mask_image = Image.fromarray(
            (mask.numpy() * 255).astype("uint8")
        )

        prediction_path = os.path.join(
            output_dir,
            "prediction.png"
        )

        mask_image.save(prediction_path)

        # Create browser-friendly preview
        image = Image.open(file_path).convert("RGB")

        preview_path = os.path.join(
            output_dir,
            "input_preview.jpg"
        )

        image.thumbnail((1200, 1200))
        image.save(preview_path, "JPEG")

        return {
            "status": "prediction_successful",
            "filename": file.filename,
            "original_size": original_size,
            "prediction": "/processed/prediction.png",
            "preview": "/processed/input_preview.jpg"
        }