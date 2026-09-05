import os
from PIL import Image

from src.inference.predict import predict


class ImagePipeline:

    def process(self, file):

        upload_dir = "data/uploads"
        output_dir = "data/processed"

        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        file_path = os.path.join(
            upload_dir,
            file.filename
        )

        with open(file_path, "wb") as buffer:
            buffer.write(file.file.read())

        mask, original_size = predict(file_path)

        mask_image = Image.fromarray(
            (mask.numpy() * 255).astype("uint8")
        )

        output_path = os.path.join(
            output_dir,
            "prediction.png"
        )

        mask_image.save(output_path)

        return {
            "status": "prediction_successful",
            "filename": file.filename,
            "original_size": original_size,
            "prediction": "/processed/prediction.png"
        }