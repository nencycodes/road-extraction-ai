import os
from src.inference.predict import predict


class ImagePipeline:

    def process(self, file):

        upload_dir = "data/uploads"
        os.makedirs(upload_dir, exist_ok=True)

        file_path = os.path.join(
            upload_dir,
            file.filename
        )

        with open(file_path, "wb") as buffer:
            buffer.write(file.file.read())

        mask, original_size = predict(file_path)

        return {
            "status": "prediction_successful",
            "filename": file.filename,
            "original_size": original_size,
            "mask_shape": list(mask.shape)
        }