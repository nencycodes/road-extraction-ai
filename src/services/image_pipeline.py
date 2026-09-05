from src.inference.predict import predict


class ImagePipeline:

    def process(self, file):
        # Temporary: prediction will be connected to uploaded file next
        return {
            "status": "pipeline_ready"
        }