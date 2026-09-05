const imageInput = document.getElementById("imageInput");
const uploadCard = document.getElementById("uploadCard");

const selectedFile = document.getElementById("selectedFile");
const fileName = document.getElementById("fileName");
const fileSize = document.getElementById("fileSize");

const removeFile = document.getElementById("removeFile");
const predictButton = document.getElementById("predictButton");

const processingStatus = document.getElementById("processingStatus");
const resultsSection = document.getElementById("resultsSection");

const originalImage = document.getElementById("originalImage");
const predictionImage = document.getElementById("predictionImage");

let selectedImage = null;


// ================= FILE SELECTION =================

imageInput.addEventListener("change", function () {

    const file = this.files[0];

    if (!file) {
        return;
    }

    selectedImage = file;

    fileName.textContent = file.name;

    const sizeMB = (file.size / (1024 * 1024)).toFixed(2);

    fileSize.textContent = `${sizeMB} MB`;

    selectedFile.style.display = "flex";

    predictButton.disabled = false;

    resultsSection.style.display = "none";

});


// ================= REMOVE FILE =================

removeFile.addEventListener("click", function () {

    selectedImage = null;

    imageInput.value = "";

    selectedFile.style.display = "none";

    predictButton.disabled = true;

    resultsSection.style.display = "none";

});


// ================= PREDICTION =================

predictButton.addEventListener("click", async function () {

    if (!selectedImage) {
        return;
    }

    predictButton.disabled = true;

    processingStatus.style.display = "flex";

    resultsSection.style.display = "none";


    const formData = new FormData();

    formData.append("file", selectedImage);


    try {

        const response = await fetch(
            "http://127.0.0.1:8000/upload",
            {
                method: "POST",
                body: formData
            }
        );


        if (!response.ok) {
            throw new Error("Prediction request failed.");
        }


        const result = await response.json();


        // Display original image

        originalImage.src =
            URL.createObjectURL(selectedImage);

        originalImage.style.display = "block";


        // Display AI prediction

        predictionImage.src =
            `http://127.0.0.1:8000${result.prediction}?t=${Date.now()}`;

        predictionImage.style.display = "block";


        // Show results

        resultsSection.style.display = "block";


        // Scroll to results

        resultsSection.scrollIntoView({
            behavior: "smooth"
        });


    } catch (error) {

        console.error(error);

        alert(
            "Unable to connect to RoadVision AI backend."
        );

    } finally {

        processingStatus.style.display = "none";

        predictButton.disabled = false;

    }

});