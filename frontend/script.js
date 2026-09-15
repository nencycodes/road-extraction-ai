// ============================================================
// RoadVision AI — Frontend Controller
// ============================================================

const API_BASE_URL = "http://127.0.0.1:8000";

// ------------------------------------------------------------
// DOM ELEMENTS
// ------------------------------------------------------------

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

// ------------------------------------------------------------
// PIPELINE RESULT ELEMENTS
// ------------------------------------------------------------

const graphSection = document.getElementById("graphSection");
const criticalitySection = document.getElementById("criticalitySection");
const resilienceSection = document.getElementById("resilienceSection");

// STEP 03
const graphNodes = document.getElementById("graphNodes");
const graphEdges = document.getElementById("graphEdges");
const graphComponents = document.getElementById("graphComponents");
const connectivityRatio = document.getElementById("connectivityRatio");

// STEP 04
const criticalNode = document.getElementById("criticalNode");
const criticalityScore = document.getElementById("criticalityScore");

// STEP 05
const baselineConnectivity =
    document.getElementById("baselineConnectivity");

const failedConnectivity =
    document.getElementById("failedConnectivity");

const connectivityLoss =
    document.getElementById("connectivityLoss");

const resilienceIndex =
    document.getElementById("resilienceIndex");

const timeBefore =
    document.getElementById("timeBefore");

const timeAfter =
    document.getElementById("timeAfter");

const travelTimeIncrease =
    document.getElementById("travelTimeIncrease");

// ------------------------------------------------------------
// STATE
// ------------------------------------------------------------

let selectedImage = null;


// ------------------------------------------------------------
// HELPERS
// ------------------------------------------------------------

function showSection(section) {

    if (!section) {
        return;
    }

    section.hidden = false;

    section.style.setProperty(
        "display",
        "block",
        "important"
    );

    section.style.setProperty(
        "visibility",
        "visible",
        "important"
    );

    section.style.setProperty(
        "opacity",
        "1",
        "important"
    );

    section.style.setProperty(
        "height",
        "auto",
        "important"
    );

    section.style.setProperty(
        "max-height",
        "none",
        "important"
    );

    section.style.setProperty(
        "overflow",
        "visible",
        "important"
    );
}


function hideSection(section) {

    if (!section) {
        return;
    }

    section.hidden = true;

    section.style.setProperty(
        "display",
        "none",
        "important"
    );
}


function formatNumber(value, decimals = 2) {

    if (
        value === null ||
        value === undefined ||
        value === ""
    ) {
        return "—";
    }

    const number = Number(value);

    if (!Number.isFinite(number)) {
        return "—";
    }

    return number.toFixed(decimals);
}


function formatPercent(value) {

    if (
        value === null ||
        value === undefined ||
        value === ""
    ) {
        return "—";
    }

    const number = Number(value);

    if (!Number.isFinite(number)) {
        return "—";
    }

    return `${number.toFixed(2)}%`;
}


function formatMinutes(value) {

    if (
        value === null ||
        value === undefined ||
        value === ""
    ) {
        return "—";
    }

    const number = Number(value);

    if (!Number.isFinite(number)) {
        return "—";
    }

    return `${number.toFixed(2)} min`;
}


// IMPORTANT:
// Number(null) becomes 0 in JavaScript.
// This helper prevents null values from becoming fake 0 values.

function hasValidNumber(value) {

    return (
        value !== null &&
        value !== undefined &&
        value !== "" &&
        Number.isFinite(Number(value))
    );
}


// ------------------------------------------------------------
// PIPELINE RESULTS
// ------------------------------------------------------------

function updatePipelineResults(result) {

    console.log(
        "[RoadVision] Updating pipeline results:",
        result
    );

    const graph =
        result.graph || {};

    const criticality =
        result.criticality || {};

    const resilience =
        result.resilience || {};


    // ========================================================
    // STEP 03 — ROAD NETWORK GRAPH
    // ========================================================

    if (graphNodes) {

        graphNodes.textContent =
            graph.nodes !== undefined
                ? graph.nodes
                : "—";
    }


    if (graphEdges) {

        graphEdges.textContent =
            graph.edges !== undefined
                ? graph.edges
                : "—";
    }


    if (graphComponents) {

        graphComponents.textContent =
            graph.components !== undefined
                ? graph.components
                : "—";
    }


    if (connectivityRatio) {

        if (
            hasValidNumber(
                resilience.baseline_connectivity
            )
        ) {

            connectivityRatio.textContent =
                formatPercent(
                    Number(
                        resilience.baseline_connectivity
                    ) * 100
                );

        } else {

            connectivityRatio.textContent =
                "—";
        }
    }


    showSection(graphSection);


    // ========================================================
    // STEP 04 — CRITICALITY
    // ========================================================

    if (criticalNode) {

        if (
            criticality.critical_node !== null &&
            criticality.critical_node !== undefined
        ) {

            criticalNode.textContent =
                criticality.critical_node;

        } else {

            criticalNode.textContent =
                "—";
        }
    }


    if (criticalityScore) {

        if (
            hasValidNumber(
                criticality.betweenness
            )
        ) {

            criticalityScore.textContent =
                formatNumber(
                    criticality.betweenness,
                    4
                );

        } else {

            criticalityScore.textContent =
                "—";
        }
    }


    showSection(criticalitySection);


    // ========================================================
    // STEP 05 — NETWORK RESILIENCE
    // ========================================================

    // ------------------------------
    // Baseline connectivity
    // ------------------------------

    if (baselineConnectivity) {

        if (
            hasValidNumber(
                resilience.baseline_connectivity
            )
        ) {

            baselineConnectivity.textContent =
                formatPercent(
                    Number(
                        resilience.baseline_connectivity
                    ) * 100
                );

        } else {

            baselineConnectivity.textContent =
                "—";
        }
    }


    // ------------------------------
    // Post-failure connectivity
    // ------------------------------

    if (failedConnectivity) {

        if (
            hasValidNumber(
                resilience.post_failure_connectivity
            )
        ) {

            failedConnectivity.textContent =
                formatPercent(
                    Number(
                        resilience.post_failure_connectivity
                    ) * 100
                );

        } else {

            failedConnectivity.textContent =
                "—";
        }
    }


    // ------------------------------
    // Connectivity loss
    // ------------------------------

    if (connectivityLoss) {

        if (
            hasValidNumber(
                resilience.connectivity_loss
            )
        ) {

            connectivityLoss.textContent =
                formatPercent(
                    resilience.connectivity_loss
                );

        } else {

            connectivityLoss.textContent =
                "—";
        }
    }


    // ------------------------------
    // Resilience index
    // ------------------------------

    if (resilienceIndex) {

        if (
            hasValidNumber(
                resilience.resilience_index
            )
        ) {

            resilienceIndex.textContent =
                formatNumber(
                    resilience.resilience_index,
                    4
                );

        } else {

            resilienceIndex.textContent =
                "—";
        }
    }


    // ------------------------------
    // Travel time BEFORE failure
    // ------------------------------

    if (timeBefore) {

        if (
            hasValidNumber(
                resilience.time_before
            )
        ) {

            timeBefore.textContent =
                formatMinutes(
                    resilience.time_before
                );

        } else {

            timeBefore.textContent =
                "—";
        }
    }


    // ------------------------------
    // Travel time AFTER failure
    // ------------------------------

    if (timeAfter) {

        if (
            hasValidNumber(
                resilience.time_after
            )
        ) {

            timeAfter.textContent =
                formatMinutes(
                    resilience.time_after
                );

        } else {

            timeAfter.textContent =
                "NO ALTERNATE ROUTE";
        }
    }


    // ------------------------------
    // Travel-time impact
    // ------------------------------

    if (travelTimeIncrease) {

        if (
            hasValidNumber(
                resilience.percentage_increase
            )
        ) {

            travelTimeIncrease.textContent =
                formatPercent(
                    resilience.percentage_increase
                );

        } else if (
            hasValidNumber(
                resilience.time_increase
            )
        ) {

            travelTimeIncrease.textContent =
                formatMinutes(
                    resilience.time_increase
                );

        } else {

            travelTimeIncrease.textContent =
                "NO ALTERNATE ROUTE";
        }
    }


    // Show complete pipeline
    showSection(resultsSection);
    showSection(graphSection);
    showSection(criticalitySection);
    showSection(resilienceSection);


    console.log(
        "[RoadVision] Step 03 — Graph rendered."
    );

    console.log(
        "[RoadVision] Step 04 — Criticality rendered."
    );

    console.log(
        "[RoadVision] Step 05 — Resilience rendered."
    );
}


// ------------------------------------------------------------
// INITIAL STATE
// ------------------------------------------------------------

if (selectedFile) {
    selectedFile.style.display = "none";
}

if (processingStatus) {
    processingStatus.style.display = "none";
}

if (resultsSection) {
    resultsSection.style.display = "none";
}

if (originalImage) {
    originalImage.style.display = "none";
}

if (predictionImage) {
    predictionImage.style.display = "none";
}

hideSection(graphSection);
hideSection(criticalitySection);
hideSection(resilienceSection);


// ------------------------------------------------------------
// FILE SELECTION
// ------------------------------------------------------------

imageInput.addEventListener(
    "change",
    function () {

        const file =
            this.files[0];

        if (!file) {
            return;
        }

        selectedImage = file;

        fileName.textContent =
            file.name;

        const sizeMB =
            (
                file.size /
                (1024 * 1024)
            ).toFixed(2);

        fileSize.textContent =
            `${sizeMB} MB`;

        selectedFile.style.display =
            "flex";

        predictButton.disabled =
            false;

        resultsSection.style.display =
            "none";

        hideSection(graphSection);
        hideSection(criticalitySection);
        hideSection(resilienceSection);

        originalImage.style.display =
            "none";

        predictionImage.style.display =
            "none";

        console.log(
            "[RoadVision] File selected:",
            file.name
        );
    }
);


// ------------------------------------------------------------
// DRAG & DROP
// ------------------------------------------------------------

uploadCard.addEventListener(
    "dragover",
    function (event) {

        event.preventDefault();

        uploadCard.classList.add(
            "dragging"
        );
    }
);


uploadCard.addEventListener(
    "dragleave",
    function () {

        uploadCard.classList.remove(
            "dragging"
        );
    }
);


uploadCard.addEventListener(
    "drop",
    function (event) {

        event.preventDefault();

        uploadCard.classList.remove(
            "dragging"
        );

        const files =
            event.dataTransfer.files;

        if (
            !files ||
            files.length === 0
        ) {
            return;
        }

        const file =
            files[0];

        if (
            !file.type.startsWith(
                "image/"
            ) &&
            !file.name
                .toLowerCase()
                .endsWith(".tif") &&
            !file.name
                .toLowerCase()
                .endsWith(".tiff")
        ) {

            alert(
                "Please select a valid satellite image."
            );

            return;
        }

        selectedImage = file;

        fileName.textContent =
            file.name;

        const sizeMB =
            (
                file.size /
                (1024 * 1024)
            ).toFixed(2);

        fileSize.textContent =
            `${sizeMB} MB`;

        selectedFile.style.display =
            "flex";

        predictButton.disabled =
            false;

        resultsSection.style.display =
            "none";

        hideSection(graphSection);
        hideSection(criticalitySection);
        hideSection(resilienceSection);

        console.log(
            "[RoadVision] File dropped:",
            file.name
        );
    }
);


// ------------------------------------------------------------
// REMOVE FILE
// ------------------------------------------------------------

removeFile.addEventListener(
    "click",
    function () {

        selectedImage = null;

        imageInput.value = "";

        selectedFile.style.display =
            "none";

        predictButton.disabled =
            true;

        resultsSection.style.display =
            "none";

        hideSection(graphSection);
        hideSection(criticalitySection);
        hideSection(resilienceSection);

        originalImage.src = "";
        predictionImage.src = "";

        originalImage.style.display =
            "none";

        predictionImage.style.display =
            "none";
    }
);


// ------------------------------------------------------------
// RUN ANALYSIS
// ------------------------------------------------------------

predictButton.addEventListener(
    "click",
    async function () {

        console.log(
            "[RoadVision] Predict button clicked."
        );

        if (!selectedImage) {

            console.warn(
                "[RoadVision] No image selected."
            );

            return;
        }

        predictButton.disabled =
            true;

        processingStatus.style.display =
            "flex";

        resultsSection.style.display =
            "none";

        hideSection(graphSection);
        hideSection(criticalitySection);
        hideSection(resilienceSection);


        const formData =
            new FormData();

        formData.append(
            "file",
            selectedImage
        );


        try {

            console.log(
                `[RoadVision] Sending image to ${API_BASE_URL}/upload`
            );


            const response =
                await fetch(
                    `${API_BASE_URL}/upload`,
                    {
                        method: "POST",
                        body: formData
                    }
                );


            console.log(
                "[RoadVision] Response:",
                response.status
            );


            if (!response.ok) {

                let errorMessage =
                    "Road extraction failed.";

                try {

                    const errorData =
                        await response.json();

                    console.error(
                        "[RoadVision] Error:",
                        errorData
                    );

                    if (
                        errorData.detail
                    ) {

                        errorMessage =
                            errorData.detail;
                    }

                } catch (error) {

                    console.error(
                        "[RoadVision] Error response was not JSON:",
                        error
                    );
                }

                throw new Error(
                    errorMessage
                );
            }


            // =================================================
            // COMPLETE BACKEND RESULT
            // =================================================

            const result =
                await response.json();

            console.log(
                "[RoadVision] COMPLETE RESULT:",
                result
            );


            // =================================================
            // STEP 02 — ORIGINAL IMAGE
            // =================================================

            if (result.preview) {

                originalImage.src =
                    `${API_BASE_URL}${result.preview}?t=${Date.now()}`;

                originalImage.style.display =
                    "block";
            }


            // =================================================
            // STEP 02 — ROAD MASK
            // =================================================

            if (result.prediction) {

                predictionImage.src =
                    `${API_BASE_URL}${result.prediction}?t=${Date.now()}`;

                predictionImage.style.display =
                    "block";
            }


            // =================================================
            // STEP 03 + STEP 04 + STEP 05
            // =================================================

            updatePipelineResults(
                result
            );


            // =================================================
            // SHOW RESULTS
            // =================================================

            resultsSection.style.setProperty(
                "display",
                "block",
                "important"
            );

            resultsSection.style.setProperty(
                "visibility",
                "visible",
                "important"
            );


            console.log(
                "[RoadVision] Full analysis displayed."
            );


            resultsSection.scrollIntoView({
                behavior: "smooth",
                block: "start"
            });


        } catch (error) {

            console.error(
                "[RoadVision AI Error]",
                error
            );

            alert(
                `Unable to process the image.\n\n${error.message}\n\n` +
                `Make sure the FastAPI backend is running at ${API_BASE_URL}.`
            );

        } finally {

            processingStatus.style.display =
                "none";

            predictButton.disabled =
                false;
        }
    }
);


// ------------------------------------------------------------
// BACKEND CONNECTION CHECK
// ------------------------------------------------------------

async function checkBackendStatus() {

    try {

        const response =
            await fetch(
                `${API_BASE_URL}/`,
                {
                    method: "GET"
                }
            );


        if (response.ok) {

            console.log(
                "✓ [RoadVision] Backend connected:",
                API_BASE_URL
            );

            return true;
        }


        console.warn(
            `[RoadVision] Backend returned ${response.status}`
        );

    } catch (error) {

        console.warn(
            `[RoadVision] Backend is not reachable at ${API_BASE_URL}`,
            error
        );
    }

    return false;
}


// ------------------------------------------------------------
// PAGE LOAD
// ------------------------------------------------------------

checkBackendStatus();