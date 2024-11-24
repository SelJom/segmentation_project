// Interactive Segmentation DOM elements
const inputCanvas = document.getElementById('inputCanvas');
const segmentedCanvas = document.getElementById('segmentedCanvas');
const imageUpload = document.getElementById('imageUpload');
const clearPointsButton = document.getElementById('clearPoints');
const voidsButton = document.getElementById('voidsButton');
const chipsButton = document.getElementById('chipsButton');
const retrainModelButton = document.getElementById('retrainModelButton');
const etaDisplay = document.getElementById('etaDisplay');

// Automatic Segmentation DOM elements
const automaticImageUpload = document.getElementById('automaticImageUpload');
const automaticProcessedImage = document.getElementById('automaticProcessedImage');
const resultsTableBody = document.getElementById('resultsTableBody');
const clearTableButton = document.getElementById('clearTableButton');
const exportTableButton = document.getElementById('exportTableButton');

// Constants for consistent canvas and SAM model dimensions
const CANVAS_SIZE = 512;
inputCanvas.width = CANVAS_SIZE;
inputCanvas.height = CANVAS_SIZE;
segmentedCanvas.width = CANVAS_SIZE;
segmentedCanvas.height = CANVAS_SIZE;

// Interactive segmentation variables
let points = { Voids: [], Chips: [] };
let labels = { Voids: [], Chips: [] };
let currentClass = 'Voids';
let imageUrl = '';
let originalImageWidth = 0;
let originalImageHeight = 0;
let trainingInProgress = false;

// Disable right-click menu on canvas
inputCanvas.addEventListener('contextmenu', (event) => event.preventDefault());

// Switch between classes
voidsButton.addEventListener('click', () => {
    currentClass = 'Voids';
    voidsButton.classList.add('active');
    chipsButton.classList.remove('active');
    clearAndRestorePoints();
});

chipsButton.addEventListener('click', () => {
    currentClass = 'Chips';
    chipsButton.classList.add('active');
    voidsButton.classList.remove('active');
    clearAndRestorePoints();
});

// Handle image upload for interactive tool
imageUpload.addEventListener('change', async (event) => {
    const file = event.target.files[0];
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/upload', { method: 'POST', body: formData });
        const data = await response.json();
        if (data.error) {
            console.error('Error uploading image:', data.error);
            return;
        }

        imageUrl = data.image_url;
        console.log('Uploaded image URL:', imageUrl);

        const img = new Image();
        img.src = imageUrl;
        img.onload = () => {
            console.log('Image loaded:', img.width, img.height);
            originalImageWidth = img.width;
            originalImageHeight = img.height;
            resizeAndDrawImage(inputCanvas, img);
            resizeAndDrawImage(segmentedCanvas, img);
        };
        img.onerror = () => {
            console.error('Failed to load image from URL:', imageUrl);
        };
    } catch (error) {
        console.error('Failed to upload image:', error);
    }
});


// Handle input canvas clicks
inputCanvas.addEventListener('mousedown', async (event) => {
    const rect = inputCanvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) * (originalImageWidth / CANVAS_SIZE);
    const y = (event.clientY - rect.top) * (originalImageHeight / CANVAS_SIZE);

    if (event.button === 2) {
        points[currentClass].push([x, y]);
        labels[currentClass].push(0); // Exclude point (red)
    } else if (event.button === 0) {
        points[currentClass].push([x, y]);
        labels[currentClass].push(1); // Include point (green)
    }

    drawPoints();
    await updateSegmentation();
});

// Clear points for current class
clearPointsButton.addEventListener('click', () => {
    points[currentClass] = [];
    labels[currentClass] = [];
    drawPoints();
    resetSegmentation();
});

function resizeAndDrawImage(canvas, img) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear the canvas

    // Scale the image to fit within the canvas
    const scale = Math.min(canvas.width / img.width, canvas.height / img.height);
    const x = (canvas.width - img.width * scale) / 2;
    const y = (canvas.height - img.height * scale) / 2;

    ctx.drawImage(img, x, y, img.width * scale, img.height * scale);
}


// Draw points on canvases
function drawPoints() {
    [inputCanvas, segmentedCanvas].forEach((canvas) => {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

        const img = new Image();
        img.src = imageUrl;
        img.onload = () => {
            resizeAndDrawImage(canvas, img);

            points[currentClass].forEach(([x, y], i) => {
                const scaledX = x * (CANVAS_SIZE / originalImageWidth);
                const scaledY = y * (CANVAS_SIZE / originalImageHeight);
                ctx.beginPath();
                ctx.arc(scaledX, scaledY, 5, 0, 2 * Math.PI);
                ctx.fillStyle = labels[currentClass][i] === 1 ? 'green' : 'red';
                ctx.fill();
            });
        };
        img.onerror = () => {
            console.error('Error loading image for canvas:', img.src);
        };
    });
}

async function updateSegmentation() {
    try {
        const response = await fetch('/segment', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ points: points[currentClass], labels: labels[currentClass], class: currentClass.toLowerCase() })
        });

        const data = await response.json();

        if (data.error) {
            console.error('Error during segmentation:', data.error);
            alert(`Segmentation error: ${data.error}`);
            return;
        }

        console.log('Segmentation result:', data);

        const img = new Image();
        img.src = `${data.segmented_url}?t=${new Date().getTime()}`; // Add timestamp to prevent caching
        img.onload = () => {
            console.log('Segmented image loaded successfully:', img.src);
            resizeAndDrawImage(segmentedCanvas, img); // Render the segmented image
        };
        img.onerror = () => {
            console.error('Failed to load segmented image:', img.src);
            alert('Failed to load the segmented image.');
        };
    } catch (error) {
        console.error('Error updating segmentation:', error);
        alert('Failed to process segmentation.');
    }
}

// Reset segmented canvas
function resetSegmentation() {
    const ctx = segmentedCanvas.getContext('2d');
    ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    const img = new Image();
    img.src = imageUrl;
    img.onload = () => resizeAndDrawImage(segmentedCanvas, img);
}

// Handle automatic segmentation
automaticImageUpload.addEventListener('change', async (event) => {
    const file = event.target.files[0];
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/automatic_segment', { method: 'POST', body: formData });
        const data = await response.json();
        if (data.error) return console.error('Error during automatic segmentation:', data.error);

        // Display the processed image
        const processedImage = document.getElementById('automaticProcessedImage');
        processedImage.src = `${data.segmented_url}?t=${new Date().getTime()}`;
        processedImage.style.display = 'block';

        // Optionally append the table data
        appendRowToTable(data.table_data); 
    } catch (error) {
        console.error('Failed to process image automatically:', error);
    }
});

function appendRowToTable(tableData) {
    // Remove duplicates based on the image name and chip number
    const existingRows = Array.from(resultsTableBody.querySelectorAll('tr'));
    const existingIdentifiers = existingRows.map(row => {
        const cells = row.querySelectorAll('td');
        return `${cells[0]?.textContent}_${cells[1]?.textContent}`; // Combine Image Name and Chip #
    });

    tableData.chips.forEach((chip, index) => {
        const uniqueId = `${tableData.image_name}_${chip.chip_number}`;
        if (existingIdentifiers.includes(uniqueId)) return; // Skip if already present

        const row = document.createElement('tr');

        // Image Name (unchanged for each chip)
        const imageNameCell = document.createElement('td');
        imageNameCell.textContent = tableData.image_name;
        row.appendChild(imageNameCell);

        // Chip # (1, 2, etc.)
        const chipNumberCell = document.createElement('td');
        chipNumberCell.textContent = chip.chip_number;
        row.appendChild(chipNumberCell);

        // Chip Area
        const chipAreaCell = document.createElement('td');
        chipAreaCell.textContent = chip.chip_area.toFixed(2);
        row.appendChild(chipAreaCell);

        // Void % (Total void area / Chip area * 100)
        const voidPercentageCell = document.createElement('td');
        voidPercentageCell.textContent = chip.void_percentage.toFixed(2);
        row.appendChild(voidPercentageCell);

        // Max Void % (Largest void area / Chip area * 100)
        const maxVoidPercentageCell = document.createElement('td');
        maxVoidPercentageCell.textContent = chip.max_void_percentage.toFixed(2);
        row.appendChild(maxVoidPercentageCell);

        resultsTableBody.appendChild(row);
    });
}

// Handle automatic segmentation
automaticImageUpload.addEventListener('change', async (event) => {
    const file = event.target.files[0];
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/automatic_segment', { method: 'POST', body: formData });
        const data = await response.json();
        if (data.error) return console.error('Error during automatic segmentation:', data.error);

        automaticProcessedImage.src = `${data.segmented_url}?t=${new Date().getTime()}`;
        automaticProcessedImage.style.display = 'block';
        appendRowToTable(data.table_data); // Append new data to the table
    } catch (error) {
        console.error('Failed to process image automatically:', error);
    }
});

// Clear table
clearTableButton.addEventListener('click', () => {
    resultsTableBody.innerHTML = '';
});

// Export table to CSV
exportTableButton.addEventListener('click', () => {
    const rows = Array.from(resultsTableBody.querySelectorAll('tr'));
    const csvContent = [
        ['Image Name', 'Chip #', 'Chip Area', 'Void %', 'Max Void %'],
        ...rows.map(row =>
            Array.from(row.children).map(cell => cell.textContent)
        ),
    ]
        .map(row => row.join(','))
        .join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'segmentation_results.csv';
    link.click();
    URL.revokeObjectURL(url);
});
saveBothButton.addEventListener('click', async () => {
    const imageName = imageUrl.split('/').pop(); // Extract the image name from the URL
    if (!imageName) {
        alert("No image to save.");
        return;
    }

    const confirmSave = confirm("Are you sure you want to save both voids and chips segmentations?");
    if (!confirmSave) return;

    try {
        const response = await fetch('/save_both', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_name: imageName })
        });
        const result = await response.json();
        if (response.ok) {
            alert(result.message);
        } else {
            alert("Failed to save segmentations.");
        }
    } catch (error) {
        console.error("Error saving segmentations:", error);
        alert("Failed to save segmentations.");
    }
});
// Update the "historyButton" click listener to populate the list correctly
document.getElementById('historyButton').addEventListener('click', async () => {
    try {
        const response = await fetch('/get_history'); // Fetch the saved history
        const result = await response.json();

        if (response.ok) {
            const historyList = document.getElementById('historyList');
            historyList.innerHTML = ''; // Clear the list

            if (result.images.length === 0) {
                historyList.innerHTML = '<li class="list-group-item">No images found in history.</li>';
                return;
            }

            result.images.forEach(image => {
                const listItem = document.createElement('li');
                listItem.className = 'list-group-item';

                const imageName = document.createElement('span');
                imageName.textContent = image;

                const deleteButton = document.createElement('button');
                deleteButton.className = 'btn btn-danger btn-sm';
                deleteButton.textContent = 'Delete';
                deleteButton.addEventListener('click', async () => {
                    if (confirm(`Are you sure you want to delete ${image}?`)) {
                        await deleteHistoryItem(image, listItem);
                    }
                });

                listItem.appendChild(imageName);
                listItem.appendChild(deleteButton);
                historyList.appendChild(listItem);
            });

            new bootstrap.Modal(document.getElementById('historyModal')).show();
        } else {
            alert("Failed to fetch history.");
        }
    } catch (error) {
        console.error("Error fetching history:", error);
        alert("Failed to fetch history.");
    }
});

// Function to delete history item
async function deleteHistoryItem(imageName, listItem) {
    try {
        const response = await fetch('/delete_history_item', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_name: imageName })
        });
        const result = await response.json();

        if (response.ok) {
            alert(result.message);
            listItem.remove(); // Remove the item from the list
        } else {
            alert("Failed to delete image.");
        }
    } catch (error) {
        console.error("Error deleting image:", error);
        alert("Failed to delete image.");
    }
}

historyButton.addEventListener('click', async () => {
    try {
        const response = await fetch('/get_history');
        const result = await response.json();

        if (response.ok) {
            const historyList = document.getElementById('historyList');
            historyList.innerHTML = ''; // Clear the list

            if (result.images.length === 0) {
                historyList.innerHTML = '<li class="list-group-item">No images found in history.</li>';
                return;
            }

            result.images.forEach(image => {
                const listItem = document.createElement('li');
                listItem.className = 'list-group-item d-flex justify-content-between align-items-center';
                listItem.textContent = image;

                const deleteButton = document.createElement('button');
                deleteButton.className = 'btn btn-danger btn-sm';
                deleteButton.textContent = 'Delete';
                deleteButton.addEventListener('click', async () => {
                    if (confirm(`Are you sure you want to delete ${image}?`)) {
                        await deleteHistoryItem(image, listItem);
                    }
                });

                listItem.appendChild(deleteButton);
                historyList.appendChild(listItem);
            });

            new bootstrap.Modal(document.getElementById('historyModal')).show();
        } else {
            alert("Failed to fetch history.");
        }
    } catch (error) {
        console.error("Error fetching history:", error);
        alert("Failed to fetch history.");
    }
});

// Function to delete history item
async function deleteHistoryItem(imageName, listItem) {
    try {
        const response = await fetch('/delete_history_item', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_name: imageName })
        });
        const result = await response.json();

        if (response.ok) {
            alert(result.message);
            listItem.remove(); // Remove the item from the list
        } else {
            alert("Failed to delete image.");
        }
    } catch (error) {
        console.error("Error deleting image:", error);
        alert("Failed to delete image.");
    }
}

// Handle Retrain Model button click
retrainModelButton.addEventListener('click', async () => {
    if (!trainingInProgress) {
        const confirmRetrain = confirm("Are you sure you want to retrain the model?");
        if (!confirmRetrain) return;

        try {
            const response = await fetch('/retrain_model', { method: 'POST' });
            const result = await response.json();

            if (response.ok) {
                // Update button to "Cancel Training"
                trainingInProgress = true;
                retrainModelButton.textContent = "Cancel Training";
                retrainModelButton.classList.replace("btn-primary", "btn-danger");
                startTrainingMonitor(); // Start monitoring the training status
            } else {
                alert(result.error || "Failed to start retraining.");
            }
        } catch (error) {
            console.error("Error starting training:", error);
            alert("An error occurred while starting the training process.");
        }
    } else {
        // Handle cancel training
        const confirmCancel = confirm("Are you sure you want to cancel the training?");
        if (!confirmCancel) return;

        try {
            const response = await fetch('/cancel_training', { method: 'POST' });
            const result = await response.json();

            if (response.ok) {
                // Reset button to "Retrain Model"
                trainingInProgress = false;
                retrainModelButton.textContent = "Retrain Model";
                retrainModelButton.classList.replace("btn-danger", "btn-primary");
                alert(result.message || "Training canceled successfully.");
            } else {
                alert(result.error || "Failed to cancel training.");
            }
        } catch (error) {
            console.error("Error canceling training:", error);
            alert("An error occurred while canceling the training process.");
        }
    }
});


function startTrainingMonitor() {
    const monitorInterval = setInterval(async () => {
        try {
            const response = await fetch('/training_status');
            const result = await response.json();

            const retrainButton = document.getElementById('retrainModelButton');
            const cancelButton = document.getElementById('cancelTrainingButton');
            const etaDisplay = document.getElementById('etaDisplay');

            if (result.status === 'running') {
                // Show training progress
                retrainButton.style.display = 'none';
                cancelButton.style.display = 'inline-block';
                etaDisplay.textContent = `Estimated Time Left: ${result.eta || "Calculating..."}`;
            } else if (result.status === 'idle' || result.status === 'cancelled') {
                // Revert button to "Retrain Model" (blue)
                cancelButton.style.display = 'none';
                retrainButton.style.display = 'inline-block';
                retrainButton.textContent = 'Retrain Model';
                retrainButton.classList.replace('btn-danger', 'btn-primary');
                etaDisplay.textContent = '';

                // Stop monitoring if training is idle
                if (result.status === 'idle') {
                    clearInterval(monitorInterval);
                }
            }
        } catch (error) {
            console.error("Error fetching training status:", error);
        }
    }, 5000); // Poll every 5 seconds
}

function resetTrainingUI() {
    trainingInProgress = false;
    retrainModelButton.textContent = "Retrain Model";
    retrainModelButton.classList.replace("btn-danger", "btn-primary");
    etaDisplay.textContent = "";
}

clearHistoryButton.addEventListener('click', async () => {
    const confirmClear = confirm("Are you sure you want to clear the history? This will delete all images and masks.");
    if (!confirmClear) return;

    try {
        const response = await fetch('/clear_history', { method: 'POST' });
        const result = await response.json();
        if (response.ok) {
            alert(result.message);
            // Optionally update UI to reflect the cleared history
            const historyList = document.getElementById('historyList');
            if (historyList) historyList.innerHTML = '<li class="list-group-item">No images found in history.</li>';
        } else {
            alert("Failed to clear history.");
        }
    } catch (error) {
        console.error("Error clearing history:", error);
        alert("Failed to clear history.");
    }
});

// Toggle training progress display
function showTrainingProgress(message = "Initializing...", timeLeft = "Calculating...") {
    document.getElementById("trainingProgress").style.display = "block";
    document.getElementById("progressMessage").textContent = message;
    document.getElementById("estimatedTimeLeft").textContent = `Estimated Time Left: ${timeLeft}`;
}

function hideTrainingProgress() {
    document.getElementById("trainingProgress").style.display = "none";
}

// Toggle Cancel Training Button
function showCancelTrainingButton() {
    document.getElementById("cancelTrainingButton").style.display = "inline-block";
    document.getElementById("retrainModelButton").style.display = "none";
}

function hideCancelTrainingButton() {
    document.getElementById("cancelTrainingButton").style.display = "none";
    document.getElementById("retrainModelButton").style.display = "inline-block";
}

// Add event listener to Cancel Training button
document.getElementById("cancelTrainingButton").addEventListener("click", async () => {
    const confirmCancel = confirm("Are you sure you want to cancel training?");
    if (!confirmCancel) return;

    try {
        const response = await fetch("/cancel_training", { method: "POST" });
        const result = await response.json();

        if (result.message) {
            alert(result.message);
            hideTrainingProgress();
            hideCancelTrainingButton();
        }
    } catch (error) {
        console.error("Error canceling training:", error);
        alert("Failed to cancel training.");
    }
});
// Handle training status updates
socket.on('training_status', (data) => {
    const trainingButton = document.getElementById('retrainModelButton');
    const cancelButton = document.getElementById('cancelTrainingButton');

    if (data.status === 'completed') {
        // Update UI: change "Cancel Training" to "Retrain Model"
        trainingButton.style.display = 'inline-block';
        cancelButton.style.display = 'none';

        // Show a popup or notification for training completion
        alert(data.message || "Training completed successfully!");
    } else if (data.status === 'failed') {
        // Update UI: change "Cancel Training" to "Retrain Model"
        trainingButton.style.display = 'inline-block';
        cancelButton.style.display = 'none';

        // Show a popup or notification for training failure
        alert(data.message || "Training failed. Please try again.");
    }
});

socket.on('button_update', (data) => {
    const retrainButton = document.getElementById('retrainModelButton');
    const cancelButton = document.getElementById('cancelTrainingButton');

    if (data.action === 'retrain') {
        // Update to "Retrain Model" button
        retrainButton.style.display = 'inline-block';
        retrainButton.textContent = 'Retrain Model';
        retrainButton.classList.replace('btn-danger', 'btn-primary');
        cancelButton.style.display = 'none';
    }
});

function updateButtonToRetrainModel() {
    const button = document.getElementById('retrainModelButton');
    button.innerText = "Retrain Model";
    button.classList.replace("btn-danger", "btn-primary");
    button.disabled = false;
}


socket.on('training_status', (data) => {
    const retrainButton = document.getElementById('retrainModelButton');
    const cancelButton = document.getElementById('cancelTrainingButton');

    if (data.status === 'completed') {
        retrainButton.style.display = 'inline-block';  // Show retrain button
        retrainButton.textContent = "Retrain Model";
        retrainButton.classList.replace("btn-danger", "btn-primary");
        cancelButton.style.display = 'none';  // Hide cancel button

        // Notify user
        alert(data.message);
    } else if (data.status === 'cancelled') {
        retrainButton.style.display = 'inline-block';
        retrainButton.textContent = "Retrain Model";
        retrainButton.classList.replace("btn-danger", "btn-primary");
        cancelButton.style.display = 'none';

        // Notify user
        alert(data.message);
    }
});

// Ensure the modal backdrop is properly removed when the modal is closed
document.getElementById('historyModal').addEventListener('hidden.bs.modal', function () {
    document.body.classList.remove('modal-open');
    const backdrop = document.querySelector('.modal-backdrop');
    if (backdrop) {
        backdrop.remove();
    }
});
