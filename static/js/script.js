document.addEventListener("DOMContentLoaded", function () {
  const form = document.querySelector("form");
  if (!form) return;

  const imageUploadInput = document.getElementById("imageUpload");
  const previewContainer = document.getElementById("previewContainer");
  const imagePreview = document.getElementById("imagePreview");
  const fileInputText = document.querySelector(".file-input-container p"); // Get the "Drag & drop" text

  if (imageUploadInput && previewContainer && imagePreview && fileInputText) {
    imageUploadInput.addEventListener("change", function () {
      const file = this.files[0];
      if (file) {
        // A file is selected
        const reader = new FileReader();
        reader.onload = function (e) {
          imagePreview.src = e.target.result;
          previewContainer.style.display = "block";
          fileInputText.textContent = `Selected file: ${file.name}`; // Update text
        };
        reader.readAsDataURL(file);
      } else {
        // No file selected
        imagePreview.src = "";
        previewContainer.style.display = "none";
        fileInputText.textContent = "Drag & drop or click to select a chest CT scan"; // Reset text
      }
    });
  }

  const progressBar = document.getElementById("progressBar");
  const progressStatus = document.getElementById("progressStatus");
  const loadingIndicator = document.getElementById("loadingIndicator");
  const submitBtn = document.getElementById("submitBtn");

  if (!progressBar || !progressStatus || !loadingIndicator || !submitBtn) {
    console.warn("Progress or loading elements not found on this page.");
    return;
  }

  form.addEventListener("submit", async function (event) {
    event.preventDefault();

    const formData = new FormData(form);
    const actionURL = form.getAttribute("action"); // Route is already set in your HTML

    submitBtn.disabled = true;
    loadingIndicator.style.display = "block";

    let progress = 0;
    const progressInterval = setInterval(() => {
      progress += 5;
      if (progress >= 95) progress = 95;
      progressBar.style.width = progress + "%";
      progressStatus.textContent = "Processing... " + progress + "%";
    }, 500);

    try {
      const response = await fetch(actionURL, { method: "POST", body: formData });

      clearInterval(progressInterval);
      progressBar.style.width = "100%";
      progressStatus.textContent = "Completed! Redirecting...";

      const data = await response.json();
      if (data.success) {
        sessionStorage.setItem("detectionResult", JSON.stringify(data.result));
        
        setTimeout(() => {
          window.location.href = data.redirect; // Redirect from server response
        }, 800);
      } else {
        alert(data.error || "Detection failed. Try again!");
      }
    } catch (err) {
      clearInterval(progressInterval);
      progressBar.style.width = "0%";
      progressStatus.textContent = "Error occurred";
      alert("An error occurred: " + err.message);
    } finally {
      setTimeout(() => {
        loadingIndicator.style.display = "none";
        submitBtn.disabled = false;
      }, 1000);
    }
  });
});
