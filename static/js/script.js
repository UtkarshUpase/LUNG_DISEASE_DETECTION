document.addEventListener("DOMContentLoaded", function () {
  const form = document.querySelector("form");
  if (!form) return;

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
