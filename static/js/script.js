let highestZ = 100;

document.querySelectorAll(".window").forEach(initWindow);

function triggerOpeningSequence() {
  const hiddenWindows = document.querySelectorAll(".hidden-start");
  const delays = [200, 400, 550, 650, 850];

  hiddenWindows.forEach((win, index) => {
    let delayTime = delays[index] !== undefined ? delays[index] : index * 500;

    setTimeout(() => {
      win.classList.remove("hidden-start");
      win.classList.add("pop-in");

      win.style.zIndex = ++highestZ;
    }, delayTime);
  });
}

function initWindow(win) {
  if (win.classList.contains("center-start")) {
    const w = parseFloat(win.style.width) || 400;
    const h = parseFloat(win.style.height) || 300;

    const leftPos = (window.innerWidth - w) / 2;
    const topPos = (window.innerHeight - h) / 2;

    win.style.left = leftPos + "px";
    win.style.top = topPos + "px";
  }

  win.style.zIndex = highestZ++;
  win.addEventListener("mousedown", () => {
    win.style.zIndex = ++highestZ;
  });

  const topbar = win.querySelector(".topbar");
  if (topbar && !win.classList.contains("intro-window")) {
    topbar.addEventListener("mousedown", (e) => {
      if (e.target.closest("button")) return;
      e.preventDefault();
      win.style.zIndex = ++highestZ;
      let prevX = e.clientX;
      let prevY = e.clientY;

      function mousemove(e) {
        let newX = e.clientX - prevX;
        let newY = e.clientY - prevY;
        const rect = win.getBoundingClientRect();
        let newLeft = rect.left + newX;
        let newTop = rect.top + newY;
        const margin = 0;

        newLeft = Math.max(
          margin,
          Math.min(newLeft, window.innerWidth - rect.width - margin),
        );
        newTop = Math.max(
          margin,
          Math.min(newTop, window.innerHeight - rect.height - margin),
        );

        win.style.left = newLeft + "px";
        win.style.top = newTop + "px";
        prevX = e.clientX;
        prevY = e.clientY;
      }

      function mouseup() {
        window.removeEventListener("mousemove", mousemove);
        window.removeEventListener("mouseup", mouseup);
      }
      window.addEventListener("mousemove", mousemove);
      window.addEventListener("mouseup", mouseup);
    });
  }

  const maximizeButton = win.querySelector(".maximize-button");
  const closeButton = win.querySelector(".close-button");
  const contentButton = win.querySelector(".minimized-content button");

  if (maximizeButton) {
    let isMaximized = false;
    let initialWidth, initialHeight, initialX, initialY;
    const rootStyles = getComputedStyle(document.body);
    const maximizeImage = rootStyles.getPropertyValue("--maximize").trim();
    const restoreImage = rootStyles.getPropertyValue("--restore").trim();

    const toggleMaximize = (e) => {
      if (e) e.stopPropagation();
      isMaximized = !isMaximized;
      const imgElement = maximizeButton.querySelector("img");

      if (isMaximized) {
        initialWidth = parseInt(win.style.width || win.offsetWidth, 10);
        initialHeight = parseInt(win.style.height || win.offsetHeight, 10);
        initialX = parseInt(win.style.left || win.offsetLeft, 10);
        initialY = parseInt(win.style.top || win.offsetTop, 10);

        win.classList.add("maximized");
        if (imgElement)
          imgElement.src = restoreImage.replace(/url\(["']?|["']?\)/g, "");
      } else {
        win.classList.remove("maximized");
        if (imgElement)
          imgElement.src = maximizeImage.replace(/url\(["']?|["']?\)/g, "");
        win.style.width = initialWidth + "px";
        win.style.height = initialHeight + "px";
        win.style.left = initialX + "px";
        win.style.top = initialY + "px";
      }
    };

    maximizeButton.addEventListener("click", toggleMaximize);
    if (
      contentButton &&
      ["Try me!", "Read more", "Try out by yourself!", "Discover"].includes(
        contentButton.innerText.trim(),
      )
    ) {
      contentButton.addEventListener("click", toggleMaximize);
    }
  }

  const closeWindow = (e) => {
    if (e) e.stopPropagation();

    win.remove();

    if (win.classList.contains("intro-window")) {
      triggerOpeningSequence();
    }
  };

  if (closeButton) {
    closeButton.addEventListener("click", closeWindow);
  }

  if (contentButton && contentButton.innerText.trim() === "Got it!") {
    contentButton.addEventListener("click", closeWindow);
  }
}

const uploadInput = document.getElementById("upload");
const loading = document.getElementById("loading");
const resultsContainer = document.getElementById("results-container");
const preview = document.getElementById("preview");
const resultJerome = document.getElementById("result-jerome");
const resultYolov8n = document.getElementById("result-yolov8n");
const metricsJerome = document.getElementById("metrics-jerome");
const metricsYolov8n = document.getElementById("metrics-yolov8n");
const sampleThumbs = document.querySelectorAll(".sample-thumb"); // Sélectionne les nouvelles images

// --- FONCTION PRINCIPALE ---
// Cette fonction fait le travail d'envoi et d'affichage
async function runAnalysis(file) {
  if (!file) return;

  resultsContainer.style.display = "none";
  loading.style.display = "flex";

  // Affiche l'aperçu local
  preview.src = URL.createObjectURL(file);

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("/compare", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) throw new Error(`Erreur: ${response.statusText}`);

    const data = await response.json();

    // Mise à jour de l'interface
    resultJerome.src = data.result_jerome;
    metricsJerome.innerHTML = `Temps: ${data.time_jerome.toFixed(3)}s<br>Détections: ${data.num_detections_jerome}`;

    resultYolov8n.src = data.result_yolov8n;
    metricsYolov8n.innerHTML = `Temps: ${data.time_yolov8n.toFixed(3)}s<br>Détections: ${data.num_detections_yolov8n}`;

    loading.style.display = "none";
    resultsContainer.style.display = "flex";
  } catch (error) {
    alert(`Erreur: ${error.message}`);
    loading.style.display = "none";
  }
}

// --- ÉCOUTEURS D'ÉVÉNEMENTS ---

// 1. Cas classique : L'utilisateur upload un fichier
uploadInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  runAnalysis(file);
});

// 2. Cas "Sample" : L'utilisateur clique sur une image exemple
sampleThumbs.forEach((img) => {
  img.addEventListener("click", async () => {
    try {
      // On récupère l'image depuis le serveur (src) pour en faire un "Blob"
      const response = await fetch(img.src);
      const blob = await response.blob();

      // On convertit le Blob en un objet File (comme si c'était un upload)
      const filename = img.getAttribute("data-name") || "sample.jpg";
      const file = new File([blob], filename, { type: blob.type });

      // On lance l'analyse
      runAnalysis(file);
    } catch (err) {
      alert("Impossible de charger l'image exemple.");
      console.error(err);
    }
  });
});
