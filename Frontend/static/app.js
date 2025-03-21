// ✅ Load Live2D Model (Standalone, No VTube Studio)
function initializeLive2D() {
    const app = new PIXI.Application({
        view: document.getElementById("live2dCanvas"),
        width: 600,
        height: 600,
        transparent: true
    });

    PIXI.live2d.Live2DModel.from("/static/models/kei_basic_free.model3.json").then(model => {
        model.scale.set(0.75);
        model.anchor.set(0.5, 1);
        model.position.set(app.screen.width / 2, app.screen.height * 1.02);
        app.stage.addChild(model);
        window.live2DModel = model;
    }).catch(error => {
        console.log("❌ Failed to load Live2D model:", error);
    });
}

document.addEventListener("DOMContentLoaded", initializeLive2D);

// ✅ Traffic Light Control
let currentTrafficLight = null;

function setTrafficLight(color) {
    currentTrafficLight = color; // Track the current light state

    const states = {
        red: { selector: ".red", text: "AI is Speaking" },
        yellow: { selector: ".yellow", text: "Generating Response" },
        green: { selector: ".green", text: "AI is Listening" }
    };

    Object.keys(states).forEach(light => {
        document.querySelector(states[light].selector).style.opacity = (light === color) ? "1" : "0.2";
    });

    document.querySelectorAll(".light-text").forEach(text => {
        text.style.display = "none";
        text.classList.remove("highlight-text");
    });

    const activeText = document.querySelector(states[color].selector).nextElementSibling;
    activeText.style.display = "block";
    activeText.classList.add("highlight-text");
}

// ✅ Reset Traffic Light
function resetTrafficLight() {
    currentTrafficLight = null;
    document.querySelectorAll(".light").forEach(light => light.style.opacity = "0.2");
    document.querySelectorAll(".light-text").forEach(text => text.style.display = "none");
}

// ✅ Microphone-Based Lip Sync (Only When AI is Speaking)
let micActive = false;

navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
    const audioContext = new AudioContext();
    const analyser = audioContext.createAnalyser();
    analyser.fftSize = 256;

    const source = audioContext.createMediaStreamSource(stream);
    source.connect(analyser);

    const dataArray = new Uint8Array(analyser.fftSize);

    function updateMouth() {
        if (!micActive || currentTrafficLight !== "red") {
            if (window.live2DModel?.internalModel?.coreModel) {
                window.live2DModel.internalModel.coreModel.setParameterValueById("ParamMouthOpenY", 0);
            }
            requestAnimationFrame(updateMouth);
            return;
        }

        analyser.getByteTimeDomainData(dataArray);
        let sum = 0;

        for (let i = 0; i < dataArray.length; i++) {
            sum += Math.abs(dataArray[i] - 128);
        }

        let amplitude = (sum / dataArray.length / 128) * 3;
        amplitude = Math.max(0, Math.min(amplitude, 1));

        if (window.live2DModel?.internalModel?.coreModel) {
            window.live2DModel.internalModel.coreModel.setParameterValueById("ParamMouthOpenY", amplitude);
        }

        requestAnimationFrame(updateMouth);
    }

    updateMouth();
}).catch((err) => {
    console.log("❌ Could not access microphone:", err.message);
});

// ✅ Handle Speech Recognition
const micButton = document.getElementById("micButton");
const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
recognition.lang = "en-US";
recognition.continuous = false;
recognition.interimResults = false;

let isListening = false;

// ✅ Handle Mic Button Click
micButton.addEventListener("click", () => {
    isListening ? stopListening() : startListening();
});

// ✅ Start Listening for User Input
function startListening() {
    micActive = true; // ✅ Activate mic
    setTrafficLight("green"); // ✅ AI is Listening
    micButton.style.background = "rgba(0, 255, 0, 0.7)";
    micButton.innerText = "🔴";
    recognition.start();
    isListening = true;
}

// ✅ Stop Listening
function stopListening() {
    micActive = false; // ✅ Deactivate mic
    recognition.stop();
    setTrafficLight("yellow"); // ✅ AI is Processing
    micButton.style.background = "rgba(255, 75, 92, 0.7)";
    micButton.innerText = "🎤";
    isListening = false;
}

// ✅ Handle Speech Recognition Result
recognition.onresult = (event) => {
    const userSpeech = event.results[0][0].transcript;
    console.log("🎤 Recognized Speech:", userSpeech);
    stopListening();
    sendMessageToBackend(userSpeech);
};

// ✅ Handle Recognition Errors
recognition.onerror = (event) => {
    console.log("❌ Speech Recognition Error:", event.error);
    stopListening();
};

// ✅ Send User Input to Backend & Play AI Response
function sendMessageToBackend(message) {
    console.log("📡 Sending message to backend:", message);

    fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: message })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error("Server Error: " + response.status);
        }
        return response.blob();
    })
    .then(blob => {
        console.log("🤖 Bot Response Audio received");
        const audioUrl = URL.createObjectURL(blob);
        const audio = new Audio(audioUrl);
        audio.play();

        setTrafficLight("red"); // ✅ AI is Speaking
        startLipSync(audio);

        audio.onended = () => {
            resetTrafficLight(); // ✅ Reset after AI response
        };
    })
    .catch(error => {
        console.log("❌ Error sending message:", error);
    });
}

// ✅ AI Voice Response Lip Sync (Only When AI is Speaking)
function startLipSync(audioElement) {
    const audioContext = new AudioContext();
    const analyser = audioContext.createAnalyser();
    analyser.fftSize = 256;

    const source = audioContext.createMediaElementSource(audioElement);
    source.connect(analyser);
    analyser.connect(audioContext.destination);

    const dataArray = new Uint8Array(analyser.fftSize);

    function animateLipSync() {
        if (currentTrafficLight !== "red") {
            if (window.live2DModel?.internalModel?.coreModel) {
                window.live2DModel.internalModel.coreModel.setParameterValueById("ParamMouthOpenY", 0);
            }
            requestAnimationFrame(animateLipSync);
            return;
        }

        analyser.getByteTimeDomainData(dataArray);
        let sum = 0;
        for (let i = 0; i < dataArray.length; i++) {
            sum += Math.abs(dataArray[i] - 128);
        }

        let amplitude = (sum / dataArray.length / 128) * 3;
        amplitude = Math.max(0, Math.min(amplitude, 1));

        if (window.live2DModel?.internalModel?.coreModel) {
            window.live2DModel.internalModel.coreModel.setParameterValueById("ParamMouthOpenY", amplitude);
        }

        if (!audioElement.paused) {
            requestAnimationFrame(animateLipSync);
        }
    }

    animateLipSync();
}
