import { useEffect, useRef, useState } from "react";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";

const MODEL_URL = "/hand_landmarker.task";
const AI_API_URL = "http://127.0.0.1:8000/predict";
const HISTORY_API_URL = "http://localhost:8080/api/recognitions";

const CONFIDENCE_THRESHOLD = 35;
const HISTORY_LENGTH = 7;
const PREDICTION_INTERVAL = 250;

export default function Webcam() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const landmarkerRef = useRef(null);
  const streamRef = useRef(null);
  const animationRef = useRef(null);
  const lastRequestRef = useRef(0);
  const predictionHistoryRef = useRef([]);
  const requestInProgressRef = useRef(false);
  const aiServiceUnavailableRef = useRef(false);

  const [cameraOn, setCameraOn] = useState(false);
  const [prediction, setPrediction] = useState("-");
  const [confidence, setConfidence] = useState(0);
  const [topPredictions, setTopPredictions] = useState([]);
  const [status, setStatus] = useState("Camera is off");
  const [error, setError] = useState("");
  const [formedText, setFormedText] = useState("");
  const [aiServiceUnavailable, setAiServiceUnavailable] = useState(false);

  const [history, setHistory] = useState([]);
  const [saving, setSaving] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);

  function clearAIError() {
    aiServiceUnavailableRef.current = false;
    setAiServiceUnavailable(false);
  }

  function showAIError() {
    aiServiceUnavailableRef.current = true;
    setAiServiceUnavailable(true);
    setPrediction("-");
    setConfidence(0);
    setTopPredictions([]);
    predictionHistoryRef.current = [];
    setStatus("Python AI service unavailable");
    setError(
      "Cannot connect to the Python AI service. Start it at http://127.0.0.1:8000, then restart the camera."
    );
  }

  useEffect(() => {
    loadHistory();

    return () => {
      cancelAnimationFrame(animationRef.current);

      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }

      if (landmarkerRef.current) {
        landmarkerRef.current.close();
      }
    };
  }, []);

  async function loadHistory() {
    setHistoryLoading(true);

    try {
      const response = await fetch(HISTORY_API_URL);

      if (!response.ok) {
        throw new Error(`Backend returned HTTP ${response.status}`);
      }

      const data = await response.json();
      setHistory(Array.isArray(data) ? data : []);
    } catch (err) {
      console.error("Could not load recognition history:", err);
      setError(
        "Could not load history. Check that Spring Boot is running on port 8080."
      );
    } finally {
      setHistoryLoading(false);
    }
  }

  async function saveText() {
    const text = formedText.trim();

    if (!text) {
      setError("Form some text before saving it.");
      return;
    }

    setSaving(true);
    setError("");

    try {
      const response = await fetch(HISTORY_API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ recognizedText: text }),
      });

      if (!response.ok) {
        throw new Error(`Backend returned HTTP ${response.status}`);
      }

      const savedRecognition = await response.json();

      setHistory((previous) => [savedRecognition, ...previous]);
      setStatus("Text saved to recognition history.");
    } catch (err) {
      console.error("Could not save recognition:", err);
      setError(
        "Could not save text. Check that Spring Boot is running on port 8080."
      );
    } finally {
      setSaving(false);
    }
  }

  async function deleteHistoryItem(id) {
    const confirmed = window.confirm(
      "Delete this recognition from history?"
    );

    if (!confirmed) return;

    try {
      const response = await fetch(`${HISTORY_API_URL}/${id}`, {
        method: "DELETE",
      });

      if (!response.ok) {
        throw new Error(`Backend returned HTTP ${response.status}`);
      }

      setHistory((previous) =>
        previous.filter((item) => item.id !== id)
      );
    } catch (err) {
      console.error("Could not delete recognition:", err);
      setError("Could not delete this history item.");
    }
  }

  async function startCamera() {
    try {
      setError("");
      clearAIError();

      setPrediction("-");
      setConfidence(0);
      setTopPredictions([]);
      setStatus("Loading hand detector...");

      predictionHistoryRef.current = [];
      lastRequestRef.current = 0;
      requestInProgressRef.current = false;

      const vision = await FilesetResolver.forVisionTasks(WASM_URL);

      const landmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: MODEL_URL,
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        numHands: 1,
      });

      landmarkerRef.current = landmarker;

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 640 },
          height: { ideal: 480 },
        },
        audio: false,
      });

      streamRef.current = stream;

      const video = videoRef.current;
      video.srcObject = stream;
      await video.play();

      setCameraOn(true);
      setStatus("Camera is on. Show one hand.");

      animationRef.current = requestAnimationFrame(detectHand);
    } catch (err) {
      console.error("Camera setup failed:", err);

      setError(
        err.name === "NotAllowedError"
          ? "Camera permission was denied. Allow camera access in your browser."
          : `Could not start the camera: ${err.message}`
      );

      setStatus("Camera could not start");

      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }

      if (landmarkerRef.current) {
        landmarkerRef.current.close();
        landmarkerRef.current = null;
      }

      setCameraOn(false);
    }
  }

  function stopCamera() {
    cancelAnimationFrame(animationRef.current);
    animationRef.current = null;

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    if (canvasRef.current) {
      const context = canvasRef.current.getContext("2d");

      context.clearRect(
        0,
        0,
        canvasRef.current.width,
        canvasRef.current.height
      );
    }

    if (landmarkerRef.current) {
      landmarkerRef.current.close();
      landmarkerRef.current = null;
    }

    predictionHistoryRef.current = [];
    requestInProgressRef.current = false;
    lastRequestRef.current = 0;

    clearAIError();

    setCameraOn(false);
    setPrediction("-");
    setConfidence(0);
    setTopPredictions([]);
    setStatus("Camera is off");
    setError("");
  }

  async function requestPrediction(landmarks) {
    if (
      requestInProgressRef.current ||
      aiServiceUnavailableRef.current
    ) {
      return;
    }

    requestInProgressRef.current = true;

    try {
      const response = await fetch(AI_API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ landmarks }),
      });

      if (!response.ok) {
        throw new Error(`AI service returned HTTP ${response.status}`);
      }

      const data = await response.json();
      const rawLetter = String(data.prediction);
      const rawConfidence = Number(data.confidence);

      const alternatives = Array.isArray(data.top_predictions)
        ? data.top_predictions
        : [];

      if (
        !Number.isFinite(rawConfidence) ||
        !/^[A-Z]$/.test(rawLetter)
      ) {
        throw new Error("The AI service returned an invalid prediction.");
      }

      clearAIError();
      setTopPredictions(alternatives);

      console.log("Top 3 model predictions:", alternatives);
      console.log(
        "RAW prediction:",
        rawLetter,
        "| Confidence:",
        rawConfidence
      );

      if (rawConfidence < CONFIDENCE_THRESHOLD) {
        predictionHistoryRef.current = [];
        setPrediction("?");
        setConfidence(Number(rawConfidence.toFixed(2)));
        setStatus("Uncertain — adjust your hand and try again.");
        return;
      }

      const history = predictionHistoryRef.current;

      history.push({
        letter: rawLetter,
        confidence: rawConfidence,
      });

      if (history.length > HISTORY_LENGTH) {
        history.shift();
      }

      const counts = {};

      history.forEach((item) => {
        counts[item.letter] = (counts[item.letter] || 0) + 1;
      });

      const mostFrequentLetter = Object.keys(counts).reduce(
        (best, letter) =>
          counts[letter] > counts[best] ? letter : best
      );

      const matchingResults = history.filter(
        (item) => item.letter === mostFrequentLetter
      );

      const averageConfidence =
        matchingResults.reduce(
          (sum, item) => sum + item.confidence,
          0
        ) / matchingResults.length;

      setPrediction(mostFrequentLetter);
      setConfidence(Number(averageConfidence.toFixed(2)));
      setStatus("Hand detected");
    } catch (err) {
      console.error("Prediction request failed:", err);
      showAIError();
    } finally {
      requestInProgressRef.current = false;
    }
  }

  function detectHand() {
    const video = videoRef.current;
    const landmarker = landmarkerRef.current;
    const canvas = canvasRef.current;

    if (
      !video ||
      !landmarker ||
      !canvas ||
      video.readyState < 2 ||
      !video.videoWidth ||
      !video.videoHeight
    ) {
      animationRef.current = requestAnimationFrame(detectHand);
      return;
    }

    const context = canvas.getContext("2d");

    if (
      canvas.width !== video.videoWidth ||
      canvas.height !== video.videoHeight
    ) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    const result = landmarker.detectForVideo(video, performance.now());

    context.clearRect(0, 0, canvas.width, canvas.height);

    if (result.landmarks && result.landmarks.length > 0) {
      const hand = result.landmarks[0];

      context.fillStyle = "#45e6b0";

      hand.forEach((point) => {
        context.beginPath();
        context.arc(
          point.x * canvas.width,
          point.y * canvas.height,
          5,
          0,
          2 * Math.PI
        );
        context.fill();
      });

      const now = Date.now();

      if (
        !aiServiceUnavailableRef.current &&
        now - lastRequestRef.current >= PREDICTION_INTERVAL &&
        !requestInProgressRef.current
      ) {
        lastRequestRef.current = now;

        const landmarks = hand.flatMap((point) => [
          1 - point.x,
          point.y,
          point.z,
        ]);

        void requestPrediction(landmarks);
      }
    } else if (!aiServiceUnavailableRef.current) {
      predictionHistoryRef.current = [];
      setPrediction("-");
      setConfidence(0);
      setTopPredictions([]);
      setStatus("No hand detected. Show one hand clearly.");
    }

    animationRef.current = requestAnimationFrame(detectHand);
  }

  function addLetter() {
    if (!/^[A-Z]$/.test(prediction) || confidence < CONFIDENCE_THRESHOLD) {
      setError("Wait for a confident letter prediction before adding it.");
      return;
    }

    setFormedText((previous) => previous + prediction);
    setError("");
  }

  function addSpace() {
    setFormedText((previous) => {
      if (!previous || previous.endsWith(" ")) {
        return previous;
      }

      return previous + " ";
    });

    setError("");
  }

  function deleteLastCharacter() {
    setFormedText((previous) => previous.slice(0, -1));
    setError("");
  }

  function clearText() {
    setFormedText("");
    setError("");
  }

  async function copyText() {
    if (!formedText) {
      setError("There is no text to copy yet.");
      return;
    }

    try {
      await navigator.clipboard.writeText(formedText);
      setError("");
      setStatus("Text copied to clipboard.");
    } catch (err) {
      console.error("Copy failed:", err);
      setError("Could not copy text. Check your browser permissions.");
    }
  }

  return (
    <section className="webcam-panel">
      <div className="webcam-heading">
        <div>
          <h2>Live ASL Recognition</h2>
          <p>Position one hand clearly in front of your camera.</p>
        </div>

        <span className={`camera-status ${cameraOn ? "active" : ""}`}>
          {cameraOn ? "Camera on" : "Camera off"}
        </span>
      </div>

      <div className="webcam-view">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="webcam-video"
        />

        <canvas ref={canvasRef} className="webcam-canvas" />

        {!cameraOn && (
          <div className="webcam-placeholder">
            <span className="camera-symbol">+</span>
            <p>Your camera preview will appear here.</p>
          </div>
        )}
      </div>

      <p className="webcam-status">{status}</p>

      {error && (
        <p className="webcam-error" role="alert" aria-live="assertive">
          {error}
        </p>
      )}

      <div className="webcam-prediction">
        <div>
          <span className="prediction-label">Predicted letter</span>
          <strong>{prediction}</strong>
        </div>

        <div>
          <span className="prediction-label">Confidence</span>
          <strong>{confidence}%</strong>
        </div>
      </div>

      <div className="top-predictions">
        <p>Model alternatives</p>

        {topPredictions.length > 0 ? (
          <div className="top-predictions-list">
            {topPredictions.map((item) => (
              <span key={item.letter}>
                {item.letter}: {item.confidence}%
              </span>
            ))}
          </div>
        ) : (
          <p>Predictions will appear when a hand is detected.</p>
        )}
      </div>

      <div className="text-builder">
        <h3>Build Your Text</h3>
        <p>Add each recognized letter to form words and sentences.</p>

        <div className="formed-text" aria-live="polite">
          {formedText || "Your text will appear here..."}
        </div>

        <div className="text-builder-actions">
          <button
            onClick={addLetter}
            disabled={
              !/^[A-Z]$/.test(prediction) ||
              confidence < CONFIDENCE_THRESHOLD
            }
          >
            Add Letter ({prediction})
          </button>

          <button onClick={addSpace}>Space</button>

          <button
            onClick={deleteLastCharacter}
            disabled={!formedText}
          >
            Delete
          </button>

          <button onClick={clearText} disabled={!formedText}>
            Clear
          </button>

          <button onClick={copyText} disabled={!formedText}>
            Copy Text
          </button>

          <button
            onClick={saveText}
            disabled={!formedText.trim() || saving}
          >
            {saving ? "Saving..." : "Save Text"}
          </button>
        </div>
      </div>

      <div className="recognition-history">
        <div className="webcam-heading">
          <div>
            <h3>Recognition History</h3>
            <p>Your saved text from previous sessions.</p>
          </div>

          <button onClick={loadHistory} disabled={historyLoading}>
            {historyLoading ? "Loading..." : "Refresh"}
          </button>
        </div>

        {historyLoading && history.length === 0 ? (
          <p>Loading history...</p>
        ) : history.length === 0 ? (
          <p>No saved recognitions yet.</p>
        ) : (
          <div className="history-list">
            {history.map((item) => (
              <div className="history-item" key={item.id}>
                <div>
                  <strong>{item.recognizedText}</strong>
                  <p>
                    {item.createdAt
                      ? new Date(item.createdAt).toLocaleString()
                      : ""}
                  </p>
                </div>

                <button
                  onClick={() => deleteHistoryItem(item.id)}
                  aria-label={`Delete recognition ${item.id}`}
                >
                  Delete
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="webcam-actions">
        {!cameraOn ? (
          <button onClick={startCamera} className="webcam-button">
            Start camera
          </button>
        ) : (
          <button onClick={stopCamera} className="webcam-button stop">
            Stop camera
          </button>
        )}
      </div>
    </section>
  );
}
