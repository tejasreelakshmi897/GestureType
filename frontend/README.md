# GestureType — ASL Hand Gesture Recognition

GestureType is a web-based application that recognizes American Sign Language (ASL) hand gestures through a webcam and converts them into text. It allows users to build words and sentences from recognized letters, copy the generated text, and save recognition history.

## Features

- **Real-Time Hand Detection:** Detects hand landmarks from webcam input using MediaPipe.
- **ASL Alphabet Recognition:** Uses a trained machine learning model to classify hand gestures.
- **Live Predictions:** Displays the predicted letter and confidence score.
- **Text Builder:** Add letters, insert spaces, delete characters, and clear the text.
- **Copy Text:** Copy the generated text for use elsewhere.
- **Recognition History:** Save recognized text and view previously saved entries with timestamps.
- **Database Persistence:** Stores recognition history in MySQL through the Spring Boot backend.
- **Interactive Dashboard:** Provides a React-based interface for recognition and history management.

## Technology Stack

| Component | Technologies |
|---|---|
| Frontend | React, Vite, JavaScript, CSS |
| AI Service | Python, FastAPI, MediaPipe, machine learning |
| Backend | Java, Spring Boot, Spring Data JPA |
| Database | MySQL |
| Communication | HTTP REST APIs |

## System Architecture

1. **Frontend:** React captures webcam input and displays predictions and text-building controls.
2. **AI Service:** Python processes hand landmarks and predicts the corresponding ASL letter.
3. **Backend:** Spring Boot exposes REST APIs to save, retrieve, and delete recognition history.
4. **Database:** MySQL stores saved text and timestamps.

The frontend communicates with the Python AI service for predictions and with the Spring Boot backend for history management.

## Project Structure

```text
GestureType/
├── ai-service/
│   ├── src/
│   ├── dataset/
│   ├── train_model.py
│   └── ...
├── backend/
│   └── src/
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── Webcam.jsx
│   │   └── index.css
│   └── package.json
└── README.md
```

*Note: The structure above is a simplified overview.*

## Prerequisites

- Python and pip
- Java Development Kit
- Node.js and npm
- MySQL Server
- Git (optional, for version control)

## Setup and Execution

### 1. Start the Python AI Service

Open a terminal:

```powershell
cd ai-service
.\venv\Scripts\Activate.ps1
python -m uvicorn src.main:app --reload --host 127.0.0.1 --port 8000
```

AI service: `http://127.0.0.1:8000`

### 2. Start the Spring Boot Backend

Open another terminal:

```powershell
cd backend
.\mvnw.cmd spring-boot:run
```

Backend: `http://localhost:8080`

Ensure MySQL is running and the database configuration in `application.properties` matches your local setup.

### 3. Start the React Frontend

Open a third terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite, usually `http://localhost:5173`.

Run all three services simultaneously to use the complete application.

## Recognition History API

The Spring Boot backend provides endpoints for managing saved recognition entries.

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/recognitions` | Save recognized text |
| GET | `/api/recognitions` | Retrieve all saved entries |
| GET | `/api/recognitions/{id}` | Retrieve an entry by ID |
| DELETE | `/api/recognitions/{id}` | Delete one entry |
| DELETE | `/api/recognitions` | Delete all entries |

Example request body:

```json
{
  "recognizedText": "HELLO"
}
```

## How to Use

1. Start the AI service, backend, and frontend.
2. Open GestureType in your browser.
3. Allow webcam access.
4. Show an ASL hand gesture to receive a predicted letter.
5. Use the text builder to construct words or sentences.
6. Copy the generated text or save it.
7. View or manage saved entries in Recognition History.

## Future Improvements

- Improve recognition accuracy across all ASL alphabet classes.
- Evaluate the model with more varied hand positions, lighting, and backgrounds.
- Expand support for continuous gesture recognition.
- Explore word-level and sentence-level sign language recognition.
- Improve accessibility and mobile responsiveness.

## Project Status

The core webcam recognition, text-building, copy, save, and recognition-history workflows have been tested.

GestureType is an educational prototype for ASL alphabet recognition. It does not currently translate the complete grammar and motion of ASL or replace a qualified interpreter.

## License

Add a license if you intend to distribute this project publicly. Ensure that datasets, model files, and third-party dependencies are used in accordance with their respective licenses.
