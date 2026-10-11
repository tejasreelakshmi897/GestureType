
import { useState } from "react";
import Webcam from "./Webcam";
import "./App.css";

function App() {
  const [activeSection, setActiveSection] = useState("recognition");

  function navigateTo(section) {
    setActiveSection(section);

    document.getElementById(section)?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a
          href="#overview"
          className="sidebar-brand"
          onClick={() => setActiveSection("overview")}
        >
          <span className="brand-icon">G</span>
          <span>
            <strong>GestureType</strong>
            <small>ASL Recognition Studio</small>
          </span>
        </a>

        <div className="sidebar-label">WORKSPACE</div>

        <nav className="sidebar-nav">
          <button
            className={activeSection === "overview" ? "nav-item active" : "nav-item"}
            onClick={() => navigateTo("overview")}
          >
            <span className="nav-icon">⌂</span>
            Overview
          </button>

          <button
            className={activeSection === "recognition" ? "nav-item active" : "nav-item"}
            onClick={() => navigateTo("recognition")}
          >
            <span className="nav-icon">◎</span>
            Live Recognition
          </button>

          <button
            className={activeSection === "how-it-works" ? "nav-item active" : "nav-item"}
            onClick={() => navigateTo("how-it-works")}
          >
            <span className="nav-icon">◇</span>
            How It Works
          </button>
        </nav>

        <div className="sidebar-bottom">
          <div className="sidebar-help-icon">?</div>
          <strong>GestureType</strong>
          <p>Making communication more accessible, one gesture at a time.</p>
          <span className="sidebar-version">ASL ALPHABET · ML POWERED</span>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar" id="overview">
          <div>
            <span className="topbar-eyebrow">YOUR WORKSPACE</span>
            <h1>ASL Recognition Studio</h1>
          </div>

          <div className="topbar-right">
            <span className="language-pill">
              <span className="language-dot" />
              ASL · English
            </span>
            <div className="profile-mark">GT</div>
          </div>
        </header>

        <section className="welcome-banner">
          <div>
            <span className="welcome-eyebrow">GESTURETYPE WORKSPACE</span>
            <h2>Every gesture tells a story.</h2>
            <p>
              Recognize ASL alphabet gestures in real time and turn
              them into text with your camera.
            </p>
          </div>

          <div className="welcome-art" aria-hidden="true">
            <div className="welcome-orbit orbit-one" />
            <div className="welcome-orbit orbit-two" />
            <div className="welcome-letter">A</div>
            <span className="welcome-spark spark-one">✦</span>
            <span className="welcome-spark spark-two">✧</span>
          </div>
        </section>

        <section className="recognition-area" id="recognition">
          <div className="section-title-row">
            <div>
              <span className="section-kicker">LIVE WORKSPACE</span>
              <h2>Gesture recognition</h2>
              <p>Start your camera, show a hand gesture, and build your text.</p>
            </div>

            <span className="section-status">
              <span />
              Real-time mode
            </span>
          </div>

          <div className="recognition-layout">
            <div className="recognition-main">
              <Webcam />
            </div>
          </div>
        </section>

        <section className="how-section" id="how-it-works">
          <div className="section-title-row">
            <div>
              <span className="section-kicker">THE PROCESS</span>
              <h2>How GestureType works</h2>
              <p>Three steps connect a hand gesture to readable text.</p>
            </div>
          </div>

          <div className="process-grid">
            <article className="process-card">
              <span className="process-number">01</span>
              <div className="process-icon">◉</div>
              <h3>Capture</h3>
              <p>
                The webcam captures your hand while MediaPipe tracks
                21 key hand landmarks.
              </p>
            </article>

            <article className="process-card">
              <span className="process-number">02</span>
              <div className="process-icon">⌘</div>
              <h3>Recognize</h3>
              <p>
                A trained machine-learning model uses the landmarks
                to predict an ASL alphabet letter.
              </p>
            </article>

            <article className="process-card">
              <span className="process-number">03</span>
              <div className="process-icon">Aa</div>
              <h3>Build text</h3>
              <p>
                Add recognized letters to create words and save them
                to your recognition history.
              </p>
            </article>
          </div>
        </section>

        <footer className="app-footer">
          <span>GestureType · ASL Recognition Project</span>
          <span>Building bridges through accessible technology.</span>
        </footer>
      </main>
    </div>
  );
}

export default App;
