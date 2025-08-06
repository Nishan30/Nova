# 🎥 Live Demo

# Watch the full 3-minute video demo showcasing Nova's key features in action:

# \[Watch on YouTube](\[https://youtu.be/MbzcBR7XoZs])

# 

# ✨ Core Philosophy: Privacy, Sovereignty, and Companionship

# In an era of cloud-centric AI, Nova represents a deliberate architectural shift towards personal, user-owned intelligence. The project was founded on a non-negotiable principle: the user's data must never leave their machine. This philosophy dictated every technical choice, from the local-only model stack to the native application framework, ensuring absolute data privacy and user sovereignty.

# Nova is not just a chatbot; it is a true cognitive partner, designed to understand context, recall memories, perceive the user's emotional state, and act on their behalf within their own digital environment.

# 🚀 Key Features

# 🧠 True Multimodal Understanding: Nova sees, hears, and reads. Powered by a local Gemma 3n model, she can analyze images and charts, transcribe voice commands with Whisper, and hold deep, context-aware conversations.

# 🗣️ Hands-Free Voice Control: Using a custom "Hey Nova" wake word powered by the lightweight pvporcupine engine, you can interact with Nova completely hands-free, without ever touching your keyboard.

# 🕵️‍♂️ Desktop Agent Capabilities: Nova is more than a passive assistant. She can act on your behalf:

# Search Local Files: "Hey Nova, find my report from last week."

# Open Applications \& Files: "Open that report and launch Spotify."

# Set Reminders: "Remind me to drink water in 5 minutes."

# 📓 Deep Memory \& Reflection: Nova builds a rich understanding of you over time.

# The Journal: A private space for your thoughts. Nova uses sentiment analysis to understand your emotional state and can answer reflective questions like, "What's been triggering my stress lately?" based on your own entries.

# RAG Knowledge Base: Upload your own documents (PDFs, TXT) to create a personal "second brain." Nova's answers can be grounded in your own data, ensuring accuracy and relevance.

# 🖥️ Native Desktop Experience: A polished, frameless side-panel built with PySide6 (Qt) that can be instantly summoned or dismissed with a global hotkey (Ctrl+Shift+X).

# 🛠️ Technology Stack \& Architecture

# Nova employs a robust, multi-threaded architecture to ensure a responsive UI while leveraging a powerful, local AI stack.

# Component	Technology Used	Purpose

# Frontend UI	PySide6 (Qt for Python)	Native desktop experience, responsive UI, frameless side-panel.

# Backend \& Task Mgmt	Python, QThread / QObject Signal \& Slot	Orchestrates all operations, ensures non-blocking UI.

# Core AI Model	Gemma 3n (Unsloth GGUF Quantization)	Multimodal reasoning, vision, agentic decision-making.

# AI Serving	Ollama	Manages and serves the local LLM via a stable API.

# Voice Transcription	OpenAI Whisper (local base model)	Fast, high-accuracy speech-to-text.

# Wake Word Engine	pvporcupine	Low-CPU, always-on listening for "Hey Nova".

# Declarative Memory	ChromaDB + all-MiniLM-L6-v2	RAG system for user documents.

# Episodic/Emotional Memory	SQLite + VADER Sentiment	Stores chat history, journal entries, and emotional tags.

# ⚙️ Setup \& Installation

# Follow these steps to get Nova running on your local machine.

# 1\. Prerequisites:

# Python 3.10+

# Ollama: Make sure the Ollama server is installed and running. Pull the Gemma 3n model:

# ollama pull hf.co/unsloth/gemma-3n-E4B-it-GGUF:UD-Q4\_K\_XL

# FFmpeg: Required by Whisper for audio processing. (Install with choco install ffmpeg on Windows or sudo apt install ffmpeg on Linux).

# Tesseract OCR: Required for image-to-text from attachments. (Download and install from the official Tesseract repository).

# 2\. Clone the Repository:

# Generated bash

# git clone https://github.com/Nishan30/Nova

# 

# 3\. Set Up Virtual Environment:

# python -m venv venv

# 

# \# Activate it

# \# On Windows:

# venv\\Scripts\\activate

# \# On macOS/Linux:

# source venv/bin/activate

# 4\. Install Dependencies:

# pip install -r requirements.txt

# (Note: Ensure you have created a requirements.txt file with pip freeze > requirements.txt)

# 5\. Configure Wake Word:

# Sign up for a free account at Picovoice Console to get your AccessKey.

# Paste your key into the access\_key variable in the start\_wake\_word\_listener function in desktop\_app.py.

# Place your custom .ppn file (e.g., Hey-Nova...ppn) in the wakewords folder.

# 6\. Run the Application:

# python desktop\_app.py

# 💡 How to Use

# Summon/Dismiss: Press Ctrl+Shift+X at any time.

# Voice Commands: Start your command with "Hey Nova".

# Chat: Type messages, attach files, and press Enter.

# 🏆 Challenges \& Solutions

# The Misleading \[WinError 2] Bug: A persistent error during Whisper transcription was traced not to a missing audio file, but a missing ffmpeg.exe in the script's runtime PATH. Solution: Programmatically add the known ffmpeg directory to the system PATH at application startup, making the app self-contained.

# UI Freezing \& "Orphaned Thread" Crashes: Early versions would crash on consecutive, rapid requests. Solution: Implemented a robust worker management system using a list (self.active\_workers) to hold strong references to every running QThread, preventing the Python garbage collector from destroying them prematurely.

# 🔮 Future Development

# Expanded Agent Toolset: Integrate tools for web search, calculator functions, and calendar management.

# Proactive Insights: Implement a weekly summary feature where Nova reflects on your journal and chat patterns.

# Deeper Long-Term Memory: Automatically extract and store key entities (people, places, projects) from conversations to build a richer memory graph.

# 📜 License

# This project is licensed under the MIT License - see the LICENSE.md file for details.

