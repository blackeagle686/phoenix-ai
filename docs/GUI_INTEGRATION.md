#  Phoenix AI: Python GUI Integration Guide

This guide provides detailed instructions and professional patterns for connecting the Phoenix Autonomous Agent to various Python GUI frameworks. Building AI-powered desktop or web-based dashboards requires careful handling of **Asynchronicity** and **Threading** to ensure the user interface remains smooth and responsive while the agent "thinks."

---

## 🏗️ Core Concept: Threading & Async Bridges

The Phoenix AI SDK is natively asynchronous. Most GUI libraries (Tkinter, PyQt, Kivy) run on a **Main Thread** (Event Loop). If you run a heavy AI task directly on this thread, the GUI will freeze.

**The Golden Rule:** Always run the Agent in a separate thread and use a thread-safe bridge to update the UI.

---

## 1. Streamlit (Modern Web Dashboard) 🚀

Streamlit is the preferred choice for AI prototypes. It handles the web server and UI layout automatically.

### Singleton Pattern in Streamlit
Use `st.session_state` to ensure the agent is only initialized once per session.

```python
import streamlit as st
import asyncio
from phoenix import init_phoenix, startup_phoenix
from phoenix import Agent

st.set_page_config(page_title="Phoenix AI Dashboard", page_icon="🐦🔥")
st.title("🐦🔥 Phoenix AI Dashboard")

# 1. Thread-safe Initialization
if "agent" not in st.session_state:
    with st.spinner("🚀 Waking up Phoenix..."):
        init_phoenix()
        # Streamlit runs top-to-bottom; we use a temporary loop for startup
        loop = asyncio.new_event_loop()
        loop.run_until_complete(startup_phoenix())
        st.session_state.agent = Agent()
        st.session_state.messages = []

# 2. Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 3. Handle Input
if prompt := st.chat_input("Ask me anything..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner(" Reasoning..."):
            # Run async agent code in the current thread
            loop = asyncio.new_event_loop()
            response = loop.run_until_complete(st.session_state.agent.run(prompt))
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
```

---

## 2. PyQt6 / PySide6 (Professional Desktop) 

PyQt is the industry standard for cross-platform desktop apps. We use `QThread` and `pyqtSignal` to bridge the AI logic with the UI.

```python
import sys
import asyncio
from PyQt6.QtWidgets import QApplication, QMainWindow, QTextEdit, QLineEdit, QVBoxLayout, QWidget
from PyQt6.QtCore import QThread, pyqtSignal
from phoenix import init_phoenix, startup_phoenix
from phoenix import Agent

# Worker Thread for the Agent
class AgentWorker(QThread):
    finished = pyqtSignal(str)

    def __init__(self, agent, prompt):
        super().__init__()
        self.agent = agent
        self.prompt = prompt

    def run(self):
        # Create a new loop for the background thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(self.agent.run(self.prompt))
        self.finished.emit(response)

class PhoenixWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(" Phoenix Desktop")
        self.resize(600, 400)

        # UI Layout
        layout = QVBoxLayout()
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.input_field = QLineEdit()
        self.input_field.returnPressed.connect(self.send_query)

        layout.addWidget(self.chat_display)
        layout.addWidget(self.input_field)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Initialize Agent
        init_phoenix()
        # For simplicity, we run startup synchronously here
        asyncio.run(startup_phoenix())
        self.agent = Agent()

    def send_query(self):
        prompt = self.input_field.text()
        if not prompt: return
        
        self.chat_display.append(f"<b>You:</b> {prompt}")
        self.input_field.clear()
        self.input_field.setDisabled(True)

        # Start background worker
        self.worker = AgentWorker(self.agent, prompt)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def on_finished(self, response):
        self.chat_display.append(f"<b>Phoenix:</b> {response}<br>")
        self.input_field.setDisabled(False)
        self.input_field.setFocus()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PhoenixWindow()
    window.show()
    sys.exit(app.exec())
```

---

## 3. Tkinter (Built-in Python GUI) 

Tkinter is available in the standard library. We use `threading.Thread` and `queue.Queue` or `root.after()` for updates.

```python
import tkinter as tk
from tkinter import scrolledtext
import threading
import asyncio
from phoenix import init_phoenix, startup_phoenix
from phoenix import Agent

class TkinterAgentApp:
    def __init__(self, root):
        self.root = root
        self.root.title(" Phoenix Standard GUI")
        
        self.chat_window = scrolledtext.ScrolledText(root, state='disabled', width=60, height=20)
        self.chat_window.pack(padx=10, pady=10)
        
        self.entry = tk.Entry(root, width=50)
        self.entry.pack(side=tk.LEFT, padx=10, pady=10)
        self.entry.bind("<Return>", lambda e: self.process_input())
        
        self.send_btn = tk.Button(root, text="Send", command=self.process_input)
        self.send_btn.pack(side=tk.RIGHT, padx=10, pady=10)

        # Setup Agent in background
        threading.Thread(target=self.initialize_agent, daemon=True).start()

    def initialize_agent(self):
        init_phoenix()
        asyncio.run(startup_phoenix())
        self.agent = Agent()
        self.log_message("System", "Phoenix is online and ready.")

    def process_input(self):
        prompt = self.entry.get()
        if not prompt: return
        
        self.log_message("You", prompt)
        self.entry.delete(0, tk.END)
        self.entry.config(state='disabled')
        
        # Run agent in thread
        threading.Thread(target=self.run_agent_task, args=(prompt,), daemon=True).start()

    def run_agent_task(self, prompt):
        # We need a new event loop for this thread
        response = asyncio.run(self.agent.run(prompt))
        # Update UI from main thread using .after()
        self.root.after(0, lambda: self.log_message("Phoenix", response))
        self.root.after(0, lambda: self.entry.config(state='normal'))

    def log_message(self, sender, message):
        self.chat_window.config(state='normal')
        self.chat_window.insert(tk.END, f"{sender}: {message}\n\n")
        self.chat_window.config(state='disabled')
        self.chat_window.yview(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = TkinterAgentApp(root)
    root.mainloop()
```

---

## 4. CustomTkinter (Modern Desktop Look) ✨

`CustomTkinter` provides a macOS/Windows 11 modern aesthetic. The logic is identical to Tkinter but with premium visuals.

```python
import customtkinter as ctk
# (Logic remains the same as standard Tkinter example above)
# Just use ctk.CTk(), ctk.CTkButton(), etc.
```

---

## ❓ FAQ: Handling Concurrency

**Q: Why does my GUI freeze?**
A: You are likely calling `await agent.run()` or `asyncio.run()` directly inside a button click handler on the main thread. GUIs must never wait for network or AI tasks.

**Q: Can I stream responses to the GUI?**
A: Yes! You can pass a callback function to the Agent (if supported by your loop implementation) or use a thread-safe Queue to push tokens from the background thread to the UI thread.

**Q: How do I handle closing the app?**
A: Ensure your background threads are set to `daemon=True` so they exit when the main window closes, or use a cleanup hook to shut down the Phoenix services gracefully.
