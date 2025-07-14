# Written by Yonatan Morocz July 10th 2025. 

import tkinter as tk
import sys

from controllers.ui_controller import UIController
from models.video_session import VideoSession
from engine.video_engine import VideoEngine


def launch_app():
    """
    Entry point for the video-processing GUI application.
    Sets up the main window, initializes core components, and starts the wizard.
    """
    # Create the main application window
    root = tk.Tk()
    root.title("Flow Analysis Lab (FAB)")
    # Optionally, set a default size or make it full-screen
    root.geometry("1024x768")

    # Initialize core data model and engine
    session = VideoSession()
    engine = VideoEngine()

    # Instantiate the UI controller with dependencies
    app = UIController(master=root, session=session, engine=engine)

    # Start the Tkinter main loop
    root.mainloop()

    

if __name__ == "__main__":
    # Allow launching via `python main.py`
    try:
        launch_app()
    except Exception as e:
        print(f"Fatal error launching application: {e}", file=sys.stderr)
        sys.exit(1)
