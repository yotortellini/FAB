# Written by Yonatan Morocz July 10th 2025. 

import sys
import os
import logging
from pathlib import Path
from PyQt5.QtWidgets import QApplication

# Suppress NSOpenPanel warning on macOS
if sys.platform == 'darwin':
    os.environ['QT_MAC_WANTS_LAYER'] = '1'

from models.video_session import VideoSession
from engine.video_engine import VideoEngine
from controllers.ui_controller import UIController

# Set up logging
def setup_logging():
    """Configure application-wide logging"""
    log_path = Path(__file__).parent / 'logs'
    log_path.mkdir(exist_ok=True)
    log_file = log_path / 'fab.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def launch_app():
    logger.info("Starting Flow Analysis Lab application")
    try:
        app = QApplication(sys.argv)
        
        # Set application-wide stylesheet for consistent look
        app.setStyle('Fusion')
        logger.debug("Applied Fusion style")
        
        # Initialize models
        session = VideoSession()
        engine = VideoEngine()
        logger.debug("Initialized session and engine")
        
        # Load application stylesheet
        style_path = Path(__file__).parent / "assets" / "style.qss"
        if style_path.exists():
            with open(style_path, 'r') as f:
                app.setStyleSheet(f.read())
            logger.debug(f"Loaded stylesheet from {style_path}")
        else:
            logger.warning(f"Stylesheet not found at {style_path}")
        
        # Create and show main window
        window = UIController(session=session, engine=engine)
        window.show()
        logger.info("Main window displayed")
        
        return app.exec_()
    except Exception as e:
        logger.exception("Failed to launch application")
        raise

if __name__ == "__main__":
    try:
        sys.exit(launch_app())
    except Exception as e:
        logger.critical(f"Fatal error launching application: {e}")
        sys.exit(1)
