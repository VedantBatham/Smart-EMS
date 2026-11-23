"""
Execution script to launch the FastAPI backend server and serve the interactive web dashboard.
"""

import os
import sys
import uvicorn
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    host = "127.0.0.1"
    port = 8000
    logger.info("======================================================================")
    logger.info("  AI-Driven Smart Energy Management System (Forecast-Assisted PPO)    ")
    logger.info("======================================================================")
    logger.info(f"Starting server on http://{host}:{port}")
    logger.info(f"API documentation: http://{host}:{port}/docs")
    logger.info(f"Web Dashboard:     http://{host}:{port}/index.html")
    logger.info("======================================================================")

    uvicorn.run("backend.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
