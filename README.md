# captcha

A reusable, modular CAPTCHA solving toolkit designed for integration into automation systems (e.g., Twitter bots).

 Repository Structure

├── src/
│   ├── captcha_solver/
│   │   ├── core/
│   │   ├── preprocessing/
│   │   ├── solvers/
│   │   ├── utils/
│   │   └── __main__.py
│   └── main.py          # Optional CLI entrypoint
│
├── tests/
│   ├── captcha_test_suite.py
│   ├── test_real_captcha.py
│   └── test_runner.py   # other test scripts
│
├── .env.example
├── requirements.txt
└── README.md

Setup

Clone the repo

git clone https://github.com/your-org/custom-captcha-solver.git
cd custom-captcha-solver

Create & activate a virtual environment

python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate    # Windows

Install Python dependencies

pip install -r requirements.txt

Install Tesseract OCR

Windows: Download and install from https://github.com/UB-Mannheim/tesseract/wiki

macOS: brew install tesseract

Linux: sudo apt-get install tesseract-ocr

Configure environment variables

cp .env.example .env
# Edit .env:
# CAPTCHA_IMAGE_PATH=path/to/local/sample.png
# CAPTCHA_IMAGE_URL=https://...

Running Tests

All test scripts live in the project root or tests/:

# Run full suite
python tests/captcha_test_suite.py

# Or run individual tests
python tests/test_real_captcha.py
python tests/test_runner.py

Watch the console for success rates and detailed logs.

Usage Example

from captcha_solver.core.orchestrator import CaptchaOrchestrator
from captcha_solver.models.captcha_types import CaptchaType

orc = CaptchaOrchestrator(config={})

# For a text CAPTCHA image:
result = orc.solve_captcha(
    target="path/to/captcha.png",
    detected_type=CaptchaType.TEXT
)
print(result.solution, result.success)

 Integration Options

API: wrap CaptchaOrchestrator behind Flask/FastAPI endpoints

Browser: call solvers from a Playwright/Puppeteer script or Chrome extension

CLI: use main.py to solve from command line

