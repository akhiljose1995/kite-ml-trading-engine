import subprocess
import sys

def install(package):
    print(f"\n# Installing: {package}")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        #subprocess.check_call([sys.executable, "pip", "install", "pip-", package])
        print(f"# Successfully installed {package}")
    except subprocess.CalledProcessError as e:
        print(f"# Failed to install {package}: {e}")

if __name__ == "__main__":
    packages = [
        "kiteconnect",
        "pandas",
        "matplotlib",
        "pytest",
        "scipy",
        "numpy",
        "ta",  # Technical Analysis library
        "scikit-learn",  # For machine learning models
        "statsmodels",  # For statistical models
        "requests",  # For HTTP requests
        "tabulate",  # For pretty-printing tables
        "mpld3",  # For interactive matplotlib plots
        "plotly", # For interactive plots
        "xgboost",  # For XGBoost models
        "fastapi",  # For building APIs
        "uvicorn",  # ASGI server for FastAPI
        "joblib",  # For model serialization
        "git+https://github.com/kobaltgit/telegram_text_splitter.git",  # For splitting long messages for Telegram
    ]

    for pkg in packages:
        install(pkg)
