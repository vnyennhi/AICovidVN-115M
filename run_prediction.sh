# -- Create and activate virtual environment
python -m venv env
source env/Scripts/activate

# -- Install dependencies
pip install -r requirements.txt

# -- Run train and predict on Private Test Set
python main.py
