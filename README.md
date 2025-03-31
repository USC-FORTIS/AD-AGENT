# OpenAD

This repository provides a unified script to run anomaly detection algorithms on `.mat` datasets. It supports both single-algorithm and parallel execution modes, and includes example commands for testing.

---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/USC-FORTIS/OpenAD.git
cd OpenAD
```

### 2. Create and Activate a Virtual Environment

#### On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

#### On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cpu.html
```

### 4. Set Your OpenAI API Key

Edit the config file to include your OpenAI API key:

```python
# File: /config/config.py

OPENAI_API_KEY = 'your-api-key-here'
```

---

## 🚀 Running the Program

### Run Normally (Sequential Execution)

```bash
python main.py
```

### Run in Parallel Mode

```bash
python main.py -p
```

---

## 🧪 Test Commands

You can also run the system with natural-language-like test commands.

### Run a Specific Algorithm

```text
#pyod
Run IForest on ./data/glass_train.mat and ./data/glass_test.mat with contamination=0.1
#pygod
Run DOMINANT on  ./data/inj_cora_train.pt and ./data/inj_cora_test.pt
```

### Run All Algorithms

```text
Run all on ./data/glass_train.mat and ./data/glass_test.mat
```

---

## 📁 Project Structure

```
.
├── config/
│   └── config.py             # Configuration file for API keys
.
.
.
├── data/
│   └── glass.mat             # Sample dataset
├── main.py                   # Main execution script
├── requirements.txt          # Required Python packages
└── README.md                 # Project documentation
```

---

## 📌 Notes

- Make sure your dataset is placed inside the `./data/` directory.
- Modify `main.py` to add support for additional algorithms or datasets if needed.


