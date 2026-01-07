# 📊 AnalystFlow-AI

**AnalystFlow-AI** is an end-to-end **automated data analysis and preprocessing web application** built using **Streamlit**.
It helps users **understand, clean, preprocess, visualize, validate, and export datasets**—all through an intuitive UI—without writing code.

This tool is especially useful for:

* Data Science students
* ML beginners
* Analysts preparing datasets for machine learning
* Hackathons & academic projects

---

## 🚀 Features Overview

### 🔹 Step 1: Dataset Upload & Understanding

* Upload **CSV or Excel** files
* Automatic dataset profiling:

  * Preview
  * Shape (rows & columns)
  * Data types
  * Missing values
  * Duplicate rows
  * Statistical summary

---

### 🔹 Step 2: Missing Value Handling

Choose the **best practice** strategy:

* ✅ Replace missing values

  * **Numerical → Median**
  * **Categorical → Mode**
* ❌ Drop rows with missing values
* Undo support for safe experimentation

---

### 🔹 Step 3: Advanced Data Cleaning

* **Duplicate row removal**
* **Outlier treatment using IQR method**

  * Applied individually to all numerical columns
  * Caps outliers instead of deleting rows

---

### 🔹 Step 4: Data Preprocessing

#### 📐 Feature Scaling

* Standardization (Z-score)
* Normalization (Min-Max)

#### 🔤 Categorical Encoding

* Label Encoding
* One-Hot Encoding

---

### 🔹 Step 5: Visual Validation (Before vs After)

Visual comparison using:

* 📦 Boxplots
* 📍 Scatter plots
* 🔥 Correlation heatmaps
* 🔗 Feature vs Target correlation analysis

> Raw data is kept **immutable**, ensuring unbiased comparison.

---

### 🔹 Step 6: Export & Data Readiness Report

* Export cleaned dataset as:

  * CSV
  * Excel
* Rule-based **data readiness summary**
* Complete operation history log

---

## 🤖 AI-Generated Data Readiness Explanation (Optional)

AnalystFlow-AI follows a **deterministic-first, AI-second architecture**.

### 🔘 When AI is ENABLED

* Groq LLM generates a **professional data readiness report**
* Uses:

  * Raw dataset statistics
  * Cleaned dataset statistics
  * Actual preprocessing steps performed
* Output explains:

  1. How data quality improved
  2. Why each preprocessing step is best practice
  3. Why the dataset is ML-ready

### 🔘 When AI is DISABLED

* No API calls are made
* Rule-based validation alone confirms dataset readiness

### 🧠 Why This Design Matters

* Prevents unnecessary API costs
* Avoids AI dependency for correctness
* Ensures reproducibility & reliability
* Graceful fallback if AI is unavailable

> AI is used **only for explanation**, not decision-making.

## 🧠 Why AnalystFlow-AI?

✔ Prevents data leakage
✔ Enforces ML best practices
✔ Beginner-friendly yet professional
✔ Undo support for safe exploration
✔ Visualization-driven validation
✔ Optional AI explanation

---

## 🛠️ Tech Stack

| Component        | Technology          |
| ---------------- | ------------------- |
| Frontend         | Streamlit           |
| Backend Logic    | Python              |
| Data Handling    | Pandas, NumPy       |
| Visualization    | Matplotlib, Seaborn |
| ML Preprocessing | Scikit-learn        |
| AI Integration   | Groq LLM (Optional) |
| Environment      | Python-dotenv       |

---

## 📁 Project Structure

```
AnalystFlow-AI/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (Groq API key)
├── README.md              # Project documentation
└── sample_datasets/       # (Optional) Example datasets
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/AnalystFlow-AI.git
cd AnalystFlow-AI
```

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate     # macOS/Linux
venv\Scripts\activate        # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Optional: Enable AI Explanation (Groq)

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_groq_api_key_here
```

> ⚠️ If not provided, the app still works perfectly using rule-based validation.

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

Then open:

```
http://localhost:8501
```

---

## 📊 Example Use Cases

* ML dataset preprocessing before training
* Academic mini & major projects
* Hackathons
* Resume projects
* Teaching data cleaning concepts visually

---

## 🧪 ML Best Practices Followed

* Immutable raw dataset
* Separate working dataset
* Median & Mode for imputation
* IQR-based outlier handling
* Scaling before distance-based models
* Visual validation before export

---

## 🧩 Future Enhancements

* Feature selection
* Train-test split
* Model training & evaluation
* Automated EDA reports
* Dataset versioning
* SHAP-based explainability

---

## 👨‍💻 Author

**Harshavardhan Nadiveedi**
B.Tech AIML Student
Focused on **Data Science, ML & AI Projects**

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and feel free to fork & extend it!



