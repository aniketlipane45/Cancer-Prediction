```markdown
# 🧠 Breast Cancer Prediction Web App

This project is a web-based machine learning application that predicts whether a breast tumor is **benign** or **malignant** using three trained models: **Random Forest**, **Support Vector Machine (SVM)**, and **XGBoost**.

The user can input 30 real-valued features from the breast cancer dataset and get an instant prediction through a Flask-powered web interface.



## 🚀 Demo

> ✅ Inputs: 30 numeric features  
> ✅ Output: Malignant or Benign  
> ✅ Choose model: Random Forest, SVM, or XGBoost

---

## 📁 Project Structure

<pre>
project/
├── app.py                    # Flask backend
├── train_and_save_models.py  # ML model training
├── data.csv                  # Dataset
├── xgboost.pkl               # Trained XGBoost model
├── random_forest.pkl         # Trained Random Forest model
├── svm.pkl                   # Trained SVM model
│
├── templates/
│   └── index.html            # Web form interface
│
└── static/
    └── style.css             # Optional styling
</pre>




## 📊 Dataset

This app uses the [Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)).

Features include:
- radius_mean, texture_mean, area_worst, smoothness_se, etc.
- 30 numerical inputs from digitized tumor image analysis.



## 🛠 How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/your-username/breast-cancer-predictor.git
cd breast-cancer-predictor
````

### 2. Install dependencies

```bash
pip install flask numpy pandas scikit-learn xgboost imbalanced-learn seaborn matplotlib
```

### 3. Train models (only once)

```bash
python train_and_save_models.py
```

### 4. Run the web app

```bash
python app.py
```

Open your browser and visit: `http://127.0.0.1:5000`



## 🧪 Sample Input

Example benign tumor input (manually fill in the form):

```
radius_mean: 14.12
texture_mean: 14.74
perimeter_mean: 89.43
...
fractal_dimension_worst: 0.075
```



## 📌 Technologies Used

* Python 3
* Flask
* Scikit-learn
* XGBoost
* Imbalanced-learn (SMOTE)
* HTML/CSS


## 🤝 Contributing

Pull requests and suggestions are welcome!



## 📜 License

This project is licensed under the MIT License.



## 🙏 Acknowledgments

* [UCI ML Repository](https://archive.ics.uci.edu/)
* [Scikit-learn](https://scikit-learn.org/)
* [Flask](https://flask.palletsprojects.com/)



