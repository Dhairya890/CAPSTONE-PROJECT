# TF Binding Sites Prediction

## Project Overview
This project focuses on predicting transcription factor (TF) binding sites using DNA sequence data. The dataset is sourced from **ENCODE**, and the goal is to classify DNA sequences into binding or non-binding sites. The project employs deep learning, traditional machine learning models, and state-of-the-art techniques to compare their effectiveness.

## Methodology
1. **Data Preprocessing**
   - One-hot encoding or k-mer encoding of DNA sequences.
   
2. **Model Development**
   - Custom **CNN model** optimized using **Keras Tuner**.
   - Comparison with **traditional ML models** (e.g., Random Forest, SVM, Logistic Regression).
   - Evaluation against **state-of-the-art models** for TF binding site prediction.
   
3. **Deployment**
   - Hosting the best-performing model on **Streamlit** for real-time predictions.

## Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/tf-binding-prediction.git
cd tf-binding-prediction
pip install -r requirements.txt
```

## Usage
Train and evaluate models:

```bash
python train.py
```

Run the Streamlit app:

```bash
streamlit run app.py
```

## Features
- Supports **multiple encoding schemes** (One-hot, k-mer)
- Custom **CNN architecture** optimized with **Keras Tuner**
- **Traditional ML models** for comparison
- **State-of-the-art model evaluation**
- **Streamlit web interface** for easy interaction

## Roadmap
- [ ] Fine-tune CNN architecture
- [ ] Experiment with additional encoding techniques
- [ ] Enhance Streamlit UI with visualizations
- [ ] Deploy on cloud (e.g., Hugging Face Spaces, AWS, GCP)

## Contributing
Feel free to submit issues or pull requests.

## License
This project is licensed under the MIT License.

## Contact
For questions, reach out at **your.email@example.com**.
