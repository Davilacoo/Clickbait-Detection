Clickbait Detection with Neural Networks
An end-to-end NLP project for classifying online news headlines as clickbait or non-clickbait. Built with PyTorch and scikit-learn, the project compares three neural architectures, analyses their predictions, and provides an interactive Streamlit application for real-time inference.
Best reported result: 97.81% accuracy · 0.9780 F1-score · 0.9972 ROC-AUC (ShallowMLP, 6,400-headline test split)
Tech stack: Python · PyTorch · scikit-learn · pandas · NumPy · Streamlit · Matplotlib

Overview
Clickbait headlines use curiosity gaps, sensational wording and other linguistic cues to encourage clicks. This project explores whether neural networks can identify those patterns from headline text alone.
The pipeline includes:
- Data preprocessing and stratified train/validation/test splitting.
- Feature engineering with TF-IDF (unigrams and bigrams) and 11 handcrafted linguistic features.
- Model development with a shallow MLP, a deeper MLP and a convolutional network for text classification (TextCNN).
- Model evaluation using accuracy, precision, recall, F1-score, ROC-AUC and confusion matrices, compared against a majority-class baseline.
- Interpretability and error analysis, including permutation importance for handcrafted features and inspection of misclassified examples.
- Interactive inference through a Streamlit web app and a command-line interface.
Results
The repository contains the following results from a stratified split of 32,000 headlines (23,040 training / 2,560 validation / 6,400 test):
Model	Accuracy	Precision	Recall	F1-score	ROC-AUC
Dummy baseline	50.00%	0.0000	0.0000	0.0000	0.5000
ShallowMLP	97.81%	0.9814	0.9747	0.9780	0.9972
DeepMLP	97.47%	0.9875	0.9616	0.9744	0.9962
TextCNN	96.73%	0.9728	0.9616	0.9672	0.9941


ShallowMLP achieved the highest F1-score in the recorded comparison, while DeepMLP achieved the highest precision. The results suggest that, on this dataset, a relatively simple feed-forward network with well-designed text features can outperform more complex architectures.
Confusion matrix	ROC curve
	


Evaluation note: The current pipeline selects its winning architecture using F1-score on the same test split shown above. These are the recorded benchmark results, not performance estimates from an untouched final test set after model selection. Selecting the model on validation data and evaluating once on a separate test set would make the comparison more rigorous.

Models and methodology
Architecture	Input representation	Network
ShallowMLP	Up to 10,000 TF-IDF features + 11 linguistic features	One 128-unit hidden layer, ReLU, dropout
DeepMLP	Same TF-IDF + linguistic features	256 → 128 → 64 hidden units, batch normalization, dropout
TextCNN	Tokenized and padded word sequences	Trainable embeddings, parallel 1D convolutions (kernel sizes 3/4/5), max pooling


The handcrafted features include headline length, word count, punctuation, capitalization, numbers, pronouns and clickbait-related keywords. TextCNN instead learns directly from token sequences without those manual inputs.
All neural models use PyTorch, binary cross-entropy with logits, the Adam optimizer and early stopping on validation loss. CUDA is used when available; a GPU is not required.
Interactive demo
The Streamlit app lets you enter a headline and see:
- A clickbait / non-clickbait prediction.
- The model's estimated clickbait score.
- A short, rule-based explanation based on linguistic signals.
- The extracted linguistic features for the submitted headline.
The explanations are descriptive heuristics, not faithful per-prediction attributions of the neural network. The model scores are not independently calibrated probabilities.
Quick start
1. Install dependencies (run these commands from the directory containing main.py):
python -m venv .venv
Activate the environment:
# macOS / Linux
source .venv/bin/activate

# Windows Git Bash (instead)
# source .venv/Scripts/activate

# Windows PowerShell (instead)
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
2. Launch the demo:
streamlit run app/streamlit_app.py
Open http://localhost:8501 in your browser. The repository already includes a serialized best-model bundle at models/best_model.joblib, so retraining is not required to try the app.
3. Predict from the terminal (alternative to the web interface):
python -m src.predict "You won't believe what happened next"
python -m src.predict "European Central Bank raises interest rates"
The command prints the predicted class, clickbait score, model name and rule-based explanation.
Retrain and generate reports
python main.py
This runs the training and evaluation pipeline, saves the selected model and metadata under models/, and generates comparison tables, plots, error examples and feature-importance outputs under reports/.
The dataset is already included in data/raw/clickbait_dataset.csv. To use a different compatible CSV, update TEXT_COL, LABEL_COL or DATA_PATH in src/config.py.
Dataset
The project uses the Chakraborty et al. (2016) clickbait headline corpus, containing 32,000 English-language headlines from a mixture of news and viral-content sources. Labels are binary: 1 = clickbait; 0 = non-clickbait.
The data is approximately balanced between classes. The repository includes the dataset for reproducibility; anyone redistributing or reusing it should check the original dataset's applicable terms.
Reference: Chakraborty, A., Paranjape, B., Kakarla, S., & Ganguly, N. (2016). Stop Clickbait: Detecting and Preventing Clickbaits in Online News Media. IEEE/ACM ASONAM.
Project structure
.
├── app/
│   └── streamlit_app.py         # Interactive classifier
├── data/
│   └── raw/
│       └── clickbait_dataset.csv
├── models/
│   ├── best_model.joblib        # Serialized model and preprocessing artifacts
│   └── metadata.json            # Recorded evaluation metrics
├── reports/
│   ├── figures/                # Comparison, confusion matrix, ROC, importance...
│   ├── error_analysis.csv
│   ├── model_comparison.csv
│   ├── permutation_importance.csv
│   └── final_report.md         # Detailed methodology and discussion
├── src/
│   ├── config.py               # Dataset paths and hyperparameters
│   ├── preprocessing.py        # Text cleaning
│   ├── features.py             # Handcrafted linguistic features
│   ├── tokenizer.py            # Tokenization for TextCNN
│   ├── models.py               # Neural architectures
│   ├── train_models.py         # Training and benchmarking
│   ├── evaluate.py             # Metrics, plots and error analysis
│   ├── predict.py              # Inference API and CLI
│   └── utils.py                # Data loading and utilities
├── clicikbait_report.pdf       # Academic project report
├── main.py                     # End-to-end training entry point
├── requirements.txt
└── README.md
Limitations and next steps
- Source bias: Models may learn stylistic differences between the publishers represented in the dataset, rather than an entirely general notion of clickbait.
- Model selection: Choose the architecture using validation performance and reserve a final, untouched test set for evaluation.
- External generalization: Evaluate on more recent headlines and sources not represented in training.
- Language coverage: The current dataset and linguistic features target English.
- Future extensions: Cross-source evaluation, cross-validation, calibrated predictions and transformer-based classifiers.
For more details, see the [technical report](reports/final_report.md) or the [PDF project report](clicikbait_report.pdf).
Academic NLP / neural networks project. Performance figures reflect the experiment artifacts included in this repository.
