# Fine-Tuning DistilBERT for Early Detection of Manufacturing Defects in Customer Reviews

## Project Summary

In large-scale manufacturing, identifying product defects early is critical to maintaining quality standards and preventing recalls. However, traditional quality monitoring relies on manual reviews or customer complaints, which often occur weeks after the product reaches the market. 

This project addresses this challenge by using Natural Language Processing (NLP) to detect manufacturing defects directly from customer reviews. The goal is to fine-tune the DistilBERT transformer model to automatically identify defect-related reviews, enabling manufacturers to act proactively—reducing costs, improving product reliability, and enhancing customer trust.

## Overview

This project fine-tunes DistilBERT, a compact transformer-based language model, to detect manufacturing defects from Amazon customer reviews. By identifying defect-related patterns in text, the model enables manufacturers to perform **early defect detection**—up to 4–6 weeks before manual review methods—potentially preventing costly recalls and safety issues.

## Project Objectives

- Fine-tune a pre-trained transformer model (DistilBERT) on real-world customer review data
- Accurately classify reviews as defect-related or non-defect-related
- Achieve high precision and recall while maintaining real-time inference speed (<100ms)
- Provide a reproducible workflow for dataset preparation, model training, evaluation, and analysis

## Repository Structure

```
.
├── LLM_Fine_Tuning_for_Early_Detection_of_Product_Defects_in_Customer_Reviews.ipynb    # Complete code and results
├── Report - Fine-Tuning DistilBERT.pdf                                                   # Technical project report
├── requirements.txt                                                                      # Python dependencies
├── README.md                                                                             # Project documentation
                                                                                
```
## 🎬 Watch Project Video
▶️ [Click here to watch the project demo](https://drive.google.com/file/d/1gNOWUPGUsAhlIsWVkeHRS8gOrp7PHB46/view?usp=drive_link)

## Methodology

### Dataset

- **Source:** Amazon Polarity Dataset
- **Sample Size Used:** Train 500, Validation 100, Test 100
- **Preprocessing Steps:**
  - Combined title and review text
  - Removed extra whitespace and special characters
  - Tokenized using DistilBERT tokenizer (max length = 128)
  - Converted to PyTorch tensors with padding and attention masks

### Model Configuration

- **Base Model:** `distilbert-base-uncased`
- **Framework:** PyTorch + Hugging Face Transformers
- **Architecture:** Binary classification head added to DistilBERT
- **Optimizer:** AdamW with weight decay
- **Loss Function:** Cross-Entropy
- **Metrics:** Accuracy, Precision, Recall, F1-score

### Hyperparameter Tuning

Three configurations were tested to find the optimal training setup:

| Configuration | Learning Rate | Batch Size | Weight Decay | F1 Score |
|--------------|---------------|------------|--------------|----------|
| Default | 5e-5 | 16 | 0.01 | 0.910 |
| High LR | 1e-4 | 16 | 0.01 | 0.910 |
| **Small Batch (Best)** | **5e-5** | **8** | **0.01** | **0.920** |

The small batch configuration achieved the best validation performance.

## Results

### Performance Metrics

| Metric | Baseline | Fine-Tuned | Improvement |
|--------|----------|------------|-------------|
| **Accuracy** | 51.0% | 94.0% | +84.3% |
| **Precision** | 26.0% | 94.1% | +261.7% |
| **Recall** | 51.0% | 94.0% | +84.3% |
| **F1 Score** | 34.5% | 94.0% | +172.8% |

### Performance

- **Inference Speed:** 298 reviews/second (~3ms per review)
- **Error Rate:** 6% (mainly mixed-sentiment and negation-related)

### Error Analysis

The model performed exceptionally well overall but faced challenges with:

- **Mixed Sentiment Reviews (83%)** – e.g., "Fun DVD but loaded with skips."
- **Negation Handling (33%)** – e.g., "Doesn't charge properly."
- **Sarcasm and Exaggeration (17%)** – e.g., "Best purchase ever—if it worked."

## Getting Started

### Clone the Repository

First, clone this repository to your local machine:

```bash
git clone https://github.com/tanv99/LLM-Fine-Tuning-for-Early-Detection-of-Product-Defects-in-Customer-Reviews.git
cd LLM-Fine-Tuning-for-Early-Detection-of-Product-Defects-in-Customer-Reviews
```

## Environment Setup

### Option 1: Google Colab (Recommended)

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File** → **Upload notebook**
3. Upload `LLM_Fine_Tuning_for_Early_Detection_of_Product_Defects_in_Customer_Reviews.ipynb` from the cloned repository
4. Change runtime type to GPU:
   - Click **Runtime** → **Change runtime type**
   - Select **GPU** (T4 recommended)
   - Click **Save**
5. Install dependencies by running this in a cell:
   ```bash
   !pip install transformers datasets torch scikit-learn pandas matplotlib seaborn
   ```
6. Execute all cells sequentially to reproduce results

### Option 2: VS Code Setup

1. Open VS Code
2. Install the **Jupyter** extension from the Extensions marketplace
3. Open the cloned repository folder:
   ```
   File → Open Folder → Select the cloned repository
   ```
4. Create a virtual environment:
   ```bash
   python -m venv llm_env
   ```
5. Activate the environment:
   - **Mac/Linux:** `source llm_env/bin/activate`
   - **Windows:** `llm_env\Scripts\activate`
6. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
7. Open `LLM_Fine_Tuning_for_Early_Detection_of_Product_Defects_in_Customer_Reviews.ipynb`
8. Select the kernel (top right) and choose the `llm_env` environment
9. Run all cells

### Option 3: Jupyter Notebook Setup

1. Create a virtual environment:
   ```bash
   python -m venv llm_env
   ```

2. Activate it:
   - **Mac/Linux:** `source llm_env/bin/activate`
   - **Windows:** `llm_env\Scripts\activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

5. Navigate to and open: `LLM_Fine_Tuning_for_Early_Detection_of_Product_Defects_in_Customer_Reviews.ipynb`
6. Run all cells sequentially

## Requirements

```txt
transformers==4.28.0
datasets==2.11.0
torch==1.13.1
scikit-learn==1.2.2
pandas==1.5.3
matplotlib==3.7.1
seaborn==0.12.2
numpy==1.24.3
```

## Reproducing Results

Run notebook cells in the following order:

1. **Data Preparation** – Load and preprocess dataset
2. **Training** – Fine-tune using all configurations
3. **Evaluation** – Compare baseline vs fine-tuned performance
4. **Error Analysis** – Analyze misclassified samples
5. **Inference** – Run sample predictions and benchmark performance

### Verification Checks

The following assertions confirm correct reproduction:

- ✅ F1 score >= 0.93
- ✅ 6 misclassified samples
- ✅ Best configuration: `small_batch`

## Future Enhancements

- Implement multi-aspect sentiment classification (battery, build, performance, etc.)
- Improve preprocessing to handle negation and mixed sentiment
- Apply data augmentation to improve generalization
- Scale to the full 3.6M review dataset
- Explore ensemble models for robustness

## Ethical Considerations

- **Bias:** Dataset skews toward electronics (60%), which may affect generalization
- **Privacy:** Reviews are anonymized and publicly available
- **Transparency:** Known limitations and potential biases are documented
- **Responsible Use:** Intended for product quality analysis, not censorship or manipulation

## References

- Sanh, V. et al. (2019). DistilBERT: A Distilled Version of BERT.
- Devlin, J. et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers.
- Howard, J. & Ruder, S. (2018). Universal Language Model Fine-Tuning for Text Classification.
- McAuley, J. et al. (2015). Amazon Product Review Dataset.
- Wolf, T. et al. (2020). Transformers: State-of-the-Art Natural Language Processing.

## Conclusion

The fine-tuned DistilBERT model achieved a **94% F1-score**, representing a **172.8% improvement** over the baseline while maintaining near real-time inference performance. This demonstrates that lightweight transformer models can effectively support early defect detection in manufacturing, offering both commercial and operational benefits.
