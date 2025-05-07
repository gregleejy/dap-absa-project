# dap-absa-project
This ABSA project uses the Reddit Mental Health dataset to detect sentiment tied to specific mental health aspects like therapy, medication, and relationships.

# 🧠 Aspect-Based Sentiment Analysis (ABSA) on Reddit Mental Health Posts

This project applies **Aspect-Based Sentiment Analysis (ABSA)** to mental health discussions from Reddit. Using the **Reddit Mental Health dataset**, we aim to identify both the **aspects** discussed (e.g., therapy, relationships, medication) and the **sentiment** associated with each aspect (positive, negative, neutral). To enhance data quality, we used the **OpenAI API** to generate synthetic, structured examples for better model training and evaluation.

---

## 📌 Objectives

- 🔍 Perform **Aspect Extraction** from Reddit mental health posts.
- 🎯 Classify **Sentiment Polarity** for each identified aspect.
- 🤖 Improve data quality with **synthetic data augmentation** using the OpenAI API.
- 🧪 Evaluate model performance on synthetic data.

---

## 🧾 Dataset

We used the **Reddit Mental Health dataset**, which includes posts from users discussing various aspects of their mental health. The dataset was enhanced through:

- **Data Preprocessing**: Cleaning, filtering, and tokenizing raw posts.
- **Synthetic Data Augmentation**: Using OpenAI’s GPT model to generate structured ABSA-style training samples.

### Files:
- `Combined_Train_MTL.csv` – Training data with augmented labels.
- `Combined_Test_MTL.csv` – Test data with original and synthetic samples.

---

## 🚀 How It Works

### 1. Preprocessing

- Remove noise, tokenize text, and identify mental health-related aspects.
- Use GPT (OpenAI API) to reformat posts into structured ABSA-friendly formats.

### 2. Training

- Fine-tune a custom ABSA model using **multitask learning**, where both aspect extraction and sentiment classification are learned together.

### 3. Evaluation

- Evaluate the model on the original and synthetic test sets.
- Metrics include precision, recall, F1-score for both aspect and sentiment predictions.

---

## 📊 Aspects Analyzed

| Aspect        | Example Sentiment |
|---------------|------------------|
| Therapy       | "My therapist really helps me stay grounded." (Positive) |
| Medication    | "The meds just make me feel numb." (Negative) |
| Relationships | "I feel like my friends don’t understand me." (Negative) |
| Work/School   | "Work has been more manageable lately." (Positive) |
| Family        | "My family support keeps me going." (Positive) |

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.8+
- pip
- OpenAI API Key

### Installation

```bash
git clone https://github.com/yourusername/absa-reddit-mental-health.git
cd absa-reddit-mental-health
pip install -r requirements.txt
---

### 🔑 Set Up OpenAI API Key

To use the OpenAI API for generating synthetic data, you’ll need to set up your API key.

1. Sign up at [OpenAI](https://platform.openai.com) and generate an API key.  
2. Create a `.env` file in the root directory of the project.  
3. Add your key to the `.env` file like this:

```
```
OPENAI_API_KEY=your_openai_api_key_here
```

4. **Important**: Do not commit your `.env` file. Make sure `.env` is included in your `.gitignore` file to keep your key safe.

---

## 📚 Future Work

- Expand dataset with additional subreddits or mental health platforms  
- Experiment with transformer-based models (e.g., BERT, RoBERTa)  
- Deploy as a web service for real-time sentiment monitoring  

---

## 👨‍💻 Author

**Darrius Ng**  
GitHub: [@darriusnjh](https://github.com/darriusnjh)

**Greg Lee**  
GitHub: [@gregleejy](https://github.com/gregleejy)

---

## 📬 Contact

For questions or collaboration opportunities, feel free to reach out via GitHub @darriusnjh or @gregleejy

---

## 📄 License

This project is licensed under the **MIT License**.  
See the `LICENSE` file for more information.
