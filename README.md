# Context-Based-Text-Generation-using-LSTMs
Text generation using artificial intelligence has made strides, but maintaining context remains a challenge. This project introduces two Long Short-Term Memory (LSTM) models enriched with semantic context for improved coherence. The models, trained on a "Lord of the Rings" dataset, include a base LSTM and a context-aware LSTM leveraging word embeddings and K-Means clustering.
## Environment
This project was developed using:
-Python 3.7
-PyTorch 1.9
-NumPy 1.21
-NLTK 3.6
-Gensim 4.1
-Scikit-learn 0.24
-Matplotlib 3.4

## Dataset

The project utilizes the "Lord of the Rings" dataset, sourced from kaggle. This dataset comprises extensive text from the renowned literary work by J.R.R. Tolkien. For convenience and resource management, a subset of 10,000 sentences from the original dataset was used.

![pos](https://github.com/poornikabonam/Context-Based-Text-Generation-using-LSTMs/assets/97566249/5fc299ea-1895-4fde-b3b6-31e23656204c)

## Methodology
1. **Data Preprocessing:**
        Text is preprocessed to remove punctuation, normalize case, and tokenized using NLTK's tools.
        Vocabulary and word-to-index mappings are created for LSTM input.
2. **Base LSTM:**
        Utilizes word embeddings to capture sequential dependencies within the text.

3. **Context-Aware LSTM:**
        Enhances the base LSTM by incorporating semantic context via K-Means clustering of word embeddings.
        Visualization of Word Embeddings clusters using t-SNE for better comprehension of semantic relationships.

4. **Training and Evaluation:**
        Both models trained over 30 epochs, monitored for cosine similarity, loss, and accuracy.
        Validation conducted to compare the performance between base and context-aware LSTM.

## Results
    Cosine Similarity: Measures the semantic similarity of generated text.
    Loss and Accuracy: Tracks training and validation performance metrics.
    
![accuracy](https://github.com/poornikabonam/Context-Based-Text-Generation-using-LSTMs/assets/97566249/be283ad6-661d-4888-abb6-b646f0b0f3bc)

![cluster](https://github.com/poornikabonam/Context-Based-Text-Generation-using-LSTMs/assets/97566249/0ee710c0-1736-48d1-a55d-ddba1f14fd2f)

![loss](https://github.com/poornikabonam/Context-Based-Text-Generation-using-LSTMs/assets/97566249/0d9f3a6e-6c73-4994-a9de-735674bdb395)


