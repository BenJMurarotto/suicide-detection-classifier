Ben Murarotto 2025-09
## Suicide Message Flag Classifier
This project aimed to classify text documents on the basis of whether or not they pertained to self harm. A custom neural network was trained on a dataset that is a collection of posts from the "SuicideWatch" and "depression" subreddits of the Reddit platform. Each document was classified "suicide" or "non-suicide".
Metrics were assessed for 4 different embedding types; TF-IDF, TF-IDF with Bigrams, Bag of Words (BoW) as well as a custom BoW embedding method. Each embedding iteration was also compared to a Naive Bayes baseline model. The decision for building a custom ANN object for this task was twofold - I wanted to reinforce my understanding of underlying mathermatical concepts and to also practice using Pytorch library.

During training, random neuron dropout was implemented and set to 0.3 for all hidden layers across all model iterations. All embeddings were created from the top 100 features except the manual embedding, for which 500 was chosen. Classification threshold remained at 0.5 as exploratory analysis the of training set showed perfect class balance.
## Methods
- **Embeddings:**
    - **BoW:** Top 100 unigrams.
    - **TF‑IDF:** Top 100 unigrams, weighted by inverse document frequency.
    - **TF‑IDF + bigrams:** Top 100 features including unigrams and bigrams.
    - **Manual BoW:** Expanded vocabulary size (500 features).

- **Models:**
    - **BoW:** Single hidden layer (4 nodes). Larger models produced numerical instabilities (NaNs).
    - **TF‑IDF / TF‑IDF bigram:** Two hidden layers with widths 32 and 16.
    - **Manual BoW:** Two hidden layers with 16 nodes each.


- **Baseline:** Naive Bayes classifier trained on the same feature sets.
    
- **Training:** Binary cross‑entropy loss, dropout=0.3, validation split held out. Evaluation included both loss trends and classification metrics.
## Results
## Bag of Words
The Bag of Words model contained only with a single hidden layer of 4 nodes. During my testing, any larger models led to exploding activations but the smaller ANN reached a validation loss of approximately 0.45 by epoch 1000. The performance comparison showed that the ANN outperformed Naive Bayes when trained on BoW embeddings, achieving an accuracy of around 0.82 compared to Naive Bayes at 0.74. Precision, recall, and F1-scores for the ANN were approximately 0.83, 0.82, and 0.82 respectively, while Naive Bayes trailed with values near 0.76, 0.74, and 0.74.


<img width="600" height="400" alt="Pasted image 20250921121226" src="https://github.com/user-attachments/assets/a0659dca-d65e-4451-8bd0-7afd36f99d30" />
<img width="600" height="400" alt="Pasted image 20250921121239" src="https://github.com/user-attachments/assets/3481538e-5ef5-4744-bfd3-e99a09e39264" />


## TF-IDF
The TF-IDF embedding with unigrams proved more effective. The ANN with two hidden layers (32 and 16 neurons) trained smoothly and reduced validation loss to about 0.26 by the final epoch. The ANN consistently outperformed Naive Bayes, with accuracy, precision, recall, and F1-scores all near 0.84. In contrast, Naive Bayes achieved only about 0.76 across these metrics.


<img width="600" height="400" alt="Pasted image 20250921095908" src="https://github.com/user-attachments/assets/4595ef52-5b09-4fe8-a862-57ea2d27a5fc" />
<img width="600" height="400" alt="Pasted image 20250921121259" src="https://github.com/user-attachments/assets/1a73251b-1b97-4066-9cbb-4b6b7a7e78b8" />


## TF-IDF Bigram
The TF-IDF bigram embedding also produced strong results. The ANN achieved a validation loss of approximately 0.38 and maintained stable convergence. Accuracy, precision, recall, and F1-scores for the ANN were close to 0.83–0.84, while Naive Bayes once again lagged behind with metrics around 0.74 to 0.76. The addition of bigrams offered modest improvements in context handling, though the benefit was not dramatically higher than the unigram TF-IDF model.


<img width="600" height="400" alt="Pasted image 20250921113945" src="https://github.com/user-attachments/assets/2f8e6e08-cb02-4017-9e48-b36d632de591" />
<img width="600" height="400" alt="Pasted image 20250921114152" src="https://github.com/user-attachments/assets/52b8b9bd-2286-422a-a572-17ac9a3f37b8" />


## Manual Embedding
The manual BoW model with a vocabulary of 500 features demonstrated the strongest performance among all embeddings. Training reached a validation loss of approximately 0.37. The network achieved an accuracy of about 0.90, with precision, recall, and F1-scores each also around 0.90. This performance exceeded that of the Naive Bayes model, which reached roughly 0.87 across all metrics.


<img width="600" height="400" alt="Pasted image 20250921120255" src="https://github.com/user-attachments/assets/095c290d-50f0-459d-8cbe-03eb69411b2f" />
<img width="600" height="400" alt="Pasted image 20250921120324" src="https://github.com/user-attachments/assets/2094da5d-1d7c-45f3-8844-8c55d5270659" />


## Discussion
This project highlights how embedding choices influence a neural network performance for text classification. BoW embeddings, while simple, proved fragile when model capacity increased, indicating instability without normalisation or careful tuning. Applying TF-IDF with and without bigrams embeddings provided a clear performance boost in this case through the down weighting of frequent words. The manual BoW embedding, achieved the highest accuracy and F1-score due to the increase in vocabulary size, suggesting that richer representations and larger sample spaces are a crucial factor for discrimination even without simply relying on weighted embeddings.
The Naive Bayes baseline performed consistently well on all embeddings but was outperformed by the ANN in each case. It should be noted that having an increased feature space also boosted predictive performance of the Bayes model.  However, we have confirmed that neural models have the capacity to surpass them with larger feature sets.
