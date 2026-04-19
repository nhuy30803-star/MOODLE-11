import nltk
import spacy
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Pre-download NLTK data (will be handled in shell)
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

def nltk_preprocess(text):
    # 1. Input Text
    print(f"Original: {text}")
    
    # 2. Lowercasing
    text = text.lower()
    
    # 3. Tokenization
    tokens = word_tokenize(text)
    print(f"Tokenized: {tokens}")
    
    # 4. Punctuation Removal
    tokens = [t for t in tokens if t not in string.punctuation]
    
    # 5. Stopword Removal
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    print(f"Stopwords/Punct removed: {tokens}")
    
    # 6. Stemming
    stemmer = PorterStemmer()
    stems = [stemmer.stem(t) for t in tokens]
    print(f"Stemmed: {stems}")
    
    # 7. Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]
    print(f"Lemmatized: {lemmas}")
    
    return lemmas

def spacy_preprocess(text, nlp):
    # 1. Input Text
    doc = nlp(text)
    
    # 2. Lowercasing (handled via token.lower_)
    # 3. Tokenization
    # 4. Punctuation Removal
    # 5. Stopword Removal
    # 7. Lemmatization (spaCy doesn't have a default Stemmer, so we focus on Lemmas)
    
    tokens = []
    for token in doc:
        if not token.is_stop and not token.is_punct:
            tokens.append(token.lemma_.lower())
    
    print(f"spaCy processed: {tokens}")
    return tokens

def calculate_metrics(predicted, gold):
    # Simple evaluation logic for token lists
    # We treat it as a set of tokens for multi-label classification style metrics or 
    # exact sequence matching. For simple sentences, let's use exact match or overlap.
    
    pred_set = set(predicted)
    gold_set = set(gold)
    
    true_positives = len(pred_set.intersection(gold_set))
    false_positives = len(pred_set - gold_set)
    false_negatives = len(gold_set - pred_set)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = true_positives / len(gold_set) if len(gold_set) > 0 else 0 # Simple overlap accuracy
    
    return {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1}

def main():
    # 3 simple different sentences
    sentences = [
        "Natural Language Processing is a fascinating field of Artificial Intelligence.",
        "The quick brown fox jumps over the lazy dog.",
        "Data science involves extracting insights from various data sources."
    ]
    
    # Gold data (Expected lemmas after stopword/punct removal)
    # 1: [natural, language, processing, fascinate, field, artificial, intelligence]
    # 2: [quick, brown, fox, jump, lazy, dog]
    # 3: [data, science, involve, extract, insight, various, data, source] -> [data, science, involve, extract, insight, various, source]
    gold_data = [
        ["natural", "language", "processing", "fascinate", "field", "artificial", "intelligence"],
        ["quick", "brown", "fox", "jump", "lazy", "dog"],
        ["data", "science", "involve", "extract", "insight", "various", "source"]
    ]
    
    print("--- NLTK Results ---")
    nltk_results = []
    for s in sentences:
        nltk_results.append(nltk_preprocess(s))
        
    print("\n--- spaCy Results ---")
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        print("SpaCy model en_core_web_sm not found. Please install it.")
        return
        
    spacy_results = []
    for s in sentences:
        spacy_results.append(spacy_preprocess(s, nlp))
        
    # Evaluation
    print("\n--- Evaluation ---")
    
    for i in range(len(sentences)):
        print(f"\nSentence {i+1} Metrics:")
        n_metrics = calculate_metrics(nltk_results[i], gold_data[i])
        s_metrics = calculate_metrics(spacy_results[i], gold_data[i])
        
        print(f"NLTK:  {n_metrics}")
        print(f"spaCy: {s_metrics}")

if __name__ == "__main__":
    main()
