from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# converts the sequences into k-mer strings
def sequence_to_kmers(sequence, k=6):
    return " ".join([sequence[i:i+k] for i in range(len(sequence)-k+1)])

sequences = []
labels = []  # 1 = dangerous, 0 = not

kmer_sequences = [sequence_to_kmers(s) for s in sequences]

# builds the feature matrix
vectorizer = CountVectorizer(analyzer='word')
X = vectorizer.fit_transform(kmer_sequences)

# train and test split
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, 
    test_size=0.2, 
    random_state=42
)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

print(classification_report(y_test, clf.predict(X_test)))