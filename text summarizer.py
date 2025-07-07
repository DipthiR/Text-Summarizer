import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

# ----------- Safe NLTK Downloads -----------
def safe_nltk_download(package):
    try:
        nltk.data.find(package)
    except LookupError:
        nltk.download(package.split("/")[-1])

# Ensure required NLTK resources are available
safe_nltk_download("tokenizers/punkt")
safe_nltk_download("corpora/stopwords")

# ----------- Text Summarization Functions -----------

def preprocess_sentence(sentence):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(sentence.lower())
    return [w for w in words if w.isalnum() and w not in stop_words]

def sentence_vectors(sentences):
    word_set = list({word for sent in sentences for word in preprocess_sentence(sent)})
    vectors = []
    for sentence in sentences:
        vec = [0] * len(word_set)
        word_freq = preprocess_sentence(sentence)
        if not word_freq:
            vectors.append(vec)
            continue
        for word in word_freq:
            if word in word_set:
                vec[word_set.index(word)] += 1
        vectors.append(vec)
    return vectors

def build_similarity_matrix(sentences):
    vectors = sentence_vectors(sentences)
    sim_matrix = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                try:
                    sim_matrix[i][j] = cosine_similarity([vectors[i]], [vectors[j]])[0, 0]
                except:
                    sim_matrix[i][j] = 0.0
    return sim_matrix

def textrank(sentences, top_n=3):
    if not sentences:
        return []
    sim_matrix = build_similarity_matrix(sentences)
    scores = np.sum(sim_matrix, axis=1)
    ranked_sentences = [sent for _, sent in sorted(zip(scores, sentences), reverse=True)]
    return ranked_sentences[:top_n]

def summarize_text(text, num_sentences=3):
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text
    summary = textrank(sentences, top_n=num_sentences)
    return ' '.join(summary)

# ----------- GUI Functions -----------

def load_file():
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if file_path:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                input_text.delete("1.0", tk.END)
                input_text.insert(tk.END, file.read())
        except Exception as e:
            messagebox.showerror("File Error", f"Could not load file:\n{e}")

def summarize():
    raw_text = input_text.get("1.0", tk.END).strip()
    try:
        num_sentences = int(summary_length.get())
        if num_sentences <= 0:
            raise ValueError
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter a valid positive number for summary length.")
        return

    if not raw_text:
        messagebox.showwarning("Empty Input", "Please enter or load text to summarize.")
        return

    try:
        summary = summarize_text(raw_text, num_sentences)
        if not summary.strip():
            raise ValueError("Summary could not be generated. Try with different or more text.")
        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, summary)
    except Exception as e:
        messagebox.showerror("Summarization Error", f"Error during summarization:\n{e}")

# ----------- GUI Layout -----------

window = tk.Tk()
window.title("Text Summarization Tool (Offline)")
window.geometry("800x600")
window.configure(bg="#f0f0f0")

# Input Label
tk.Label(window, text="Input Text:", font=("Arial", 12, "bold"), bg="#f0f0f0").pack(pady=(10, 0))
input_text = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=95, height=10)
input_text.pack(padx=10, pady=5)

# Summary Length Input
frame = tk.Frame(window, bg="#f0f0f0")
frame.pack(pady=5)
tk.Label(frame, text="Summary Sentences:", font=("Arial", 10), bg="#f0f0f0").pack(side=tk.LEFT)
summary_length = tk.Entry(frame, width=5)
summary_length.insert(0, "3")
summary_length.pack(side=tk.LEFT, padx=5)

# Buttons
button_frame = tk.Frame(window, bg="#f0f0f0")
button_frame.pack(pady=10)
tk.Button(button_frame, text="Load Text File", command=load_file).pack(side=tk.LEFT, padx=10)
tk.Button(button_frame, text="Summarize", command=summarize).pack(side=tk.LEFT, padx=10)

# Output Text
tk.Label(window, text="Summary:", font=("Arial", 12, "bold"), bg="#f0f0f0").pack(pady=(10, 0))
output_text = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=95, height=10, bg="#e8f5e9")
output_text.pack(padx=10, pady=5)

# Start GUI
window.mainloop()
