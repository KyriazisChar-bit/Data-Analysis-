# 🧠 Assignment 5 — Neural Networks: MNIST Digit Classification

A standalone deep-learning exercise: building, training, and comparing two neural-network architectures on the MNIST handwritten-digit dataset — a simple fully-connected network versus a convolutional network.

**Institution:** Aristotle University of Thessaloniki (School of Mechanical Engineering, AUTh)
**Course:** Data Analysis — 8th Semester
**Author:** Kyriazis Charitopoulos 

---

## ⚙️ Tech Stack

- Python 3
- TensorFlow / Keras
- pandas, NumPy
- Matplotlib, seaborn

---

## 🏗️ What It Does

**Dataset & Preprocessing**
Loads MNIST (28×28 grayscale digits, 0–9) from CSV using **AEM = 7137** as the seed. Subsamples to a 54,000-image train set (90%) and 8,000-image test set (80%), checks class balance, normalises pixels to [0, 1], and flattens images to 784-vectors for the dense model.

**Section A — Simple Neural Network**
A fully-connected `Sequential` model (784 → 256 → 128 → 10) with ReLU activations and a Softmax output, trained with Adam over 30 epochs. Evaluated on accuracy, loss curves, and a confusion matrix — exposing overfitting and characteristic digit confusions (e.g. 9 ↔ 4).

**Section B — Convolutional Neural Network**
A CNN (Conv2D → MaxPooling → Conv2D → Flatten → Dense → Softmax) trained with identical settings for a fair comparison, exploiting the 2D spatial structure the dense model ignores.

**Comparison**
Side-by-side validation curves, confusion matrices, and a metrics table covering accuracy, loss, parameter count, and overfitting behaviour.

---

## 📁 Key Files

| File | Description |
|------|-------------|
| `report.pdf` / `.docx` / `.tex` | Full methodology, architectures, curves, confusion matrices, conclusions |
| `labels.csv` | MNIST data source (place alongside the script) |

---

## 🚀 Running

```bash
pip install tensorflow pandas numpy matplotlib seaborn
python <your_script>.py
```

---

## 📊 Key Results

| Metric | Simple NN | CNN |
|--------|-----------|-----|
| Test Accuracy | 98.47% | **99.21%** |
| Test Loss | 0.0896 | **0.0425** |
| Trainable Parameters | 850,586 | **515,146** |

The CNN won on every axis — **higher accuracy with 39.4% fewer parameters** and far less overfitting, thanks to weight sharing and translation invariance. Confirms that convolutional architectures are the natural fit for image classification.

---

*Academic coursework — Aristotle University of Thessaloniki. Not for resubmission in other academic contexts.*
