# Semantica

This project uses sentence embeddings to visualize and analyze the relationship between different sentences in a 2D or 3D space. It supports two main dimensionality reduction techniques: PCA (Principal Component Analysis) for 3D visualization and t-SNE (t-Distributed Stochastic Neighbor Embedding) for 2D visualization.

## Folder Structure

    ├── README.md
    ├── requirements.txt
    └── src
        ├── app.py
        └── utils.py

## Features

- **Generate Sentence Embeddings**: Using pre-trained models from [SentenceTransformers](https://www.sbert.net/).
- **Visualize Embeddings**: Visualize the relationship between sentences using 3D PCA or 2D t-SNE.
- **Clustering**: Optionally apply KMeans clustering to group similar sentences.
- **Export Results**: Download the projection results as a CSV file.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/sentence-embedding-visualization.git
    cd sentence-embedding-visualization
    ```

2. Create a virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate 
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the app locally, use the following command:

```bash
streamlit run app.py