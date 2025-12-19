# Book Recommender RAG

A semantic book recommendation system built with Retrieval-Augmented Generation (RAG) that uses vector embeddings to find books similar to user queries. The system includes sentiment analysis and category filtering capabilities, all accessible through an interactive Gradio dashboard.

## Features

- **Semantic Search**: Uses Google Generative AI embeddings to find books based on semantic similarity to user queries
- **Category Filtering**: Filter recommendations by book categories
- **Emotional Tone Filtering**: Sort recommendations by emotional tones (Happy, Surprising, Angry, Suspenseful, Sad)
- **Interactive Dashboard**: Beautiful Gradio interface for easy interaction
- **Vector Database**: Uses ChromaDB for efficient vector storage and retrieval
- **Data Analysis**: Comprehensive notebooks for data exploration, vector search, text classification, and sentiment analysis

## Project Structure

```
BookRecommenderRAG/
├── 1.data-exploration.ipynb          # Data exploration and analysis
├── 2.vector-search.ipynb             # Vector search implementation
├── 3.text-classification.ipynb       # Text classification tasks
├── 4.sentiment-analysis.ipynb        # Sentiment analysis implementation
├── 5.gradio-dashboard.py             # Main Gradio dashboard application
├── books.csv                         # Original book dataset
├── books_cleaned.csv                 # Cleaned book data
├── books_with_categories.csv         # Books with category labels
├── books_with_emotions.csv           # Books with emotion scores
├── tagged_description.txt            # Processed book descriptions for vectorization
├── books_chroma_db/                  # ChromaDB vector database
├── cover-not-found.jpg               # Default cover image
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Prerequisites

- Python 3.8 or higher
- Google Generative AI API key (for embeddings)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd BookRecommenderRAG
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
   - Create a `.env` file in the project root
   - Add your Google Generative AI API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

## Usage

### Running the Dashboard

Launch the Gradio dashboard:
```bash
python 5.gradio-dashboard.py
```

The dashboard will open in your browser, typically at `http://127.0.0.1:7860`.

### Using the Dashboard

1. **Enter a book description**: Type a description of the kind of book you're looking for (e.g., "A story about forgiveness")
2. **Select a category** (optional): Choose a book category to filter results, or select "All" for all categories
3. **Select an emotional tone** (optional): Choose an emotional tone to prioritize, or select "All" for no tone preference
4. **Click "Find recommendations"**: The system will return up to 16 book recommendations with covers and descriptions

### How It Works

1. **Vector Search**: Your query is converted to an embedding using Google Generative AI's text-embedding-004 model
2. **Similarity Matching**: The system searches the ChromaDB vector database for the 50 most similar book descriptions
3. **Filtering**: Results are filtered by category (if specified)
4. **Tone Sorting**: Results are sorted by emotional tone scores (if specified)
5. **Display**: Top 16 recommendations are displayed with book covers and descriptions

## Notebooks

The project includes several Jupyter notebooks for different aspects of the system:

- **1.data-exploration.ipynb**: Exploratory data analysis of the book dataset
- **2.vector-search.ipynb**: Implementation of vector search using ChromaDB
- **3.text-classification.ipynb**: Text classification experiments
- **4.sentiment-analysis.ipynb**: Sentiment analysis to extract emotional tones from book descriptions

## Technologies Used

- **LangChain**: Framework for building applications with LLMs
- **ChromaDB**: Vector database for storing and querying embeddings
- **Google Generative AI**: Embedding model (text-embedding-004)
- **Gradio**: Interactive web interface
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Transformers**: NLP models for sentiment analysis
- **Matplotlib/Seaborn**: Data visualization

## Dependencies

See `requirements.txt` for the complete list of dependencies. Key packages include:

- pandas >= 2.0.0
- numpy >= 1.24.0
- langchain-community >= 0.2.0
- langchain-google-genai >= 1.0.0
- langchain-chroma >= 0.1.0
- chromadb >= 0.4.0
- gradio >= 4.0.0
- transformers >= 4.30.0
- torch >= 2.0.0

## Notes

- The ChromaDB database is automatically created on first run if it doesn't exist
- Book covers are fetched from Google Books API (via thumbnail URLs)
- If a cover image is not available, a default "cover-not-found.jpg" is used
- The system uses semantic similarity, so queries don't need to match exact keywords


## Author
https://github.com/Ayush-Repositories