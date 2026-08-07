# Local RAG Pipeline

A local Retrieval-Augmented Generation (RAG) pipeline for question answering over PDF documents.

The project combines document processing, text embeddings, semantic retrieval and LLM-based generation to produce answers grounded in retrieved document context.

## What It Does

The pipeline:

1. Processes a PDF document and extracts its text.
2. Splits the document into smaller text chunks.
3. Creates vector embeddings for the text chunks.
4. Performs semantic search to retrieve context relevant to a user query.
5. Passes the retrieved context to a language model.
6. Generates an answer based on the retrieved information.

## Experiments

The project also includes experiments with RAG configuration parameters:

- Number of retrieved context chunks (`n_resources_to_return`)
- LLM generation temperature
- Query formulation
- Answer relevance and consistency

Different configurations were compared to analyze how retrieval depth and generation temperature affect answer quality.

During the experiments, increasing the amount of retrieved context generally provided the model with more relevant information, while temperature affected the stability and variability of generated answers.

## Technologies

- Python
- Jupyter Notebook / Google Colab
- Retrieval-Augmented Generation
- Large Language Models
- Text Embeddings
- Semantic Search

## Project File

- [`RAG.ipynb`](./RAG.ipynb) – implementation, experiments and results

## Notes

This is an academic project focused on understanding and implementing the core components of a RAG system and evaluating how retrieval and generation parameters influence the final response.
