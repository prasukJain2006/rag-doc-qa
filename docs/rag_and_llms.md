# RAG and Large Language Models

## What is RAG
Retrieval-Augmented Generation, or RAG, answers a question by first retrieving relevant passages from a document collection and then asking a language model to answer using only those passages. RAG grounds answers in your own documents, which reduces hallucinations compared with asking the model to answer from memory alone. It also lets answers stay current by updating the documents instead of retraining the model.

## Chunking
Long documents are split into chunks before indexing. A chunk size of around 500 characters with an overlap of 50 characters is a practical starting point for short notes. Chunks that are too large mix unrelated topics, while chunks that are too small lose context. Overlap helps keep sentences that cross a boundary intact.

## Embeddings and Vector Stores
An embedding turns text into a numeric vector so similar meanings have similar vectors. FAISS and ChromaDB are vector stores that index embeddings and return the nearest neighbours for a query. The retrieval step usually returns the top-k chunks, with k set to 3 to 5 for short answers. Cosine similarity is a common way to rank chunks.

## Grounded Prompting and Hallucinations
A grounded prompt tells the model to answer only from the retrieved context and to say "Not found in the documents" when the context does not contain the answer. Lowering temperature makes the output more deterministic and less creative, which suits factual question answering. Citations to the source chunk let a user verify the answer.

## RAG versus Fine-Tuning
Fine-tuning with LoRA updates a small set of model weights for a specific task or style. RAG is better when facts change often or when answers must cite sources. Fine-tuning is better when the goal is a consistent style, format, or behaviour rather than new facts. Many systems combine both: fine-tuning for behaviour and RAG for facts.

## Evaluating a RAG System
Retrieval accuracy, also called hit rate, measures how often the correct document appears in the top-k retrieved chunks. Answer accuracy requires human or model judgement of whether the final answer is correct. Latency is the time from question to answer, and cost is the price of embedding and language-model calls. Report all of these together, because a fast system that retrieves the wrong chunks is not useful.
