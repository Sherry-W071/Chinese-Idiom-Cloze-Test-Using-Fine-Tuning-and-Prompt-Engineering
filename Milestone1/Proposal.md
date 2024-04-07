# Project proposal
### *Title:*
A Comparative Study of Prompting and Fine-tuning in Transformer Models for CNN/Daily Mail Text Summarization

### *Introduction:* 
The goal of this project is to investigate the effectiveness of prompting and fine-tuning strategies for the extractive text summarization task using transformer models (fine-tuning model will be chosen from BERT, GPT-2, T5, RoBERTa, ELECTRA, mBERT, ALBERT; prompt engineering LLMs will be chosen from (GPT-2, ELECTRA, OPT / BLOOM / Claude / Bard / ChatGPT)). Extractive text summarization involves generating concise summaries of input text documents by selecting a subset of the most important sentences, preserving the main ideas while reducing redundancy.

### *Data:*
We will be using the CNN/Daily Mail Dataset, a widely used dataset for text summarization tasks. The dataset consists of news articles from CNN and Daily Mail, along with their respective human-generated summaries, which serve as ground truth. The dataset contains over 300,000 training examples and is in English. The data is publicly available and can be downloaded from [github cnn-dailymail repository]( https://github.com/abisee/cnn-dailymail). The data will be stored locally or on a cloud storage service, depending on the chosen computing infrastructure.

### *Engineering:*
We will use Google Colab for the computing infrastructure, depending on the available resources and computational requirements. These platforms provide access to powerful GPUs, which can significantly speed up the training and evaluation process. We will also apply the technologies to train the models more efficiently.
The deep learning methods employed in this project will be based on transformer models. We will use the PyTorch framework for implementing and training our models to do summarization. 
Existing codebases, such as the [Hugging Face Transformers library](https://github.com/huggingface/transformers), will serve as a starting point for our implementation.

### *Previous Works:*
Several works have been done on text summarization using transformer models. For example, [Liu et al. (2019)](https://arxiv.org/abs/1908.08345) proposed BERTSUM, a BERT-based approach for extractive summarization. BERTSUM extends the original BERT model by adding a summarization-specific architecture consisting of an interval segment layer and a document-level encoder. The interval segment layer helps the model capture sentence-level information, while the document-level encoder enhances the model's ability to understand the overall document structure. BERTSUM demonstrated state-of-the-art performance on the CNN/Daily Mail dataset for extractive summarization, achieving substantial improvements over previous methods.

[Raffel et al. (2019)](https://arxiv.org/abs/1910.10683) introduced T5, a transformer model that was pre-trained on a large-scale multi-task learning setup, using the "Text-to-Text Transfer Transformer" (T5) framework. T5 frames various NLP tasks as a text-to-text problem, converting inputs and outputs into a unified format. By pre-training the model on multiple tasks simultaneously, T5 was able to achieve state-of-the-art performance on a wide range of benchmarks, including summarization, translation, and question-answering. The success of T5 highlights the potential of using pre-trained models in combination with task-specific fine-tuning or prompting strategies for extractive text summarization.

These works will serve as a basis for our project and help us understand the nuances of implementing transformer models for text summarization tasks. In our project, we aim to build upon the foundations laid by these studies to investigate the comparative effectiveness of prompting and fine-tuning strategies for extractive text summarization using transformer models like BERT and T5 and other additional models like GPT-2, ELECTRA, etc..


### *Evaluation:*
We will evaluate the performance of our models using the ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metric, which is a widely used evaluation metric for text summarization tasks. ROUGE measures the overlap between the generated summaries and the reference summaries (ground truth). We will report ROUGE-1, ROUGE-2, and ROUGE-L scores to provide a comprehensive evaluation of the models’ performance.
In addition, we will use cosine similarity to measure the similarity between the vector representations of the generated and reference summaries. We will convert the summaries into a bag-of-words representation, create vector space models using techniques such as TF-IDF, and compute the cosine similarity between the vectors. This metric provides a different perspective on the quality of the generated summary compared to ROUGE scores, as it focuses on the similarity of the vector representations rather than the exact word overlaps.

### *Conclusion:*
In this project, we aim to investigate and compare prompting and fine-tuning strategies for extractive text summarization using transformer models. By the end of this project, we will have a clear understanding of the comparative strengths and weaknesses of these strategies, which can be used to guide future research and practical applications in the field of NLP.

