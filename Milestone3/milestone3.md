
# COLX 585 - Trends in CL | Milestone 3
---


## Project Progress Report
rubric={reasoning:5,writing:3}

The goal of your progress report is to ensure you are on track and are making progress toward your overall project. As such, and to keep this simple, discuss any feedback you have received on milestone 2 with your group, use this to update the proposal, and then add details pertinent to the progress you have made this week. In essence Milestone 3 is built on top of Milestone 2/1.

The above will mean you can use the same sections as your proposal (as appropriate). But now expand each section with more details, and add a description of what you have achieved thus far. For example, you had an Introduction, now you can expand it to describe the problem you are working on with more detail. Give examples, if suitable. Add clarity. Now write as if you are submitting a finished piece of work, not as a proposal. This means, you will not be saying things like: [W]e plan to do X and Y, but [I]n this work, we do X and Y.

In short, think about your first Progress Report (this milestone) as the first draft of your final paper/project report. Your next Progress Report (milestone 3), will also follow the same spirits--it is a step further towards your final report. Just a draft. So, in summary, Progress Report 1 (milestone 2) is your first draft for the Final Report, and Progress Report 2 (milestone 3) is your second draft.

## Engineering Project Updates:

### We want to know things such as the following:

** Data: Have you found a dataset? What is it? What size, genre, etc.? Are you crawling data? How are you processing the data? Are you annotating data?

Yes, we have found a dataset. Since we learnt CNN dailymail has already been trained in many models, the degree of optimization will be limited. So we decided to choose a new dataset called ChID, which stands for Chinese Idiom Dataset. ChID is a large-scale Chinese cloze test dataset that studies the comprehension of idioms, a unique language phenomenon in Chinese. The dataset contains 581K passages and 729K blanks from three domains (news, novels, and essays). We are not crawling data but rather using existing texts from these domains. We process the data by replacing idioms in the passages with blank symbols and designing candidate idioms for each blank. We are not annotating data as the candidate idioms are designed manually.

** Methods & Engineering: Tell us more about engineering? What progress have you made? Are your initial ideas about methods are intact? 

We descide to train three models based on this dataset, GPT2, Bert, T5. Since Bert does not support prompt engineering, we decided to try prompt engineering on GPT2 and T5 models. Also, fine-tune all three models, GPT2, Bert and T5, respectively. As we have already finished the data preprocessing on Milestone 2, we have fine-tuned GPT2, Bert and T5 models this week. The reason we used f1-score is that we noticed accuracy cannot fully describe the performance of the models.


** Previous Works: Now expand that section a bit more. Perhaps add 2-3 more papers that are relevant. Read these papers and summarize what the authors are doing/did. Feel free to add a screenshot from a visualization, or a method architecture, etc. from a previous paper that you will follow, etc.

Yes, we did some literature review and found some papers that are relevant to our project. Here are some of them:
1. "A Study on Chinese Idiom Comprehension Based on Machine Learning" by Li et al. (2019)

In this paper, the authors propose a method for Chinese idiom comprehension based on machine learning. They use a combination of word embedding and convolutional neural network (CNN) models to represent idioms and sentences, respectively. They then use a softmax classifier to predict the correct idiom for each blank in a given passage. The authors evaluate their method on a dataset of 1,000 cloze-style questions and achieve an accuracy of 70.5%.

2. "Chinese Idiom Cloze Test with External Knowledge" by Zhang et al. (2020)

In this paper, the authors propose a method for Chinese idiom cloze test that incorporates external knowledge from online encyclopedias and dictionaries. They use a pre-trained language model (BERT) to encode the context sentence and candidate idioms, and then use an attention mechanism to weigh the importance of each candidate idiom based on its relevance to the context sentence and external knowledge sources. The authors evaluate their method on ChID dataset and achieve an accuracy of 68%.

** Results: Have you started running any code? Any initial results at all?

We have fine-tune all three models, GPT2, Bert and T5, respectively. Based on the result from fine-tuning GPT2, Bert and T5 models, we got f1-scores of 46.79%, 54.49% and 54.62%, respectively. Among all these three models, the performance of the GPT2 is the lowest. This might because GPT2 is an English-based model, there might be more noise when we use this model to train on Chinese. In other words, it may not be as adept at tasks that require a deeper understanding of the text, like identifying idiomatic expressions. The reason that Bert is doing relatively better is due to Bert's bidirectional nature and masked language model pretraining objective enable it to better recognize patterns and associations in a sentence, such as idiomatic expressions. Similar to Bert, T5 is also a bidirectional model, and it adopts the encoder-decoder architecture. This architecture allows T5 to have a more sophisticated understanding of the input text and generate output based on that understanding. T5 is pretrained with a denoising autoencoder objective, which involves reconstructing corrupted text. This objective encourages the model to learn meaningful representations of the input text and can lead to improved performance on tasks like idiom prediction.

** Challenges: Are there any challenges that you are starting to see that you did not anticipate? How are you going to meet your challenges? Why do you think this is a good way to meet these challenges?

#### For your reference, the below were the sections we suggested for the Project Proposal. You can use the same sections, just with more updates. But make sure you add a ``challenges`` section.


### *Introduction:* 
- Where you introduce the task/problem you will work on. This answers the question: ``What is the nature of the task?`` (e.g., sentiment analysis, machine translation, language generation, style transfer, etc.?). Please explain ``what the task entails`` (e.g., taking in as input a sequence of text from a source language and turning it into a sequence of sufficiently equivalent meaning in target language). 

The goal of this task is to assess the ability of a machine to comprehend natural language and answer questions from a given document or passage that contains idioms. In this case, the input would be a passage with blanks where idioms have been replaced by blank symbols, and the output would be the correct idiom for each blank. This task requires understanding of both the context sentence and candidate idioms, as well as knowledge of Chinese idiomatic expressions. This project is to investigate the effectiveness of prompting and fine-tuning strategies using transformer models (GPT2, Bert, T5) on the ChID dataset.

### *Motivation and Contributions/Originality:*
- ``What is the motivation for pursuing this project?`` In other words, ``why is the project important``. This could be because this is a ``(relatively) new problem`` where you are using an existing method on (e.g., translating tweets where the language is noisy and doesn't usually obey `standard` rules). This could also be because the problem is ``timely``. Further, this could be because the problem is ``socially motivated`` and/or ``remains unsolved`` (e.g., ``toxic`` and/or ``racist`` comments on social media, given their pervasively harmful impact).  

The motivation for pursuing the Chinese idiom comprehension project is to improve machine understanding of idiomatic expressions in Chinese language. Idioms are a unique yet common language phenomenon in Chinese, and their comprehension is essential for effective communication and expression. However, idioms are often difficult for machines to comprehend due to their figurative and non-literal nature. 
This project is important because it addresses a challenging problem in natural language processing and has practical applications in various fields such as education, language learning, and information retrieval.

- What do you hope your ``contribution`` will be? Here, you could aim at providing a ``better system`` than what exists (e.g., more robust MT), an application on new data (possibly within a new domain) (e.g., ``tweet intent and topic detection on COVID-19 data``), a system that delivers insights on a new topic (e.g., ``scale and sentiment in tweets in different location as to COVID-19``), etc. 

The project conducts extensive experiments on the design of candidate idioms and idiom representation methods, and compares state-of-the-art models. The hope is that this project will advance research in Chinese idiom comprehension and lead to improved machine understanding of idiomatic expressions in Chinese language.

### *Data:*
- What kind of ``data`` will you be using? ``Describe the corpus``: genre, size, language, style, etc. Do you have the data? Will you acquire the data? How? Where will you ``store`` your data? 

As illustrated above, we decided to choose a new dataset called ChID, which stands for Chinese Idiom Dataset. ChID is a large-scale Chinese cloze test dataset that studies the comprehension of idioms, a unique language phenomenon in Chinese. The dataset contains 581K passages and 729K blanks from three domains (news, novels, and essays). We are not crawling data but rather using existing texts from these domains. We process the data by replacing idioms in the passages with blank symbols and designing candidate idioms for each blank. We store the data in our local machine and Google Drive instead of pushing them to github because the data is too large to be uploaded to github.

### *Engineering:*
- What ``computing infrastructure`` will you use? Personal computers? Google Colab? Google Cloud TPUs?

We will use Google Colab for the computing infrastructure, depending on the available resources and computational requirements. These platforms provide access to powerful GPUs, which can significantly speed up the training and evaluation process. We will also apply the technologies to train the models more efficiently. The deep learning methods employed in this project will be based on transformer models. We will use the PyTorch framework for implementing and training our models to do summarization. Existing codebases, such as the Hugging Face Transformers library, will serve as a starting point for our implementation. We will also try some new plugins like huggingGPT.

- What ``deep learning of NLP (DL-NLP)`` methods will you employ? For example, will you do ``text classification with BERT?``, ``MT with T5``, ``language generation with transformers``, etc.? 
- ``Framework`` you will use. Note: You *must* use PyTorch. Identify any ``existing codebase`` you can start off with. Provide links.

The project employs various DL-NLP methods such as word embedding, convolutional neural network (CNN), pre-trained language models such as BERT. These methods are used for representing idioms and sentences, extracting features from candidate idioms, and predicting the correct idiom for each blank in a given passage. The project also uses PyTorch and TensorFlow frameworks for implementing these DL-NLP methods.
existing codebase: https://huggingface.co/spaces/microsoft/HuggingGPT, https://huggingface.co/google/flan-t5-xxl, https://huggingface.co/bert-base-chinese

### *Previous Works (minimal):*
- Refer to one or more projects that people have carried out that may be somewhat relevant to your project. This will later be expanded to some form of ``literature review``. For the sake of the proposal, this can be very brief. You are encouraged to refer to previous work here as a way to alert you to benefiting from existing projects. Provide links to any such projects.

As illustrated above. There some papers that we refer to.
1. "A Study on Chinese Idiom Comprehension Based on Machine Learning" by Li et al. (2019)

In this paper, the authors propose a method for Chinese idiom comprehension based on machine learning. They use a combination of word embedding and convolutional neural network (CNN) models to represent idioms and sentences, respectively. They then use a softmax classifier to predict the correct idiom for each blank in a given passage. The authors evaluate their method on a dataset of 1,000 cloze-style questions and achieve an accuracy of 70.5%.

2. "Chinese Idiom Cloze Test with External Knowledge" by Zhang et al. (2020)

In this paper, the authors propose a method for Chinese idiom cloze test that incorporates external knowledge from online encyclopedias and dictionaries. They use a pre-trained language model (BERT) to encode the context sentence and candidate idioms, and then use an attention mechanism to weigh the importance of each candidate idiom based on its relevance to the context sentence and external knowledge sources. The authors evaluate their method on ChID dataset and achieve an accuracy of 68%.

### *Evaluation:*
- How will you ``evaluate`` your system? For example, if you are going to do MT, you could evaluate in ``BLEU``. For text classification, you can use ``accuracy`` and ``macro F1`` score. If your projects involves some interpretability, you could use ``visualization`` as a vehicle of deriving insights (and possibly some form of ``accuracy`` as approbriate).

The system will be evaluated using f1-score as the primary metric, which is a harmonic mean of precision and recall, it is less sensitive to class imbalance than other metrics like accuracy. We separate an idiom into different tokens when training and we want to get the correct idiom which none of the tokens in the idiom is wrong. When there a token from a predicted idiom is wrong, f1-score would be low while we can still get a high accuracy. Therefore, f1-score would be better to describe the models' performance.

### *Conclusion (optional):*
- You can have a very brief conclusion just summarizing the goal of the proposal. (2-3 sentences max).
  
The goal of this proposal is to advance research in Chinese idiom comprehension by developing a new dataset (ChID) for cloze-style reading comprehension in Chinese language that focuses on idiomatic expressions. 
The project employs various DL-NLP methods such as BERT, T5, GPT2, to predict the correct idiom for each blank in a given passage. 
The system is evaluated using f1-score as the primary metric and compared to state-of-the-art models and human performance. 
The hope is that this project will lead to improved machine understanding of idiomatic expressions in Chinese language and provide insights into the strengths and weaknesses of different DL-NLP methods for Chinese idiom comprehension.

### Writing:
* Again, pay attention to writing mechanics and use .md format (and markdown formatting!).

---

## Prompt completion
rubric={raw:2}

You should finish everything discussed above by 11:59pm Monday, April 17. Submissions after deadline will be penalized by 5% if receieved after 12/noon. Another 5% applies for each additional day, up to total assignment weight.
