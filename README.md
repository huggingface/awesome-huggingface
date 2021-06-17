<p align="center"> 
<img src="https://raw.githubusercontent.com/huggingface/awesome-huggingface/main/logo.svg?token=AFLYUK4HQBJT734TLKYP2R3A2CKW2" width="100px">
</p>

# awesome-huggingface
🤗 Hugging Face Ecosystem: A list of wonderful open-source projects based on Hugging Face libraries.

[How to contribute](https://github.com/huggingface/awesome-huggingface/blob/main/CONTRIBUTING.md)

## 🤗 Official Libraries
*First-party cool stuff made with ❤️ by 🤗 Hugging Face.*
* [transformers](https://github.com/huggingface/transformers) - State-of-the-art natural language processing for Jax, PyTorch and TensorFlow.
* [datasets](https://github.com/huggingface/datasets) - The largest hub of ready-to-use NLP datasets for ML models with fast, easy-to-use and efficient data manipulation tools.
* [tokenizers](https://github.com/huggingface/tokenizers) - Fast state-of-the-Art tokenizers optimized for research and production.
* [knockknock](https://github.com/huggingface/knockknock) - Get notified when your training ends with only two additional lines of code.
* [accelerate](https://github.com/huggingface/accelerate) - A simple way to train and use PyTorch models with multi-GPU, TPU, mixed-precision.
* [autonlp](https://github.com/huggingface/autonlp) - Train state-of-the-art natural language processing models and deploy them in a scalable environment automatically.
* [nn_pruning](https://github.com/huggingface/nn_pruning) - Prune a model while finetuning or training.
* [huggingface_hub](https://github.com/huggingface/huggingface_hub) - Client library to download and publish models and other files on the huggingface.co hub.

## 👩‍🏫 Tutorials
*Learn how to use Hugging Face toolkits, step-by-step.*
* [Official Course](https://huggingface.co/course) (from Hugging Face) - The official course series provided by 🤗 Hugging Face.
* [transformers-tutorials](https://github.com/nielsrogge/transformers-tutorials) (by @nielsrogge) - Tutorials for applying multiple models on real-world datasets.

## 🧰 NLP Toolkits
*NLP toolkits built upon Transformers. Swiss Army!*
* [AllenNLP](https://github.com/allenai/allennlp) (from AI2) - An open-source NLP research library.
* [Graph4NLP](https://github.com/graph4ai/graph4nlp) - Enabling easy use of Graph Neural Networks for NLP.
* [Lightning Transformers](https://github.com/PyTorchLightning/lightning-transformers) - Transformers with PyTorch Lightning interface.

## 🥡 Text Representation
*Converting a sentence to a vector.*
* [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) (from UKPLab) - Widely used encoders computing dense vector representations for sentences, paragraphs, and images.
* [WhiteningBERT](https://github.com/Jun-jie-Huang/WhiteningBERT) (from Microsoft) - An easy unsupervised sentence embedding approach with whitening.
* [SimCSE](https://github.com/princeton-nlp/SimCSE) (from Princeton) - State-of-the-art sentence embedding with contrastive learning.
* [DensePhrases](https://github.com/princeton-nlp/DensePhrases) (from Princeton) - Learning dense representations of phrases at scale.

## ⚙️ Inference Engines
*Highly optimized inference engines implementing Transformers-compatible APIs.*

* [TurboTransformers](https://github.com/Tencent/TurboTransformers) (from Tencent) - An inference engine for transformers with fast C++ API.
* [FasterTransformer](https://github.com/NVIDIA/FasterTransformer) (from Nvidia) - A script and recipe to run the highly optimized transformer-based encoder and decoder component on NVIDIA GPUs.
* [lightseq](https://github.com/bytedance/lightseq) (from ByteDance) - A high performance inference library for sequence processing and generation implemented in CUDA.
* [FastSeq](https://github.com/microsoft/fastseq) (from Microsoft) - Efficient implementation of popular sequence models (e.g., Bart, ProphetNet) for text generation, summarization, translation tasks etc.

## 🏎️ Model Compression/Acceleration
*Compressing or accelerate models for improved inference speed.*
* [torchdistill](https://github.com/yoshitomo-matsubara/torchdistill) - PyTorch-based modular, configuration-driven framework for knowledge distillation.
* [TextBrewer](https://github.com/airaria/TextBrewer) (from HFL) - State-of-the-art distillation methods to compress language models.
* [BERT-of-Theseus](https://github.com/JetRunner/BERT-of-Theseus) (from Microsoft) - Compressing BERT by progressively replacing the components of the original BERT.

## 🏹️ Adversarial Attack
*Conducting adversarial attack to test model robustness.*
* [TextAttack](https://github.com/QData/TextAttack) (from UVa) -  A Python framework for adversarial attacks, data augmentation, and model training in NLP.
* [TextFlint](https://github.com/textflint/textflint) (from Fudan) - A unified multilingual robustness evaluation toolkit for NLP.
* [OpenAttack](https://github.com/thunlp/OpenAttack) (from THU) - An open-source textual adversarial attack toolkit.

## 🔁 Style Transfer
*Transfer the style of text! Now you know why it's called transformer?*
* [Styleformer](https://github.com/PrithivirajDamodaran/Styleformer) - A neural language style transfer framework to transfer text smoothly between styles.
* [ConSERT](https://github.com/yym6472/ConSERT) - A contrastive framework for self-supervised sentence representation transfer.

## 💢 Sentiment Analysis
*Analyzing the sentiment and emotions of human beings.*
* [conv-emotion](https://github.com/declare-lab/conv-emotion) - Implementation of different architectures for emotion recognition in conversations.

## 🙅 Grammatical Error Correction
*You made a typo! Let me correct it.*
* [Gramformer](https://github.com/PrithivirajDamodaran/Gramformer) - A framework for detecting, highlighting and correcting grammatical errors on natural language text.

## 🗺 Translation
*Translating between different languages.*
* [dl-translate](https://github.com/xhlulu/dl-translate) - A deep learning-based translation library based on HF Transformers.
* [EasyNMT](https://github.com/UKPLab/EasyNMT) (from UKPLab) - Easy-to-use, state-of-the-art translation library and Docker images based on HF Transformers.

## 📖 Knowledge and Entity
*Learning knowledge, mining entities, connecting the world.*
* [PURE](https://github.com/princeton-nlp/PURE) (from Princeton) - Entity and relation extraction from text.

## 🎙 Speech
*Speech processing powered by HF libraries. Need for speech!*
* [s3prl](https://github.com/s3prl/s3prl) - A self-supervised speech pre-training and representation learning toolkit.
* [speechbrain](https://github.com/speechbrain/speechbrain) - A PyTorch-based speech toolkit.

## 🤯 Multi-modal
*Understanding the world from different modalities.*
* [ViLT](https://github.com/dandelin/ViLT) (from Kakao) - A vision-and-language transformer Without convolution or region supervision.

## 🤖 Reinforcement Learning
*Combining RL magic with NLP!*
* [trl](https://github.com/lvwerra/trl) - Fine-tune transformers using Proximal Policy Optimization (PPO) to align with human preferences.

## ❓ Question Answering
*Searching for answers? Transformers to the rescue!*
* [Haystack](https://haystack.deepset.ai/) (from deepset) - End-to-end framework for developing and deploying question-answering systems in the wild. 
