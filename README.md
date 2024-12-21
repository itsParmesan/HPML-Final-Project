# HPML-Final-Project
This repository contains all the code written by Asmi Sriwastawa (as7045) and Sally Zhao (swz2108) for their project "Optimizing LLM Inference Latency on Chain of Thought Tasks" for COMS E6998 High Performance Machine Learning (Fall 2024).

## Background & Problem Statement
Chain of Thought (CoT) in LLMs is the idea that explicitly getting the model to break down its response into logical steps can lead to more accurate responses and also enable the model to tackle more complex tasks. Answering with step by step solutions to logic based tasks would also be more suitable for LLMs designed for chatting/assistance as this behavior is closer to that of humans working in the same role.

A trained LLM can be made to follow the CoT approach by:
1) Providing it with prompts that explicitly ask the model to demonstrate step-by-step thinking, include examples related to the task at hand etc
2) Fine tuning the model on a CoT dataset

The first approach, i.e. prompt engineering, results in model responses of higher quality but leads to slower response times as CoT task prompts are generally longer and context heavy, leading to longer input processing time and an increased number of output tokens to be generated. The second approach, i.e. fine tuning, can also be time consuming as LLMs contain a large number of parameters.

## Our Solution
We propose the application of quantization and torch.compile() to reduce response times for pre-trained LLMs on CoT tasks. We also explore the reduction of finetuning time using LoRA.

<img width="449" alt="Screenshot 2024-12-21 at 6 44 44 AM" src="https://github.com/user-attachments/assets/4afef5f2-153e-4de4-8ac6-1d3e3c1183bf" />

## Set Up
We experimented with 2 models in this project: Flan T5 (specifically trained for CoT tasks) and Llama2 (a general LLM). Our dataset of choice was the [CoT Collection](https://www.kaggle.com/datasets/konradb/chain-of-thought-collection/data/CoT_collection.json). All code was implemented on Google Colab. A T4 GPU was used when the models were only used for inference. A A100 GPU was used when finetuning.

## Code Structure
The code for each experiment is in a separate notebook. For each model the following experiments are performed (and the corresponding notebook is named as shown below):
1) \<Model>_Inference_Baseline: Measure the time taken by the model to generate responses for a small sample of CoT prompts.
2) \<Model>_Inference_torch_compile(): Measure the time taken by the model, compiled using torch.compile(), to generate responses for a small sample of CoT prompts. All the available modes for torch.compile() are tried out.
3) \<Model>_Inference_Quantization: Measure the time taken by the quantized version of the model to generate responses for a small sample of CoT prompts. Several different precisions are tried out for each model.
4) \<Model>_Finetuning_Baseline: Measure the time taken to finetune the model for a couple of epochs on a specific CoT task.
5) \<Model>_Finetuning_LoRA: Measure the time taken to finetune the model for a couple of epochs on a specific CoT task using LoRA.

## Results
Please find details on the obtained results and the related discussion in the paper submitted along with this repository.
