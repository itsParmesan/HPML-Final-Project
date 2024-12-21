# HPML-Final-Project
This repository contains all the code written by Asmi Sriwastawa (as7045) and Sally Zhao (swz2108) for their project "Optimizing LLM Inference Latency on Chain of Thought Tasks" for COMS E6998 High Performance Machine Learning (Fall 2024).

## Background
Chain of Thought (CoT) in LLMs is the idea that explicitly getting the model to break down its response into logical steps can lead to more accurate responses and also enable the model to tackle more complex tasks. Answering with step by step solutions to logic based tasks would also be more suitable for LLMs designed for chatting/assistance as this behavior is closer to that of humans working in the same role.

A trained LLM can be made to follow the CoT approach by:
1) Providing it with prompts that explicitly ask the model to demonstrate step-by-step thinking, include examples related to the task at hand etc
2) Fine tuning the model on a CoT dataset
The first approach, i.e., prompt engineering, results in model responses of higher quality but leads to slower response times as CoT task prompts are generally longer and context heavy, resulting in longer input processing time
