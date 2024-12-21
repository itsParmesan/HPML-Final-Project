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
1) `\<Model>_Inference_Baseline`: Measure the time taken by the model to generate responses for a small sample of CoT prompts.
2) `\<Model>_Inference_torch_compile()`: Measure the time taken by the model, compiled using torch.compile(), to generate responses for a small sample of CoT prompts. All the available modes for torch.compile() are tried out.
3) `\<Model>_Inference_Quantization`: Measure the time taken by the quantized version of the model to generate responses for a small sample of CoT prompts. Several different precisions are tried out for each model.
4) `\<Model>_Finetuning_Baseline`: Measure the time taken to finetune the model for a couple of epochs on a specific CoT task.
5) `\<Model>_Finetuning_LoRA`: Measure the time taken to finetune the model for a couple of epochs on a specific CoT task using LoRA.

## Code Execution
Any notebook can be run by simply selecting the "Run all" option in the "Runtime" dropdown. Kaggle or Hugging Face login might be required at some points in the notebooks, the required credentials for which are present in a text box above the cell running the login function in each of the notebooks.

## Results
### Results for torch.compile()

![torch_comp](https://github.com/user-attachments/assets/3fcca69f-8978-4dc3-93e2-5a8a8d263132)

 In the case of Flan-T5 we see the expected trend, i.e., the more aggressively torch.compile() optimizes the code the more reduction we observe in inference time. However, the same cannot be said for Llama-2. In this case, the same gains are not observed since Llama-2 contains components that cannot be handled by torch.compile() which results in graph breaks, increasing the inference time for the compiled model.

### Results for Quantization
 
![quantized_flan](https://github.com/user-attachments/assets/88bee037-4141-4722-bf98-f51872c054be)

The precisions evaluated in this study for Flan T5 included float16, int8, and int4. During implementation, we observed that while the data type for the float16 version of the model was correctly set to float16, the weights for the int8 and int4 versions were also stored as float16. This indicates that quantization for integer-level precisions is simulated rather than natively implemented. This approach is likely necessitated by the inability of hardware commonly accessible to us to support such levels of quantization like specialized hardware can. This explains the observed increase in inference time for lower precision configurations.

![quantized_llama](https://github.com/user-attachments/assets/badb8f98-c152-4c54-bfd9-65891b152a07)

The quantized versions of Llama-2 utilized in this project were generated using the llama.cpp framework, which effectively reduces model size while maintaining accuracy. Surprisingly, the baseline model was outperformed by some of its quantized counterparts in this study. This disparity may stem from the baseline model being the original Meta-optimized implementation, designed for efficient execution across a wide range of platforms -- a characteristic not inherently shared by the quantized models which were created using a third party library. Among the quantized versions, the model with int4 precision demonstrated the lowest inference latency.

### Results for LoRA

![lora_parameters](https://github.com/user-attachments/assets/f60145e6-c484-4094-b44d-5dfd68da96d7)

In the fine-tuning case, when LoRA is applied to both Flan T5 and Llama-2, all the original weights are frozen. This allow updates only for the newly added weights. We checked how many parameters are trainable in each case and found that after modifying the model for LoRA finetuning, only 2\% and 1.6\% of the parameters are now trainable for Flan T5 and Llama2 respectively.

![lora_finetuning](https://github.com/user-attachments/assets/ed27f5e7-a12c-46ec-bf15-11821a78a70d)

This obviously leads to a dramatic decrease in inference latency. A 27.98\% drop for Flan T5 and a 99.55\% drop for Llama-2 are observed. As a design choice, LoRA adaptors were added to all weight matrices in Flan T5 but only to query and value weight matrices in Llama-2 which is why a larger drop is observed for Llama-2.

## Conclusion
LLMs can be made to follow Chain-of-Thought reasoning (regardless of whether they were specifically trained for it or not) to provide better, more accurate answers without requiring exorbitantly high inference or finetuning time. However, different optimization techniques do indeed yield different performance results.

NOTE: Further details can be found in the paper submitted along with this repository.
