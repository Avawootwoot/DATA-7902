# DATA-7902
Generative Adaptive AI Interviewing Agent for Biography Generation in Elderly Care 
This project, developed for the Master of Data Science Capstone at The University of Queensland, aims to preserve the life stories of older adults. It uses an adaptive AI agent to conduct multi-turn interviews, ensuring narrative preservation through a grounded, hallucination-resistant framework.

# Overview
Unlike standard chatbots focused on simple companionship, this system is designed for structured narrative elicitation. It bridges the gap between engagement and formal biography generation by using a specialized RAG (Retrieval-Augmented Generation) pipeline.

# Key Features
Adaptive Dialogue: Uses reflection and planning prompts to guide the conversation based on user responses.

Fact-Grounded Generation: Employs a FAISS-based Grounding Module to ensure all generated content is traceable to original transcripts.

Structured Output: Automatically organizes life stories into chaptered narratives.

Traceability: Implements citation mapping between the final biography and the interview fact-table.

# System Architecture
The system consists of three core pillars:

Dialogue Manager: Powered by OpenAI GPT-5.2, handling multi-turn conversation logic and memory buffering.

Grounding Module: A FAISS vector store that indexes interview segments to prevent hallucinations.

Biography Generator: A controlled prompting engine that transforms raw transcripts into polished, chaptered text.
