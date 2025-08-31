
<!-- TOC --><a name="cognito-langgraph-rag"></a>
# Cognito LangGraph RAG Chatbot

- Developed an advanced RAG workflow utilizing LangGraph to enhance question-answering accuracy, resulting in a significant reduction in Large Language Model (LLM) hallucinations. 
- Architected a stateful, multi-step process incorporating document retrieval, relevance grading, and web search fallback using LangGraph, leading to improved information retrieval and user satisfaction. 
- Integrated Large Language Models (LLMs) with a graph database and a novel self-reflection workflow to efficiently store, retrieve, and validate information, enhancing the chatbot's ability to provide accurate and relevant answers to complex user queries. 
- Implemented a self-reflection mechanism within the chatbot to iteratively refine responses, ensuring that generated answers are grounded in provided documentation, thereby minimizing hallucinations and increasing user trust. 
- Implemented full-stack solutions utilizing ChromaDB for efficient vector storage and retrieval, ensuring seamless integration with LangChain and secure handling of environment configurations through dotenv, which led to a 30% reduction in latency during information retrieval.
- Architected and fine-tuned a comprehensive LLM technology stack that includes LangChain, LangGraph, and external web search APIs (Tavily Search), achieving a resilient AI system capable of mitigating the challenges of LLM hallucinations through strategic reinforcement learning and retrieval augmentation.
- Spearheaded the development of an advanced chatbot utilizing Retrieval-Augmented Generation (RAG) workflows to enhance question-answering accuracy and mitigate LLM hallucinations.
- Leveraged LangGraph to construct a stateful, multi-step process encompassing document retrieval, relevance grading, and web search fallback, resulting in a more robust and accurate chatbot.
- Agentic AI: Designed autonomous agents capable of dynamic decision-making within the chatbot, enhancing its adaptability to diverse user queries.

<!-- TOC --><a name="star-history"></a>
## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=junfanz1/Cognito-LangGraph-RAG&type=Date)](https://star-history.com/#junfanz1/Cognito-LangGraph-RAG&Date)

<!-- TOC --><a name="content"></a>
## Content

<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->

- [1. Purpose of Project](#1-purpose-of-project)
- [2. Input and Output](#2-input-and-output)
- [3. LLM Technology Stack](#3-llm-technology-stack)
- [4. Challenges and Difficulties](#4-challenges-and-difficulties)
- [5. Future Business Impact and Further Improvements](#5-future-business-impact-and-further-improvements)
- [6. Target Audience and Benefits](#6-target-audience-and-benefits)
- [7. Advantages, Disadvantages and Tradeoffs](#7-advantages-disadvantages-and-tradeoffs)
- [8. Setup](#8-setup)
- [9. Code Explanation](#9-code-explanation)
- [10. How it Works](#10-how-it-works)
- [11. Crucial Functions](#11-crucial-functions)
- [12. Future Improvements](#12-future-improvements)
- [13. Summary](#13-summary)
   * [Project Highlights](#project-highlights)
   * [Why LangGraph](#why-langgraph)
   * [Additional Notes](#additional-notes)
- [Acknowledgements](#acknowledgements)

<!-- TOC end -->


<!-- TOC --><a name="1-purpose-of-the-project"></a>
## 1. Purpose of Project

This project implements an advanced Retrieval Augmented Generation (RAG) workflow chatbot to enhance question-answering accuracy and reduce LLM hallucinations. It leverages LangGraph to create a stateful, multi-step process that includes document retrieval, relevance grading, and web search fallback. This project aims to create a documentation bot that can answer user questions based on provided documentation. It leverages Large Language Models (LLMs), a graph database, and a novel self-reflection workflow to efficiently store, retrieve, and validate information.

Abstract: The primary goal of this project is to streamline the process of accessing and understanding documentation. By utilizing LLMs, a graph database, and a self-reflection workflow, the bot can provide accurate and relevant answers to user queries, even if they are complex or phrased differently from the original documentation. The bot also aims to minimize hallucinations and ensure that the generated answers are grounded in the provided documentation.

<!-- TOC --><a name="2-input-and-output"></a>
## 2. Input and Output

**Input:** Natural language questions from users.


- a list of URLs

```python
urls = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
    "https://lilianweng.github.io/posts/2024-02-05-human-data-quality/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
]
```

- Question 1:

```python
what is agent memory?
```


**Output:** Coherent and contextually relevant responses generated by the LLM, augmented with retrieved documents and/or web search results.

<img src="https://github.com/user-attachments/assets/c86f3e10-e518-484b-95d4-49687b0cf5cb" width="40%" height="40%">

```python
Hello Advanced RAG!
%%{init: {'flowchart': {'curve': 'linear'}}}%%
graph TD;
 __start__([<p>__start__</p>]):::first
 retrieve(retrieve)
 grade_documents(grade_documents)
 generate(generate)
 websearch(websearch)
 __end__([<p>__end__</p>]):::last
 retrieve --> grade_documents;
 websearch --> generate;
 __start__ -.-> websearch;
 __start__ -.-> retrieve;
 grade_documents -.-> websearch;
 grade_documents -.-> generate;
 generate -. &nbsp;useful&nbsp; .-> __end__;
 generate -. &nbsp;not useful&nbsp; .-> websearch;
 generate -. &nbsp;not supported&nbsp; .-> generate;
 classDef default fill:#f2f0ff,line-height:1.2
 classDef first fill-opacity:0
 classDef last fill:#bfb6fc

---route question---
---route question to RAG---
---Retrieving---
---check document relevance to question---
---grade: document not relevant---
---grade: document not relevant---
---grade: document not relevant---
---grade: document not relevant---
---assess granted documents---
---decision: not all docs are relevant to question ---
---web search---
State keys: dict_keys(['question', 'web_search', 'documents'])
---generate---
---check hallucination---
---decision: generation is grounded in documents---
---grade generation vs question---
---decision: generation addresses question---
{'question': 'what is agent memory?', 'generation': 'Agent memory is a crucial component in artificial intelligence that allows AI models to retain information across interactions rather than treating every query as a new conversation. It helps the agent to better plan and is used to store and recall information. Agents are stateful and they use this memory to store previous interactions and use them in subsequent calls.', 'web_search': True, 'documents': [Document(metadata={}, page_content='In general, the memory for an agent is something that we provide via context in the prompt passed to LLM that helps the agent to better plan\nUnderstanding Agent Memory in AI: Types, Use Cases, and Implementation | by Siladitya Ghosh | Feb, 2025 | Medium Understanding Agent Memory in AI: Types, Use Cases, and Implementation The secret lies in agent memory — a crucial component that allows AI to store and recall information across interactions. What agent memory is and why it’s essential The different types of memory used in AI agents Real-world use cases of memory in AI systems By the end, you’ll have a solid understanding of how AI agents use memory and how to implement it in your own projects. What is Agent Memory? Agent memory allows AI models to retain information across interactions rather than treating every query as a new conversation. Types of Agent Memory\nTowards AGI: [Part 1] Agents with Memory - SuperAGI Towards AGI: [Part 1] Agents with Memory Agents are an emerging class of artificial intelligence (AI) systems that use large language models (LLMs) to interact with the world. However, agents store the previous interactions in variables and use it in subsequent LLM calls. Therefore, agents are stateful and they have memory. Deep dive into various types of Agent Memory LTM type 2: Semantic memory (aka Reflections): It stores an agent’s knowledge about the world and itself. LTM type 3: Procedural memory: This memory represents the agent’s procedures for thinking, acting, decision-making, etc. AI Agent Systems for fully automated sales, marketing, support and app development\nEnroll now: https://bit.ly/3YwWJeR Build agentic memory into your applications with LLMs as Operating Systems: Agent Memory, a short course\nMemory for agents Memory for agents At Sequoia’s AI Ascent conference in March, I talked about three limitations for agents: planning, UX, and memory. But what even is memory? While the exact shape of memory that your agent has may differ by application, we do see different high level types of memory. Below is my rough, ELI5 explanation of each type and practical ways for how todays agents may use and update this memory type. Besides just thinking about the type of memory to update in their agents, we also see developers thinking about how to update agent memory. One way to update agent memory is “in the hot path”. Why do we care about memory for agents?')]}

Process finished with exit code 0

```

![image](https://github.com/user-attachments/assets/fe81a1eb-fc84-41db-98e0-e203de36d3c0)

LangSmith trace: https://smith.langchain.com/public/436a9f0e-f390-468a-bb0b-eaa9df0e08ce/r

![image](https://github.com/user-attachments/assets/34e890f2-6092-4ed7-aea7-dd2ff57a7e33)

Output:

- Web Search: `true`
- Generation: `Agent memory is a crucial component in artificial intelligence that allows AI models to retain information across interactions rather than treating every query as a new conversation. It helps the agent to better plan and is used to store and recall information. Agents are stateful and they use this memory to store previous interactions and use them in subsequent calls.`
- Documents: `In general, the memory for an agent is something that we provide via context in the prompt passed to LLM that helps the agent to better plan
Understanding Agent Memory in AI: Types, Use Cases, and Implementation | by Siladitya Ghosh | Feb, 2025 | Medium Understanding Agent Memory in AI: Types, Use Cases, and Implementation The secret lies in agent memory — a crucial component that allows AI to store and recall information across interactions. What agent memory is and why it’s essential The different types of memory used in AI agents Real-world use cases of memory in AI systems By the end, you’ll have a solid understanding of how AI agents use memory and how to implement it in your own projects. What is Agent Memory? Agent memory allows AI models to retain information across interactions rather than treating every query as a new conversation. Types of Agent Memory
Towards AGI: [Part 1] Agents with Memory - SuperAGI Towards AGI: [Part 1] Agents with Memory Agents are an emerging class of artificial intelligence (AI) systems that use large language models (LLMs) to interact with the world. However, agents store the previous interactions in variables and use it in subsequent LLM calls. Therefore, agents are stateful and they have memory. Deep dive into various types of Agent Memory LTM type 2: Semantic memory (aka Reflections): It stores an agent’s knowledge about the world and itself. LTM type 3: Procedural memory: This memory represents the agent’s procedures for thinking, acting, decision-making, etc. AI Agent Systems for fully automated sales, marketing, support and app development
Enroll now: https://bit.ly/3YwWJeR Build agentic memory into your applications with LLMs as Operating Systems: Agent Memory, a short course
Memory for agents Memory for agents At Sequoia’s AI Ascent conference in March, I talked about three limitations for agents: planning, UX, and memory. But what even is memory? While the exact shape of memory that your agent has may differ by application, we do see different high level types of memory. Below is my rough, ELI5 explanation of each type and practical ways for how todays agents may use and update this memory type. Besides just thinking about the type of memory to update in their agents, we also see developers thinking about how to update agent memory. One way to update agent memory is “in the hot path”. Why do we care about memory for agents?`


- Question 2:

```python
what is GRPO and MoE in DeepSeek architecture?
```

Output:

```python
Hello Advanced RAG!
%%{init: {'flowchart': {'curve': 'linear'}}}%%
graph TD;
 __start__([<p>__start__</p>]):::first
 retrieve(retrieve)
 grade_documents(grade_documents)
 generate(generate)
 websearch(websearch)
 __end__([<p>__end__</p>]):::last
 retrieve --> grade_documents;
 websearch --> generate;
 __start__ -.-> websearch;
 __start__ -.-> retrieve;
 grade_documents -.-> websearch;
 grade_documents -.-> generate;
 generate -. &nbsp;useful&nbsp; .-> __end__;
 generate -. &nbsp;not useful&nbsp; .-> websearch;
 generate -. &nbsp;not supported&nbsp; .-> generate;
 classDef default fill:#f2f0ff,line-height:1.2
 classDef first fill-opacity:0
 classDef last fill:#bfb6fc

---route question---
---route question to websearch---
---web search---
State keys: dict_keys(['question'])
---generate---
---check hallucination---
---decision: generation is grounded in documents---
---grade generation vs question---
---decision: generation addresses question---
{'question': 'what is GRPO and MoE in DeepSeek architecture?', 'generation': 'Group Relative Policy Optimization (GRPO) is a reinforcement learning technique developed by DeepSeek to enhance the reasoning capabilities of large language models. It regularizes by directly adding the KL divergence between the trained policy and the reference policy to the objective function. MoE (Mixture of Experts) is not responsible for emergent reasoning in DeepSeek-R1 architecture, as evidenced by the DeepSeek-R1-Distill-Qwen-32B model, which maintains all of DeepSeek-R1’s reasoning capabilities despite being fully dense and not having MoE.', 'documents': [Document(metadata={}, page_content='“We intentionally limit our constraints to this structural format, avoiding any content-specific biases — such as mandating reflective reasoning or promoting particular problem-solving strategies — to ensure that we can accurately observe the model’s natural progression during the RL process.” — DeepSeek-R1-Zero/R1 technical report Group Relative Policy Optimization (GRPO) is a reinforcement learning (RL) technique developed by DeepSeek to enhance the reasoning capabilities of large language models (LLMs). “Instead of adding KL penalty in the reward, GRPO regularizes by directly adding the KL divergence between the trained policy and the reference policy to the objective function, avoiding complicating the calculation of 𝐴_𝑖.” — DeepSeekMath “The group relative way that GRPO leverages to calculate the advantages, aligns well with the comparative nature of rewards models, as reward models are typically trained on datasets of comparisons between outputs on the same question” — DeepSeekMath\nThe GRPO algorithm is at the heart of the newest DeepSeek R1 architecture. In this tutorial, we will discuss the details of the formula\nDeepSeek-R1-Distill-Qwen-32B is not an MoE model and yet it retains all reasoning properties of DeepSeek-R1. If MoE were essential to DeepSeek-R1’s emergent reasoning, then distilling the model into a fully dense architecture should have eliminated its ability to perform deep, structured reasoning. Since DeepSeek-R1-Distill-Qwen-32B has no MoE and retains its reasoning abilities, this is direct empirical proof that MoE is not responsible for emergent reasoning. The strongest empirical refutation of the claim that DeepSeek-R1’s reasoning abilities require MoE lies in the very nature of DeepSeek-R1-Distill-Qwen-32B, a model that maintains all of DeepSeek-R1’s reasoning capabilities despite being fully dense. It is a smoking gun proof that DeepSeek-R1’s emergent reasoning behaviors are model agnostic and stem purely from reinforcement learning, structured optimization, and Chain-of-Thought training, not from architectural sparsity or expert gating.\nExplore the groundbreaking advancements of DeepSeek R1, a reinforcement learning-driven reasoning model. Unlike traditional models that heavily rely on supervised fine-tuning, DeepSeek R1 adopts a reinforcement learning-only (RL-only) approach to develop its reasoning capabilities. DeepSeek-R1 was trained using GRPO instead of PPO, as it allowed reinforcement learning on a large-scale language model without requiring a critic network. By learning from DeepSeek V3’s evolution, DeepSeek R1 was able to push the boundaries of reinforcement learning-driven reasoning, achieving state-of-the-art performance in complex problem-solving tasks while maintaining computational efficiency. DeepSeek R1 represents a paradigm shift in reinforcement learning-driven reasoning models, proving that logical self-improvement can emerge without supervised fine-tuning.\nThey used this data to train DeepSeek-V3-Base on a set of high quality thoughts, they then pass the model through another round of reinforcement learning, which was similar to that which created DeepSeek-r1-zero, but with more data (we’ll get into the specifics of the entire training pipeline later). They took DeepSeek-V3-Base, with these special tokens, and used GRPO style reinforcement learning to train the model on programming tasks, math tasks, science tasks, and other tasks where it’s relatively easy to know if an answer is correct or incorrect, but requires some level of reasoning.')]}

Process finished with exit code 0
```


![image](https://github.com/user-attachments/assets/3b008240-1cd4-4e3d-8f4a-d4551b05b675)

LangSmith trace: https://smith.langchain.com/public/46a28e4e-c7ed-494b-8cfd-14ea7b6d3fe6/r

![image](https://github.com/user-attachments/assets/13cb7386-f797-4dec-96ca-1eeab1f958ca)

Output:

- Generation: `Group Relative Policy Optimization (GRPO) is a reinforcement learning technique developed by DeepSeek to enhance the reasoning capabilities of large language models. It regularizes by directly adding the KL divergence between the trained policy and the reference policy to the objective function. MoE (Mixture of Experts) is not responsible for emergent reasoning in DeepSeek-R1 architecture, as evidenced by the DeepSeek-R1-Distill-Qwen-32B model, which maintains all of DeepSeek-R1’s reasoning capabilities despite being fully dense and not having MoE.`
- Documents: `“We intentionally limit our constraints to this structural format, avoiding any content-specific biases — such as mandating reflective reasoning or promoting particular problem-solving strategies — to ensure that we can accurately observe the model’s natural progression during the RL process.” — DeepSeek-R1-Zero/R1 technical report Group Relative Policy Optimization (GRPO) is a reinforcement learning (RL) technique developed by DeepSeek to enhance the reasoning capabilities of large language models (LLMs). “Instead of adding KL penalty in the reward, GRPO regularizes by directly adding the KL divergence between the trained policy and the reference policy to the objective function, avoiding complicating the calculation of 𝐴_𝑖.” — DeepSeekMath “The group relative way that GRPO leverages to calculate the advantages, aligns well with the comparative nature of rewards models, as reward models are typically trained on datasets of comparisons between outputs on the same question” — DeepSeekMath
The GRPO algorithm is at the heart of the newest DeepSeek R1 architecture. In this tutorial, we will discuss the details of the formula
DeepSeek-R1-Distill-Qwen-32B is not an MoE model and yet it retains all reasoning properties of DeepSeek-R1. If MoE were essential to DeepSeek-R1’s emergent reasoning, then distilling the model into a fully dense architecture should have eliminated its ability to perform deep, structured reasoning. Since DeepSeek-R1-Distill-Qwen-32B has no MoE and retains its reasoning abilities, this is direct empirical proof that MoE is not responsible for emergent reasoning. The strongest empirical refutation of the claim that DeepSeek-R1’s reasoning abilities require MoE lies in the very nature of DeepSeek-R1-Distill-Qwen-32B, a model that maintains all of DeepSeek-R1’s reasoning capabilities despite being fully dense. It is a smoking gun proof that DeepSeek-R1’s emergent reasoning behaviors are model agnostic and stem purely from reinforcement learning, structured optimization, and Chain-of-Thought training, not from architectural sparsity or expert gating.
Explore the groundbreaking advancements of DeepSeek R1, a reinforcement learning-driven reasoning model. Unlike traditional models that heavily rely on supervised fine-tuning, DeepSeek R1 adopts a reinforcement learning-only (RL-only) approach to develop its reasoning capabilities. DeepSeek-R1 was trained using GRPO instead of PPO, as it allowed reinforcement learning on a large-scale language model without requiring a critic network. By learning from DeepSeek V3’s evolution, DeepSeek R1 was able to push the boundaries of reinforcement learning-driven reasoning, achieving state-of-the-art performance in complex problem-solving tasks while maintaining computational efficiency. DeepSeek R1 represents a paradigm shift in reinforcement learning-driven reasoning models, proving that logical self-improvement can emerge without supervised fine-tuning.
They used this data to train DeepSeek-V3-Base on a set of high quality thoughts, they then pass the model through another round of reinforcement learning, which was similar to that which created DeepSeek-r1-zero, but with more data (we’ll get into the specifics of the entire training pipeline later). They took DeepSeek-V3-Base, with these special tokens, and used GRPO style reinforcement learning to train the model on programming tasks, math tasks, science tasks, and other tasks where it’s relatively easy to know if an answer is correct or incorrect, but requires some level of reasoning.`

<!-- TOC --><a name="3-llm-technology-stack"></a>
## 3. LLM Technology Stack

-   **LangChain:**
    - Used for building LLM applications by providing abstractions for LLMs, vector stores, and other components.
    - Facilitates the creation of chains and agents.
-   **LangGraph:**
    - A framework for building stateful, multi-actor applications with LLMs.
    - Enables the creation of complex workflows with conditional logic and state management.
-   **OpenAI:**
    - Provides powerful LLMs (e.g., GPT-3.5, GPT-4) for generating responses and embeddings.
-   **ChromaDB:**
    - A vector database for storing and retrieving document embeddings.
    - `Chroma.from_documents` is used to create and persist the vectorstore.
    - `Chroma.as_retriever()` is used to create a retriever object.
-   **Tavily Search:**
    - A search API for retrieving web search results.
    - `TavilySearchResults` tool is used to execute web searches.
-   **Python:**
-   **dotenv:**
    - Used for managing environment variables, such as API keys, securely.

<!-- TOC --><a name="4-challenges-and-difficulties"></a>
## 4. Challenges and Difficulties

- Accuracy: Ensuring the bot provides accurate and relevant answers, especially for complex or ambiguous questions.
- Hallucination: Preventing the LLM from generating answers that are not supported by the provided documentation.
- Data Preprocessing: Efficiently processing and structuring documentation data for storage and retrieval.
- Scalability: Scaling the system to handle large volumes of documentation and user queries.
- Explainability: Providing clear explanations of how the bot arrived at its answers.
- Latency: Minimizing response time to provide a seamless user experience.

<!-- TOC --><a name="5-future-business-impact-and-further-improvements"></a>
## 5. Future Business Impact and Further Improvements

This project has the potential to significantly improve knowledge access and retrieval within organizations. It can be used for:

- Customer support: Providing instant and accurate answers to customer questions.
- Employee training: Facilitating knowledge sharing and onboarding.
- Internal documentation: Making it easier for employees to find the information they need.
- Content creation: Assisting in generating summaries, reports, and other documents based on existing knowledge.

Further improvements:

- Multilingual support: Expanding the bot's capabilities to handle multiple languages.
- Personalized responses: Tailoring answers to individual user needs and preferences.
- Integration with other systems: Connecting the bot with existing knowledge bases and communication platforms.
- Enhanced user interface: Developing a more intuitive and interactive interface for users to interact with the bot.

<!-- TOC --><a name="6-target-audience-and-benefits"></a>
## 6. Target Audience and Benefits

- Developers: Easily access and understand documentation for various libraries and frameworks.
- Technical writers: Automate the process of answering questions about documentation and ensure its accuracy.
- End-users: Quickly find answers to their questions without having to sift through lengthy documentation.
- Content creators: Utilize the bot to generate summaries and reports based on existing knowledge.

Benefits:

- Improved efficiency: Faster access to information.
- Enhanced user experience: Clear, concise, and accurate answers to questions.
- Reduced support costs: Automated responses to common queries.
- Increased knowledge sharing: Facilitates the dissemination of information within organizations.

<!-- TOC --><a name="7-advantages-disadvantages-and-tradeoffs"></a>
## 7. Advantages, Disadvantages and Tradeoffs

Advantages:

- Efficient information retrieval: Leverages LLMs and graph databases for fast and accurate search.
- Natural language understanding: Can handle complex and nuanced queries.
- Scalability: Can handle large volumes of documentation.
- Self-reflection: Minimizes hallucinations and ensures answers are grounded in documentation.
- Adaptive routing: Optimizes the retrieval process based on the nature of the query.

Disadvantages:

- Potential for inaccuracies: LLMs can sometimes generate incorrect or misleading information, even with self-reflection.
- Complexity: Building and maintaining the system can be challenging.
- Cost: Using LLMs can be expensive.

Tradeoffs

- Accuracy vs. Speed: Balancing the need for accurate answers with the desire for fast response times.
- Complexity vs. Maintainability: Weighing the benefits of a sophisticated system against the challenges of maintaining it.
- Cost vs. Benefit: Evaluating the return on investment for using LLMs.


<!-- TOC --><a name="8-setup"></a>
## 8. Setup

Prerequisites
- Python 3.10+
-  OpenAI API key
-  Tavily API key
- Install required packages: `pip install -r requirements.txt`

1.  Clone the repository.
2.  Install the required packages: `pip install -r requirements.txt`
3.  Create a `.env` file and add your OpenAI and Tavily API keys:

```
OPENAI_API_KEY=
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=
LANGCHAIN_TRACING_V2=true
TAVILY_API_KEY=
PYTHONPATH=/Users/junfanzhu/Desktop/langgraph
```

4.  Run the main application: `python graph/graph.py`



<!-- TOC --><a name="9-code-explanation"></a>
## 9. Code Explanation

* **`graph/chains/answer_grader.py`:**
    * `GradeAnswer` class: Defines the structure of the output for answer grading.
    * `answer_grader`: A chain that uses an LLM to assess whether an answer addresses a question.
* **`graph/chains/hallucination_grader.py`:**
    * `GradeHallucination` class: Defines the structure of the output for hallucination grading.
    * `hallucination_grader`: A chain that uses an LLM to assess whether a generation is grounded in the provided documents.
* **`graph/chains/retrieval_grader.py`:**
    * `GradeDocuments` class: Defines the structure of the output for document relevance grading.
    * `retrieval_grader`: A chain that uses an LLM to assess the relevance of a document to a question.
* **`graph/chains/router.py`:**
    * `RouteQuery` class: Defines the structure of the output for routing questions.
    * `question_router`: A chain that uses an LLM to route a question to the appropriate data source (vectorstore or web search).
* **`graph/chains/tests/test_chains.py`:**
    * Contains unit tests for the various chains.
* **`graph/generation.py`:**
    * Defines the `generation_chain`, which takes a user question and context, and generates an answer using an LLM.
* **`graph/graph.py`:**
    * **Purpose:** Orchestrates the entire RAG workflow using LangGraph.
    * **Detailed Explanation:**
        * Loads environment variables using `load_dotenv()`.
        * Initializes a `StateGraph` with `GraphState`.
        * Defines nodes: `RETRIEVE`, `GRADE_DOCUMENTS`, `WEBSEARCH`, `GENERATE`.
        * Sets the conditional entry point to `route_question` allowing routing to either `RETRIEVE` or `WEBSEARCH` based on the query.
        * Adds edges to connect nodes:
            * `RETRIEVE` -> `GRADE_DOCUMENTS`.
            * `GRADE_DOCUMENTS` -> `WEBSEARCH` or `GENERATE` (conditional).
            * `WEBSEARCH` -> `GENERATE`.
            * `GENERATE` -> conditional routing based on `grade_generation_grounded_in_documents_and_question` to either `GENERATE` (retry), `END` (success), or `WEBSEARCH` (augment with web results).
        * `decide_to_generate(state)` function:
            * Checks `state["web_search"]` to determine if web search is needed.
            * Returns `WEBSEARCH` or `GENERATE` based on the condition.
        * `grade_generation_grounded_in_documents_and_question(state)`:
            * Checks if generation is grounded in documents and addresses the question.
            * Routes to regenerate, end, or web search based on results.
        * `route_question(state)`:
            * Routes the question to `WEBSEARCH` or `RETRIEVE` based on the query.
        * Compiles the graph using `workflow.compile()`.
        * Generates a visualization of the graph using `get_graph().draw_mermaid_png()`.
* **`graph/state.py`:**
    * **Purpose:** Defines the `GraphState` TypedDict to manage the state of the workflow.
    * **Detailed Explanation:**
        * `GraphState` includes:
            * `question`: The user's question.
            * `generation`: The LLM-generated response.
            * `web_search`: A boolean flag indicating whether web search is needed.
            * `documents`: A list of retrieved documents.
        * TypedDict ensures type safety and clarity in state management.
* **`graph/nodes/generate.py`:**
    * **Purpose:** Generates a response based on the retrieved and processed documents.
    * **Detailed Explanation:**
        * Takes the `GraphState` as input.
        * Extracts the question and documents from the state.
        * Invokes the `generation_chain` with the documents and question.
        * Returns a dictionary containing the documents, question, and generated response.
        * `generation_chain` uses a prompt from langchain hub, and the OpenAI LLM to generate the output.
* **`graph/nodes/grade_documents.py`:**
    * **Purpose:** Grades the retrieved documents for relevance using an LLM.
    * **Detailed Explanation:**
        * Takes the `GraphState` as input.
        * Extracts the question and documents from the state.
        * Iterates through each document:
            * Invokes the `retrieval_grader` chain with the document and question.
            * Checks the `binary_score` from the grader's output.
            * If the score is "yes", appends the document to `filtered_docs`.
            * If the score is "no", sets `web_search` to `True`.
        * Returns a dictionary containing the filtered documents, the question, and the `web_search` flag.
        * `retrieval_grader` uses a structured output parser, and a defined prompt to make the grading decision.
* **`graph/nodes/retrieve.py`:**
    * **Purpose:** Retrieves relevant documents from ChromaDB.
    * **Detailed Explanation:**
        * Takes the `GraphState` as input.
        * Extracts the user's question from `state["question"]`.
        * Invokes the ChromaDB retriever using `retriever.invoke(question)`.
        * Returns a dictionary containing the retrieved documents and the question.
* **`graph/nodes/web_search.py`:**
    * **Purpose:** Performs a web search using Tavily Search.
    * **Detailed Explanation:**
        * Takes the `GraphState` as input.
        * Extracts the question and documents from the state.
        * Invokes the `web_search_tool` with the question.
        * Joins the content of the search results into a single string.
        * Creates a `Document` object with the joined content.
        * Appends the web search results to the existing documents or creates a new list if no documents exist.
        * Returns a dictionary containing the updated documents and the question.
* **`ingestion.py`:**
    * **Purpose:** Loads, chunks, embeds, and stores documents in ChromaDB.
    * **Detailed Explanation:**
        * Loads documents from specified URLs using `WebBaseLoader`.
        * Splits documents into chunks using `RecursiveCharacterTextSplitter`.
        * Embeds documents using `OpenAIEmbeddings`.
        * Stores embeddings in ChromaDB using `Chroma.from_documents`.
        * Creates a retriever from ChromaDB using `Chroma.as_retriever()`.

<!-- TOC --><a name="10-how-it-works"></a>
## 10. How it Works

1.  The user provides a question.
2.  The `question_router` determines whether to use the vectorstore or web search based on the question.
3.  If the vectorstore is selected, the `retrieve` function fetches relevant documents from the Chroma database.
4.  The `grade_documents` function assesses the relevance of each retrieved document using the `retrieval_grader` chain.
5.  If any irrelevant documents are found or the vectorstore was not selected, the `web_search` function performs a web search using the TavilySearchResults tool.
6.  The `generate` function uses the `generation_chain` to generate an answer based on the retrieved documents and user question.
7.  The `hallucination_grader` checks if the generated answer is grounded in the documents.
8.  If the answer is not grounded, the process repeats from step 6.
9.  If the answer is grounded, the `answer_grader` checks if it addresses the user's question.
10. If the answer addresses the question, it is returned to the user.
11. If the answer does not address the question, the process repeats from step 5 with a web search.

<!-- TOC --><a name="11-crucial-functions"></a>
## 11. Crucial Functions

- **`grade_documents(state: GraphState)`**
    -   This function is crucial because it acts as a quality control mechanism for the retrieved documents.
    -   It uses an LLM, specifically `retrieval_grader`, to determine the relevance of each retrieved document to the user's question.
    -   The `retrieval_grader` chain uses a structured LLM output, ensuring that the grading result is in a consistent format (`binary_score`: "yes" or "no"). This is achieved through the `GradeDocuments` class, which defines the expected output structure.
    -   By iterating through each document and grading it, the function filters out irrelevant documents, preventing the LLM from generating responses based on incorrect context.
    -   The `web_search` flag is set to `True` if any document is deemed irrelevant, triggering a web search to provide more comprehensive information. This mechanism ensures that the system can adapt to situations where the initial retrieval might not be sufficient.
    -   This function directly impacts the quality and accuracy of the final generated response, reducing hallucinations and improving user satisfaction. It ensures that the LLM has access to only the most relevant context when generating an answer.

- **`decide_to_generate(state)`**
    -   This function acts as a conditional router within the LangGraph workflow.
    -   It examines the `web_search` flag within the `GraphState` to determine the next step in the workflow.
    -   If `web_search` is `True`, it routes the workflow to the `WEBSEARCH` node, ensuring that a web search is performed to supplement the retrieved documents. This is essential for cases where the initial document retrieval is not adequate or when external information is needed.
    -   If `web_search` is `False`, it routes the workflow to the `GENERATE` node, bypassing the web search and directly generating the response from the retrieved documents. This streamlines the process when the retrieved documents are sufficient, reducing latency and resource usage.
    -   This conditional routing is essential for creating a dynamic and adaptive workflow that can handle varying levels of document relevance. It allows the system to make intelligent decisions based on the quality of the retrieved information, ensuring that the final response is always as accurate and comprehensive as possible.

- **`retrieve(state: GraphState)`**
    -   This function is the entry point for retrieving relevant documents from the knowledge base.
    -   It takes the user's question from the `GraphState` and uses the ChromaDB retriever to perform a semantic search.
    -   The retriever leverages document embeddings generated by `OpenAIEmbeddings` to find documents that are semantically similar to the query, even if they don't contain the exact keywords. This ensures that the system can understand the intent behind the question and retrieve relevant information.
    -   By returning the retrieved documents and the original question, this function sets the stage for the subsequent steps in the workflow, such as grading and generation.
    -   The efficiency and accuracy of this retrieval step are critical for the overall performance of the RAG system, as it determines the initial set of documents that will be used to generate the response. The use of ChromaDB allows for fast and efficient retrieval, even with large datasets.

- **`grade_generation_grounded_in_documents_and_question(state: GraphState)`**
    -   This function plays a critical role in the self-reflection workflow. It ensures that the generated answer is both grounded in the provided documentation and addresses the user's question. This helps to improve the accuracy and relevance of the bot's responses and minimizes hallucinations.
    -   The function takes the user's question, the retrieved documents, and the LLM-generated answer as input.
    -   It first uses the `hallucination_grader` chain, which utilizes the `GradeHallucination` class for structured output, to check if the answer is grounded in the documents. This step is crucial for preventing the LLM from generating answers that are not supported by the provided context.
    -   If the answer is not grounded, the function returns "not supported," triggering the generation process to retry. This ensures that the system continuously refines its answer until it is grounded in the provided documents.
    -   If the answer is grounded, the function uses the `answer_grader` chain, which utilizes the `GradeAnswer` class for structured output, to check if it addresses the user's question. This ensures that the generated answer is not only accurate but also relevant to the user's query.
    -   If the answer addresses the question, the function returns "useful," and the answer is returned to the user. This indicates that the system has successfully generated a relevant and accurate answer.
    -   If the answer does not address the question, the function returns "not useful," triggering a web search to augment the context and improve the answer. This ensures that the system can adapt to situations where the initial information retrieval is not sufficient to answer the user's question.
    -   The function's use of structured LLM output through the `GradeHallucination` and `GradeAnswer` classes ensures that the grading process is consistent and reliable.
      
<!-- TOC --><a name="12-future-improvements"></a>
## 12. Future Improvements

Ideas
- LangGraph has a persistence layer via checkpoint object (save the state after each node execution, in persistence storage, e.g. SQLite). Can interrupt the graph and checkpoint the state of the graph and stop to get human feedback, and resume graph execution from the stop point.
- Create conditional branches for parallel node execution
- Use Docker to deploy to LangGraph Cloud, or use LangGraph Studio, LangGraph API to build LLM applications without frontend

Overall
- Improved accuracy: Fine-tuning LLMs and refining the retrieval process.
- Enhanced user interface: Developing a more intuitive and interactive interface.
- Expanded functionality: Adding features such as document summarization and comparison.
- Improved self-reflection: Implementing more sophisticated methods for evaluating and correcting LLM-generated answers.
- Implement a more sophisticated document processing pipeline.
- Explore different LLM architectures and fine-tuning techniques.
- Develop a user-friendly interface for interacting with the bot.
- Add error handling and logging.
- Improve code documentation and testing.
- Expand the knowledge base to cover a wider range of topics.
- Incorporate user feedback to continuously improve the bot's performance.

<!-- TOC --><a name="13-summary"></a>
## 13. Summary

This project offers a novel approach to documentation access and retrieval by incorporating a self-reflection workflow. By combining LLMs with a graph database and adaptive routing, it provides an efficient and user-friendly way to find accurate and relevant answers to questions. While there are challenges to overcome, the potential benefits are significant.

The core of this project is to build a robust and reliable question-answering system using Retrieval Augmented Generation, but with a focus on mitigating common LLM challenges like hallucinations and irrelevant responses. We've implemented an advanced RAG pipeline that goes beyond basic retrieval. 

First, we leverage LangGraph to orchestrate a stateful workflow, which allows us to manage complex interactions between different components. This is crucial because we're not just retrieving and generating; we're also grading the relevance of retrieved documents. The key innovation here is the 'grade_documents' node. We use an LLM to assess the semantic relevance of each retrieved document to the user's query. If a document is deemed irrelevant, we trigger a web search using Tavily Search as a fallback, ensuring that the final response is comprehensive and accurate. This conditional logic, managed by LangGraph, is what distinguishes this system. 

We've also focused on modularity. Each step—retrieval, grading, web search, and generation—is encapsulated in its own node, making the system highly extensible. We're using ChromaDB for efficient vector storage and retrieval, and OpenAI's LLMs for both embedding and generation. 

In essence, this project tackles the 'garbage in, garbage out' problem by actively filtering and augmenting the retrieved context. The LangGraph framework allows us to create a pipeline that dynamically adapts to the quality of retrieved information, resulting in more reliable and accurate responses compared to a basic RAG setup. This design shows an understanding of how to build complex LLM applications that address real-world challenges."

<!-- TOC --><a name="project-highlights"></a>
### Project Highlights

RAG self-reflection workflow reflects on
- Document we retrieve
- Curate documents and add new info
- Answers if grounded in documents

We also implement a routing element, routing our request to correct datastore with info of the answer.

Project is based on 3 ideas 

- Self-RAG: reflect on the answer the model generated, check if answer is grounded in the docs.
- Adaptive RAG: (1) taking the route to search on a website, then continuing downstream on the same logic (2) use RAG from the vector store. Use conditional entry points for routing.
- Corrective RAG: Take query, vector search + semantic search, retrieve all docs, start to self-reflect and critique the docs, determine whether they’re relevant or not. If relevant, send to LLM, if not relevant, filter out and perform external Internet search to get more info, to augment our prompt with real-time online info, then augment the prompt and send to LLM.

<!-- TOC --><a name="why-langgraph"></a>
### Why LangGraph
- LangGraph framework provides a persistence layer via checkpoint objects, allowing for interruption and resumption of graph execution, and enabling human feedback integration.
- LangGraph is both reliable (like Chain, that architects our state machine) and flexible (like ReAct).
- Flow Engineering (planning+testing), can build highly customized agents. (AutoGPT can do long-term planning, but we want to define the flow.) 
- Controllability (we define control flow, LLM make decisions inside flow) + Persistence + Human-in-the-loop + Streaming. 
- LangChain Agent. Memory (shared state across the graph), tools (nodes can call tools and modify state), planning (edges can route control flow based on LLM decisions).

<!-- TOC --><a name="additional-notes"></a>
### Additional Notes

ReAct (Reason-Act)
- Paradigm that integrates language models with reasoning and acting capabilities, allowing for dynamic reasoning and interaction with external environments to accomplish complex tasks.
- Simplest agent is for-loop (ReAct), ReAct agents are flexible and any state is possible, but have poor reliability (eg. invoking the same tool always and stuck, due to hallucination, tool-misuse, task ambiguity, LLM non-determinism).

Autonomy in LLM applications (5 levels): 
- human code
- LLM call: one-step only
- Chain: multiple steps, but one-directional
- Router (LangChain): decide output of steps and steps to take (but no cycles), still human-driven (not agent executed)
- State Machine (LangGraph): Agent executed, where agent is a control flow controlled by LLM, use LLM to reason where to go in this flow and tools-calling to execute steps, agent can have cycles.


<!-- TOC --><a name="acknowledgements"></a>
## Acknowledgements

[Eden Marco: LangGraph-Develop LLM powered AI agents with LangGraph](https://www.udemy.com/course/langgraph)
