# Experiment Analysis: Transformer Planning Capabilities in Path-Finding Tasks

## Introduction

Planning is a fundamental aspect of both human cognition and the functionality of large language models (LLMs). The study "[ALPINE: Unveiling the Planning Capability of Autoregressive Learning in Language Models](https://arxiv.org/abs/2405.09220)" investigates how Transformer-based LLMs develop planning abilities through next-word prediction mechanisms. This experiment demonstrates how Transformers can effectively capture both short and long-range relationships, focusing on source and target nodes in graph-based path-finding tasks.

## Experiment Overview

The experiment models planning as a graph path-finding task, where the objective is to generate a valid path from a source node to a target node. It evaluates the Transformer's ability to capture both local and global context, focus on critical nodes, and handle relationships inherent to graph-based structures.

---

## Configuration Details

| Parameter           | Single Attention | Multiple Attention |
|---------------------|------------------|--------------------|
| Vocabulary Size     | 50,000           | 50,000             |
| Context Length      | 50               | 50                 |
| Embedding Dimension | 120              | 120                |
| Number of Heads     | 1                | 3                  |
| Number of Layers    | 1                | 2                  |
| Dropout Rate        | 0.1              | 0.1                |
| QKV Bias            | True             | True               |
| Batch Size          | 10               | 10                 |
| Learning Rate       | 0.0004           | 0.0004             |
| Weight Decay        | 0.1              | 0.1                |

---

## Validation Results

| Metric                 | Single Attention | Multiple Attention |
|------------------------|------------------|--------------------|
| Successful Validations | 4,104            | 4,414              |
| Failed Validations     | 376              | 66                 |
| Hallucinations         | 5                | 5                  |
| Not Reachable          | 371              | 61                 |
| Error Percentage       | 9.16%            | 1.50%              |

---

## Attention Visualization

| Configuration                | Image Path                                  |
|------------------------------|---------------------------------------------|
| GPT Config Multi Attention   | ![](img/gpt_config_multi_attention.png)    |
| GPT Config Single Attention  | ![](img/gpt_config_attention.png)   |

---

## Attention Mechanism Analysis

### Single Attention Head
Observations:
- **Uniform Focus**:
  - The single head exhibits consistently high attention weights for previous, source, and target tokens.
  - This uniform distribution limits the model's ability to specialize in particular aspects of the graph traversal task.
- **Limited Differentiation**:
  - The attention map shows no clear preference for distinguishing between source and target nodes.
  - As a result, the model may struggle with complex relationships in graph structures.
- **Smoother Attention Distribution**:
  - The attention weights become smoother as token generation progresses, helping the model maintain coherence. However, this smoothness does not fully address accuracy issues caused by the lack of head specialization.

### Multiple Attention Heads
Observations:
- **Layer Specialization**:
  - Different attention heads across layers demonstrate clear roles. For example, some heads focus specifically on source and target nodes, while others capture broader contextual relationships.
- **Diverse Relationships**:
  - Multiple heads allow the model to distinguish between short and long-range dependencies effectively, enabling better performance on graph-based tasks.
- **Smoother Attention Distribution**:
  - Like the single-head configuration, attention distributions become smoother over time. However, this smoothness is complemented by the specialization of individual heads, leading to more accurate path generation.

---

## Negative Aspects

While the experiment demonstrates the strengths of Transformers in graph-based path-finding, several challenges remain:
- **Hallucinations**:
  - Both single and multi-head models occasionally generate edges that do not exist in the graph, leading to invalid paths.
- **Unreachable Paths**:
  - Certain paths remain unreachable, particularly in more complex graph topologies, due to incomplete or erroneous attention assignments.
- **Testing Limitations**:
  - The current experiment focuses on a limited set of Directed Acyclic Graph (DAG) topologies. Expanding the test set to include more diverse and complex DAG structures is crucial for evaluating the generalizability of the model.

---

## Key Observations from Attention Visualization

| **Aspect**                 | **Single Attention Head**                         | **Multiple Attention Heads**                           |
|----------------------------|--------------------------------------------------|-------------------------------------------------------|
| **Focus on Source/Target** | Uniform, lacks specialization.                   | Specific focus on source and target nodes by certain heads. |
| **Context Understanding**  | Limited differentiation of relationships.         | Diverse relationships captured by different heads.     |
| **Path Generation**        | Struggles with accuracy due to lack of focus.    | Smoother and more accurate token progression.         |
| **Hallucinations**         | Occasional hallucination of nonexistent edges.   | Same issue observed but less frequent.               |
| **Error Rate**             | Higher error percentage due to uniform attention.| Lower error percentage due to head specialization.    |

---

## Conclusion

The experiment demonstrates that introducing multiple attention heads enhances the model's ability to handle both short and long-range relationships in graph-based path-finding tasks. Key findings include:
- **Strengths**:
  - Multiple attention heads allow for specialization, smoother attention transitions, and improved accuracy.
  - Both configurations exhibit smooth attention distributions, aiding in coherent path generation.
- **Weaknesses**:
  - Challenges such as hallucinations of nonexistent edges and unreachable paths remain unresolved.
  - Testing on additional and more complex DAG topologies is necessary for robust evaluation.

These findings align with the principles outlined in the ALPINE study, showcasing the potential and limitations of Transformers in planning through autoregressive learning.