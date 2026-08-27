# Information Retrieval – Assignment

**Building a Boolean Retrieval System with Vocabulary Processing and Tolerant Retrieval**

> Verbatim transcription of `IR_Assignment.pdf` (4 pages). Wording preserved; only
> layout and list formatting normalised.

---

## 1. Problem Statement

Design and implement a Boolean Information Retrieval system for a collection of text
documents.

The system should transform raw documents into a searchable vocabulary and inverted
index, process Boolean queries, and support tolerant retrieval for vocabulary/spelling
variations.

The assignment should demonstrate your understanding of the complete IR pipeline:

```
Documents → Preprocessing → Vocabulary/Dictionary → Inverted Index → Postings Lists
          → Boolean Retrieval → Tolerant Retrieval → Evaluation
```

---

## 2. Dataset

Select a substantial and sufficiently diverse document collection from a publicly
available source and work **within one domain**, such as news, agriculture, healthcare,
education, finance, government, etc.

The collection should contain sufficient variation in **vocabulary**, **document length**
and **topics** to support meaningful experiments.

Report:

- Dataset source and attribution
- Domain
- Number of documents
- Approximate corpus size
- Number of tokens and unique terms

---

## 3. System Requirements

### A. Text Processing

Implement and compare:

- Tokenization
- Case normalization
- Stop-word removal
- Stemming
- Lemmatization

Study the principles of **Porter's stemming algorithm** and include representative
examples of stemming results.

### B. Vocabulary and Indexing

Construct:

- Term dictionary/vocabulary
- Document frequency information
- Inverted index
- Sorted postings lists

Report vocabulary and index statistics **before and after** preprocessing.

### C. Boolean Retrieval

Implement queries using:

- `AND`
- `OR`
- `NOT`
- Parentheses

The Boolean operations **must use the postings lists created by your system**.

Also compare **normal vs. postings-list-length-based query processing** using number of
comparisons and execution time.

### D. Tolerant Retrieval

Implement **at least one**:

- Wildcard retrieval
- Edit-distance-based retrieval
- k-gram-based retrieval

Evaluate tolerant retrieval using **at least 20 deliberately modified/misspelled
queries**.

### E. Evaluation

Create **at least 30 test queries** covering simple, compound, Boolean, morphological and
tolerant-retrieval cases.

Create relevance judgments and report:

- Precision
- Recall
- F1-score

---

## 4. Required Experiments

Your report must include:

1. **Preprocessing comparison** — effect of normalization, stop-word removal, stemming
   and lemmatization on vocabulary/index statistics and retrieval.
2. **Stemming vs. lemmatization** — compare retrieval results and identify undesirable
   transformations.
3. **Boolean query optimization** — compare number of comparisons and execution time.
4. **Tolerant retrieval** — compare exact vs. tolerant retrieval and discuss
   recall/false-positive trade-offs.

---

## 5. Demonstration

Demonstrate the working system in the **BITS Virtual Lab** platform.

Submit **one screenshot** showing the assignment being executed successfully in the
Virtual Lab platform.

---

## 6. Submission

Submit one **group submission** containing:

### Source Code

Executable and documented implementation of:

- Preprocessing
- Dictionary/vocabulary
- Inverted index and postings lists
- Boolean retrieval
- Tolerant retrieval
- Evaluation

### Technical Report

Containing:

1. Problem statement
2. Dataset
3. System methodology
4. Experimental setup
5. Results
6. Error analysis
7. Conclusion

The report must include sample intermediate outputs, sample postings lists, Boolean query
traces, relevant calculations, tables/plots and experimental results.

---

## 7. AI and Library Usage

The assignment is intended to assess students' understanding of fundamental IR concepts.

> **Do not use generative AI/LLMs to generate the implementation, experimental results or
> report.**
>
> **Do not use RAG/LLM retrieval systems or pre-built search engines such as
> Elasticsearch/OpenSearch as the core retrieval implementation.**

Standard programming and NLP libraries may be used for individual components where
appropriate. Clearly identify any external libraries and distinguish library-provided
functionality from your own implementation.

---

## 8. Marks Distribution

| Component | Marks |
| --- | --- |
| Text preprocessing and stemming/lemmatization | 1.5 |
| Vocabulary, dictionary, inverted index & postings lists | 1.5 |
| Boolean retrieval & query optimization | 2.0 |
| Tolerant retrieval | 1.5 |
| Experiments, evaluation & analysis | 1.5 |
| Code, report & documentation | 1.0 |
| Virtual Lab execution screenshot | 1.0 |
| **Total** | **10.0** |
