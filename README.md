# Interview Insights — Placement & Internship RAG Assistant

A lightweight Retrieval-Augmented Generation (RAG) application that helps students find practical placement and internship preparation advice from previous students' experiences.

The project is intentionally kept simple so that every component can be understood and explained clearly in interviews. It uses a small college-specific dataset, a lightweight Sentence Transformer for semantic retrieval, cosine similarity for ranking, and a single LLM call for answer generation.

## 1. Problem Statement

Placement and internship information is often scattered across previous students' experiences. Students may have to manually search through many responses to answer questions such as:

- What should I prepare for a particular company?
- What topics were asked in the technical interview?
- What was the selection procedure?
- What DSA/DBMS/OOPS topics should I prepare?
- What did previous students recommend?
- What was the eligibility requirement?
- How did previous students prepare for the test?

This project provides a single interface where students can ask such questions in natural language.

## 2. Project Goal

Build a simple RAG-based assistant that:

1. Takes a student's question.
2. Converts the question into an embedding.
3. Retrieves the most relevant previous student experiences.
4. Passes the retrieved experiences to an LLM.
5. Generates a concise answer grounded in the retrieved data.
6. Shows relevant student experiences as sources.

## 3. Why RAG?

A normal LLM may not know college-specific placement experiences.

For example:

> "What should I prepare for Cisco based on our previous students' experiences?"

The required information exists in our own dataset, not necessarily in the LLM's training data.

RAG solves this by retrieving relevant information from our dataset first and then giving that information to the LLM.

### Basic RAG flow

```text
User Question
      |
      v
Query Embedding
      |
      v
Cosine Similarity
      |
      v
Top 3 Relevant Experiences
      |
      v
Context + Question
      |
      v
LLM
      |
      v
Final Answer
```

## 4. Design Philosophy

The project deliberately avoids unnecessary complexity.

### We use

- Python
- Pandas
- NumPy
- Sentence Transformers
- `all-MiniLM-L6-v2`
- Cosine similarity
- A single LLM API call
- FastAPI
- React

### We currently avoid

- LangChain
- LlamaIndex
- Agents
- Multi-agent systems
- Query expansion
- Rerankers
- Fine-tuning
- Training an LLM
- Complex vector databases
- Multiple LLM calls
- Advanced Transformer architectures

The goal is to have a system whose complete data flow can be understood and explained.

# 5. Dataset

The dataset combines two original sources:

- Placement Talks 2025
- Internship Talks 2025

The cleaned dataset contains 57 student experiences.

Each experience represents one student's placement or internship journey.

### Important design decision

```text
1 student experience = 1 document = 1 chunk
```

We do not split each experience into arbitrary fixed-size chunks.

This is appropriate because each response represents a logically complete experience containing information such as the selection process, preparation, interview experience, important topics, and advice.

## 6. Dataset Cleaning

The original placement and internship forms did not have identical fields.

Instead of forcing the raw datasets to have identical structures, their fields were mapped into a common schema.

Examples:

```text
Placement:
Position you were recruited for

Internship:
Role

                ↓

Canonical field:
role
```

and:

```text
Placement:
CTC offered

Internship:
Stipend

                ↓

Canonical field:
compensation
```

Eligibility information from the separate branch-specific internship fields is consolidated into a single `eligibility` field.

Location was removed because it is not central to the intended retrieval task and may change between hiring cycles.

## 7. Removed Information

The cleaned dataset excludes information that is unnecessary for the assistant.

This includes:

- Personal identifying information
- Contact details
- Email addresses
- Names where unnecessary
- Location
- General questions about whether a company was a "dream company"
- General recession/IT-sector discussion
- Motivation when friends get placed
- Inspiration for choosing a field
- General opinions about whether the IT sector is luxurious
- Non-CSE coding guidance
- Off-campus vs on-campus comparison
- Recruitment-drive tracking
- AIDE-specific eligibility

The objective is to keep the retrieval corpus focused on actionable placement and internship information.

# 8. Canonical Dataset Structure

The cleaned dataset contains fields such as:

```text
experience_id
experience_type
branch
company
role
compensation
job_description
students_recruited
academic_profile
projects
eligibility
selection_procedure
test_description
test_preparation
interview_experience
important_topics
preparation_start
resources
last_minute_preparation
junior_advice
document_text
```

Not every experience contains every field.

For example, an internship may contain a job description while an older placement response may not.

Missing information is left empty rather than fabricated.

# 9. `document_text`

After cleaning, the structured fields are combined into a searchable document.

Example:

```text
Company: Cisco

Experience Type: Internship

Branch: Electronics and Communication Engineering

Role: Software Engineering Intern

Compensation: ...

Eligibility:
...

Selection Procedure:
...

Test Description:
...

Test Preparation:
...

Interview Experience:
...

Important Topics:
...

Resources:
...

Advice:
...
```

This complete text is what is converted into an embedding.

# 10. Embeddings

We use:

```text
all-MiniLM-L6-v2
```

from Sentence Transformers.

The model converts each document into a numerical vector.

For this project, the resulting embedding has 384 dimensions.

Conceptually:

```text
"Prepare DSA and DBMS"
          |
          v
Sentence Transformer
          |
          v
[0.12, -0.31, 0.44, ...]
```

We create document embeddings once and save them as:

```text
data/embeddings.npy
```

This prevents recalculating embeddings every time the application starts.

# 11. Semantic Retrieval

When a user asks a question, we also convert the question into an embedding.

For example:

```text
"What should I prepare for coding interviews?"
```

becomes a vector.

We then compare the query vector with every document vector using cosine similarity.

Conceptually:

```text
                    Query
                      |
                      v
                Query Vector
                      |
          +-----------+-----------+
          |           |           |
          v           v           v
       Doc 1        Doc 2       Doc 3
       0.82         0.31        0.76
          |
          v
      Rank results
          |
          v
      Top 3 documents
```

Higher cosine similarity means the vectors are closer in the embedding space.

# 12. Why Cosine Similarity?

We are interested in the direction of the vectors rather than their absolute magnitude.

Cosine similarity measures how similar two vectors are based on their angle.

The formula is:

```text
cosine_similarity(A, B)
    = (A · B) / (||A|| ||B||)
```

For this small dataset, calculating similarity against all documents is simple and fast.

A complex vector database is not necessary for the current dataset size.

# 13. RAG Generation

After retrieval, the top relevant experiences are combined into a context.

For example:

```text
Question:
What should I prepare for Cisco?

Retrieved Context:

Experience 1:
Cisco interview focused on ...

Experience 2:
Important topics included ...

Experience 3:
Previous student recommended ...
```

This context and the question are sent to the LLM.

The LLM is instructed to:

- Use the supplied student experiences.
- Avoid inventing information.
- Say when the available information is insufficient.
- Produce a concise answer.

The LLM is therefore primarily responsible for answer generation, not retrieval.

# 14. Separation of Responsibilities

The architecture intentionally separates retrieval and generation.

```text
Sentence Transformer
        |
        | "Which documents are relevant?"
        v
     Retriever
        |
        | Top 3 documents
        v
       LLM
        |
        | "How should I explain them?"
        v
     Answer
```

This keeps the system simple and makes it easier to debug.

# 15. Proposed Backend Structure

```text
backend/
│
├── create_embeddings.py
├── retrieval.py
├── generation.py
├── main.py
│
└── data/
    ├── interview_insights_cleaned.csv
    └── embeddings.npy
```

### `create_embeddings.py`

```text
CSV
 ↓
document_text
 ↓
Sentence Transformer
 ↓
embeddings.npy
```

### `retrieval.py`

```text
question
 ↓
query embedding
 ↓
cosine similarity
 ↓
top-k documents
```

### `generation.py`

```text
question + retrieved context
 ↓
LLM API
 ↓
answer
```

### `main.py`

Responsible for exposing the backend API.

# 16. Similarity Threshold

A useful improvement is a minimum similarity threshold.

This prevents the system from giving an apparently confident answer when none of the stored experiences are actually relevant.

Conceptually:

```text
Best similarity >= threshold
        |
        v
Retrieve context and generate answer
```

otherwise:

```text
"I could not find enough relevant information
in the available student experiences."
```

The threshold should be selected through testing rather than arbitrarily.

# 17. API Design

The intended backend endpoint is:

```http
POST /ask
```

Request:

```json
{
  "question": "What should I prepare for a Cisco technical interview?"
}
```

Response:

```json
{
  "answer": "Based on previous student experiences...",
  "sources": [
    {
      "experience_id": "...",
      "company": "Cisco",
      "branch": "ECE",
      "similarity": 0.81
    }
  ]
}
```

Returning sources is important because users should be able to see which experiences influenced the answer.

# 18. Frontend

The frontend will be implemented using React.

A minimal interface will contain:

```text
+---------------------------------------+
|       Interview Insights              |
|                                       |
| Ask about placements & internships    |
|                                       |
| [ What should I prepare for Cisco? ] |
|                                       |
|              [Ask]                    |
|                                       |
| Answer                                |
| ------------------------------------- |
| Based on previous experiences...      |
|                                       |
| Sources                               |
| • Cisco - ECE                         |
| • Cisco - CSE                         |
+---------------------------------------+
```

Optional filters can later be added for:

- Placement / Internship
- Branch
- Company

These filters are metadata operations and do not require another ML model.

# 19. Example Questions

### Preparation

```text
What should I prepare for a software placement?

What DSA topics were commonly asked?

How should I prepare for technical interviews?
```

### Company-specific

```text
What should I prepare for Cisco?

What was the selection process at Wells Fargo?

What topics were asked in Barclays interviews?
```

### Branch-specific

```text
What should an ECE student prepare for internships?

What are the important subjects for electrical students?
```

### Process

```text
What was the selection procedure?

What did students face in the coding test?

What resources did previous students use?
```

# 20. Evaluation

Create a small manually curated evaluation set of 20–30 questions.

For each query evaluate:

### Retrieval

Was the retrieved experience relevant?

### Generation

Was the final answer supported by the retrieved experiences?

### Hallucination

Did the LLM introduce information that was not present in the retrieved context?

A simple evaluation table can be maintained:

| Question | Relevant Retrieval | Answer Supported |
|---|---|---|
| Cisco preparation | Yes | Yes |
| DSA preparation | Yes | Yes |
| ECE internship preparation | Yes | Yes |

# 21. Installation

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scriptsctivate
```

Install the basic dependencies:

```bash
pip install pandas numpy sentence-transformers fastapi uvicorn groq
```

For the React frontend:

```bash
npm install
```

# 22. Environment Variables

Store the LLM API key as an environment variable:

```text
GROQ_API_KEY=your_api_key
```

Do not commit the API key to GitHub.

If using a `.env` file locally, add it to `.gitignore`.

# 23. Running the Project

### Step 1 — Create embeddings

```bash
python create_embeddings.py
```

This creates:

```text
data/embeddings.npy
```

### Step 2 — Start FastAPI

```bash
uvicorn main:app --reload
```

### Step 3 — Start React

From the frontend directory:

```bash
npm run dev
```

# 24. Current Project Status

### Completed

- [x] Original placement dataset collected
- [x] Original internship dataset collected
- [x] Dataset fields analyzed
- [x] Placement and internship schemas normalized
- [x] Unnecessary fields removed
- [x] Location removed
- [x] Personal information removed
- [x] 57 experiences retained
- [x] One experience treated as one document/chunk
- [x] `document_text` created
- [x] Sentence Transformer embedding generation implemented
- [x] 384-dimensional embeddings generated
- [x] Semantic similarity tested
- [x] Basic retrieval implemented

### Next

- [ ] Regenerate embeddings using cleaned dataset
- [ ] Refactor retrieval into a reusable function
- [ ] Add similarity threshold
- [ ] Connect one LLM API
- [ ] Implement RAG prompt
- [ ] Return sources
- [ ] Create FastAPI endpoint
- [ ] Build React interface
- [ ] Add optional metadata filters
- [ ] Evaluate using 20–30 test questions
- [ ] Deploy

# 25. Interview Explanation

A concise explanation of the project:

> "I built a lightweight RAG-based placement and internship assistant using a dataset of previous students' experiences. I treated each complete student experience as one document because it is a logically coherent unit of information. I converted these documents into 384-dimensional embeddings using the lightweight `all-MiniLM-L6-v2` Sentence Transformer. For a user query, I generate a query embedding and use cosine similarity to retrieve the top relevant experiences. These retrieved experiences are then passed as context to an LLM, which generates a concise answer grounded in the retrieved information. I kept retrieval and generation separate and avoided unnecessary frameworks and complex components because the dataset is small and I wanted the system to remain interpretable."

# 26. Key Concepts to Understand Before the Interview

You should be able to explain:

```text
1. What is an embedding?
2. Why use embeddings?
3. What is Sentence Transformer?
4. Why MiniLM?
5. What does 384 dimensions mean?
6. What is cosine similarity?
7. What is semantic search?
8. What is retrieval?
9. What is a chunk?
10. Why is one experience one chunk?
11. What is RAG?
12. Why not directly ask the LLM?
13. What is the role of the LLM?
14. What is hallucination?
15. Why provide retrieved context to the LLM?
16. Why use top-k retrieval?
17. Why use a similarity threshold?
18. Why separate metadata from document text?
19. Why not use a vector database for 57 documents?
20. How would the system scale to a much larger dataset?
```

# Architecture Summary

```text
                  CLEAN DATASET
                       |
                       v
             57 Student Experiences
                       |
             One Experience = One
                  Document
                       |
                       v
            all-MiniLM-L6-v2
                       |
                       v
                384D Embeddings
                       |
                       v
                 embeddings.npy


User Question
      |
      v
Query Embedding
      |
      v
Cosine Similarity
      |
      v
Top 3 Relevant Experiences
      |
      v
Question + Retrieved Context
      |
      v
       LLM
      |
      v
Grounded Answer + Sources
      |
      v
     React
```

This is the intended minimal architecture for the project.
