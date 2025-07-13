# special-purpose-ai

special-purpose-ai  



\## Overview  



Problem: All AI Models are designed to store your questions and prompts as part of the learning process. That means your information will not be private, it will be part of global knowledge, and it can be given to other users. This is a problem for the Government or Finance or Medical industry, since it is very difficult to hide/mask personal information during interaction with AI. Also, it is very difficult to force the hosted AI model to use specific information from your organization to answer the question.  AI will hallucinate and may not provide focused answers.  
  
Solution: Special purpose private AI model can use your organization's knowledge to answer your question. Store your organization information in vector store. Then, use the vector store to create prompt that can be sent to local AI model or cloud AI model along with a question.  
  
DataJoin.net provides in-depth education and consultation on special purpose AI model.  
  
https://github.com/milan888-design/special-purpose-ai
  
## Flowchart- Special purpose AI  
```mermaid  
flowchart TD  
    A[specific knowledge] -->|Synchronize with| B[vector store]  
    C[your question] -->|is used by|  D[vector search algorithm]     
    D[vector search algorithm]  -->|searches vector store to create| E[prompt] 
    C[your question] -->|is used by|  D[AI model]  
    E[prompt]  -->|is used by|  F[AI model]  
    F[AI model]    -->|uses reasoning to produce| G[the answer] 
```  
  
## Flowchart- Special purpose AI details  
```mermaid  
flowchart TD  
    H[specific knowledge in database] -->|is a| A[specific knowledge]  
    I[specific knowledge in docx,pdf,xlsx, etc] -->|is a| A[specific knowledge]  
    J[on prem vector store] -->|is a| B[vector store]  
    K[cloud vector store] -->|is a| B[vector store] 
    L[on prem AI model] -->|is a|  D[AI model]  
    M[cloud AI model] -->|is a|  D[AI model]  
    A[specific knowledge] -->|Synchronize with| B[vector store] 
    C[your question] -->|is used by|  D[vector search algorithm]     
    D[vector search algorithm]  -->|searches vector store to create| E[prompt] 
    C[your question] -->|is used by|  D[AI model]  
    E[prompt]  -->|is used by|  F[AI model]  
    F[AI model]    -->|uses reasoning to produce| G[the answer] 
```  

