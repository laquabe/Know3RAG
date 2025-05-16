# Know³-RAG
code of Know³-RAG
## Start
Most environments can be installed:

```bash
 pip install -r requirements.txt
```
For entity linking, please see [Spacy Entity Linker](https://github.com/egerber/spaCy-entity-linker) to install local knowledge base.
## Datasets
See `datasets`
## Run
core code of Know³-RAG is in `code`. 
## Evaluation
For [hotpotQA](https://github.com/hotpotqa/hotpot) and [2wikimultihopQA](https://github.com/Alab-NII/2wikimultihop), we use the the official evaluation. Please see the evaluation in their repo.

For PopQA, we process the PopQA data into 2wikimultihopQA form and test it with the officail evaluation of 2wikimultihopQA.