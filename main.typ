#set text(font: "Noto Sans")
#set text(lang: "en", region: "au")
#[
  #set text(size: 50pt)
  INFS7410 Revision
]

Foreword:

These are notes written by a web-developer who has no idea on web-indexing, read at your own risk. If you have thoughts comment on this document via Typst.app or via the GitHub discussion. Input is welcome! 

#pagebreak()

#outline()

#pagebreak()
#set heading(numbering: "1.")
= The Basics
#set quote(block: true)
#show quote: set pad(x: 5em)
#quote(attribution: [Gerald Salton, IR pioneer])["Information Retrieval is the field concerned with the structure, analysis, organisation, storage, searching and retrieval of information"]

Typically, you interact with IR by entering a query (which expresses what information a user needs) into a specific vertical (different search engines into one, i.e. image search, or news search). It will return documents (which is an ordered list of information according to the relevance of the query) which are ranked by some criteria or model. SERP or search engine results page is the entire page of results.

A search engine retrieves documents according to their relevance to the query. The term document is used to describe the unit of information to retrieve. Documents are not necessarily just text, they could be images or songs as well and are unstructured or semi-structured. Some parts of documents are important than others: 
- document parsers often recognise structure such as markup, HTML, and more.
- head, anchor text, and bold text are all likely to be more important
- links within the document can be used for link analysis.


== Table of Relevance
#figure(image("images/table of relevance.png"), caption: "Table of Relevance Model showing costs of each model (Week 8 Lecture)")<table-relevance>
 
*- Term-based lexical models (Week 3)*
  - #link(<bm25>)[BM25]
  - Boolean Model
  - TF
  - TF-IDF
  - VSM
  - Binary Independence Model

*- Learned-sparse retrievers (Week 6)*
  - #link(<tilde>)[TILDE]
  - #link(<tildev2>)[TILDEv2]

*- Simple Traditional LTR Approaches (like point wise) (Week 7)*

*- Traditional LTR (complex ones like pairwise/likewise) (Week 7?)*

*- Single-vector embedding-based retrievers (Week 5)*
  - #link(<dpr>)[DPR]
  - #link(<ance>)[ANCE]
  - #link(<repbert>)[RepBERT]
  - #link(<contriever>)[Contriever]
  - #link(<e5>)[E5]
  - #link(<llm2vec>)[LLM2Vec]
  - #link(<repllama>)[RepLlama]

*- Multi-vector embedding-based retrievers (Week 5)*
  - #link(<colbert>)[ColBERT]
  
*- Simple Neural Reranker (Monobert, Pointwise LLM-based etc) (Week 8)*

*- More Sofiscatred Neural Reranker (DuoBERT, Pairwise etc) (Week 8)*

= Indexing

== Zipf's Law
Distribution of word frequencies is very skewed a few words occur very often, many words hardly ever occur e.g., two most common words (“the”, “of”) make up about 10% of all word occurrences in text documents

- Observation that rank (r) of a word times its frequency (f) is approximately a constant (k)
- Assuming words are ranked in order of decreasing frequency
  - i.e., r.f \~ k or r.Pr \~ c, where;
    - Pr : probability of word occurrence 
    - c \~ 0.1 coefficient for English
=== Consequences of Zipf’s Law
Words need to be ignored as words that appear extremely infrequently are likely typos etc.,words that appear too often do not help discriminate between documents.

== Tokenizing<tokenisation>
Forming words from sequence of characters

=== Early IR systems:
  - any sequence of alphanumeric characters of length 3 or more
  - terminated by a space or other special character
  - upper-case changed to lower-case

Issues:
- Removal of special characters like hyphens lost information or made URLs, code and tags not function
- Capitals can have different meanings, Apple vs apple
- Numbers that are less than 100 are removed.
- Periods can occur in numbers, abbreviations etc. and would stop tokenizing early

=== Modern Tokenization Process
First step is to use parser to identify appropriate parts of document to tokenize (you may not want to tokenise all fields)
- Defer complex decisions to other components
- word is any sequence of alphanumeric characters, terminated by a space or special character, with everything converted to lower-case
- everything indexed

=== Stopping
Function words, prepositions have little meaning on their own and often have high occurrence frequencies can be removed to reduce index space, improve response time and improve effectiveness. One drawback is that they can be important in combinations, e.g. 'to be or not to be'

=== Stemming
Many words have many morphological variations i.e. plurals and tenses. These words while different have similar meanings. Stemmers attempt to reduce morphological variations of words to a common stem, usually involving removing suffixes. Can be crucial for some languages while small improvements for others, 5-10% in English, 50% in Arabic.

- Porter Stemmer
  - Algorithmic stemmer used in IR experiments since the 70s. Consists of a series of rules designed to the longest possible suffix at each step. Makes a number of errors and difficult to modify.
- Krovetz Stemmer
  - Hybrid algorithmic-dictionary. Word checking dictionary, if present left alone of replace with 'exception', if not present, word is checked for suffixes that could be removed, check again. Lower false positive rate, somewhat higher false negative.

== Inverted Index

#stack(dir: ltr, spacing: 10%)[
  - Has the word and the the docId stored
  - Inverted list (usually ordered in increasing docid)  
  - Can also contain frequency / count or position data.
]

#stack(dir: ltr, spacing: 10%)[
  *Normal*
    - and : {1}
    - aquarium : {3}
    - are : {3, 4}
][
  *Frequency*
    - and : {1:1}
    - aquarium : {3:1}
    - are : {3:1, 4:2}
][
  *Position*
    - and : {1,15}
    - aquarium : {3,4}
    - are : {3:,3} 4,14}
]

With position data you can check phrases, for example, we know 'are aquarium' happens in doc 3 from 3:4.

= Transformers<transformers>

While lexical term based retrievers such as BM25 are used very commonly, there is on issue; they require term overlap. Consider a query: "where is the closest bank", clearly, this query relates does not related to rivers but a term based lexical retriever may associate a high score to a document about "river bank" since the term "bank" appears in both the query and document. Inversely, cases exist where a document that is related to the query but is not retrieved by a lexical model: consider a query "car", a lexical retriever would not fetch documents containing "automobile" as these are not overlapping terms, but have semantically similar meanings.

The key problem:
- Semantic Matching: Understand meaning, not just words - "car" should match "automobile"
- Context Awareness: "Apple" in tech vs fruit context - Query intent understanding

There are many problems with term-based lexical retrievers, for examples they require overlapping terms and do not have the ability to use semantic understanding of query / documents.

== What are transformers?

In 2017 Google published a paper called "Attention is all you need" which outlined the transformer architecture which could understand semantic within language. This architecture originally was purposed for language translation where they considered the semantics by considering the distribution of words in language. Consider the sentence "the cat sat on the mat". @transformer-matrix shows the attention matrix of this sentence, which shows that "the" and "cat" are highly related, and "cat" and "sat" are highly related amongst others. By understanding attention, we can understand what words relate to one another. This multi-head attention is a core part of the transformer architecture.

#figure(image("images/transformer matrix.png"), caption: "Transformer Matrix showing Attention between Words in Sentance (Week 4 Lecture)")<transformer-matrix>

To use a transformer:
- first the text is tokenised (see @tokenisation) into token embeddings
- input embeddings go through the encoder to create encoded representations
- the decoder takes the start token, and adds encoded representation from encoders
- the decoder predicts the next token based on the output of the token probabilities
  - it predicts which token is most likely to be the next word
  - it picks the highest probability token, which becomes the word generated

#figure(image("images/probability.png"), caption: "Transformer Achitecture (Week 4 Lecture)")<transformer-use>


The basic steps of transformer are:

1. Text are tokenised and encoded into input embeddings
2. Input embeddings fo through encoder, and being encoded into encoded representation.
3. Decoder takes \<start> token, and add encoded representation from encoders
4. Decoder predict next token based on output token probabilities

Steps 1 & 2 happen within the encoder, and 3 & 4 in the decoder. Steps 3 & 4 also loop until it is over.

=== Parts of a Transformer
Transformers are made up of two parts, the encoder and the decoder, while they are commonly used together they aren't always, for example BERT is a encoder model.

While in the original transformer architecture these were used together, they do not have to be used together. 

== Encoder
The encoder takes text and encodes it into a rich vector representation. This creates the understanding of context and relationship between the words as described in @transformers. Bi-directional attention means that the encoder can see future words to create this deeper understanding that is require for the rich vector embedding. A sentence such as "how to fix a laptop?" may output a dense vector [0.23, -0.45, 0.67, …]. Applications on an embedding only model is in search or classifications using these embeddings. An example of an encoder only model is BERT.

*Summary*

Purpose:
- Takes input text
- Creates rich vector representation
- Understands context and relationships
- Bidirectional attention (like BERT)
Example:
- Input: How to fix laptop?
- Output: dense vector [0.23, -0.45, 0.67, ...]
Application:
- Search/Classification using embedding

== Decoder
The decoder takes a vector from the encoder and start token, and predicts the next token. The prediction and generation is the purpose of the decoder. It takes the encoded representation, generates the output text step by text, and predicts next token probabilities. Through using masked attention the decoder does not know future tokens and hence, can generate words without bias of knowing what is next. For example, the decoder might take "UQ is" and maybe some encoded vectors and output "a great university". Decoder models are highly applicable and all modern LLMs are decoder only models.

*Summary*

Purpose:
- Takes encoded representations
- Generates output text step by step
- Predicts next token probabilities
- Masked attention (can't see future words)
Example:
- Input: “UQ is”+ encoded vectors (optional)
- Output: "the" → "University" → “of" → ...
Application:
- Translation, Text Generation, all morden LLMs usage.

== Transformer Input Embeddings
=== Token Embedding
Converting words to vectors

Example:

'Information Retrieval' ---_Tokenization_---> ["information", "retrieval"] 

["information", ...]  ---_Embedding lookup_---> Information: [0.1, 0.3, ...] , ...

Also has a lookup table storing learnt term embeddings.
=== Position Embedding
Teaching position awareness

For example "Information retrieval" != "retrieval information"
=== Segment Embedding
When needed for multi-sequence input

Is useful when you have the query and the document being embedded together and allows for them to be represented differently. Special tokens are used to separate the query and the document for this.

==== Special Tokens
- [PAD]: Used for making sure batching text in same length
- [CLS]: Classification task signal
- [SEP]: Separates query from document
- [SEP]: Marks end of input sequence


== Examples of Encoders and Decoders
=== BERT <bert>
BERT or Bidirectional Encoder Representation from Transformers is a bidirectional attention (sees entire sentence at once) encoder only transformer.

BERT uses token, segment and position embeddings to transform a string into a sequence of vectors.

Requires large amounts of text and compute to train, but can then only require a few labelled examples for fine-tuning the model.

When fine-tuning BERT, you can do so for different tasks, i.e. Single-Input Classification, Two-Input Classification, Single-Input Sequence Labelling, and Two-Input Sequence Labelling Tasks.

There are two different BERT models (large and base), large has more parameters and more layers allowing for higher accuracy.

BERT uses in IR:
- As a base for a dense retriever (BERT-based dense retriever)
- A re-ranker (BERT-based) re-ranker

*Limitations of BERT*
- Input Length Constraint:
  - BERT input is constrained to maximum of 512 tokens including special tokens;
  - Can not process long documents directly
  - Requires chunking/truncation for longer texts, but will cause information loss this way
- Cannot Generate Text Auto-regressively
  - Pretraining using MLM requires randomly masking ~15% of tokens
  - Model predicts masked tokens using bidirectional context
  - Cannot generate text token-by-token like GPT
  - This sampling makes it impossible to generate coherent sequences naturally

=== Auto-regressive Decoders
In an auto-regressive text generation model, at each time step t, the model takes a sequence of token ${y}_(<t)$ as input , and outputs a new token $hat(y)_t$

==== GPT-3 / ChatGPT
- Decoder-only transformer
- Unidirectional attention (auto-regressive)

Undergoes in-context learning / next token prediction (pre-training), gets given sequences showing what comes next, for example 5 + 8 = 13, gaot => goat, thanks => merci.

Then instruction-follow fine-tuning happens where you tain the LLM on instruction-response pairs, which teaches the model to follow human instructions, for example "Summarise this text [text]" -> [summary]"

Decoder uses in IR:
- Prompting LLMs for retrieval and ranking
- RAG (retrieval augmented generation)
- Model-based IR (LLM as indexer and retriever)

= Term Based Lexical Models

== BM25 (Best Match 25) <bm25>
=== BM25 Ranking Function

BM25 is a popular and effective ranking algorithm based on the binary independence model.

It expands upon the tf-idf idea (under its most general form with no relevance information provided), with two main differences:
- Term frequency saturation
- Document length normalisation

Empirically, it has been shown to be a quite reliable and robust model, which works well out-of-the-box in most situations.  
It is also the default retrieval model in many open-source search engines such as Lucene and Elasticsearch.

For each subsequent section, we will separate the BM25 formula into its components and analyse them.  
The full equation is as follows:

$sum_(i=1)^(|Q|) log ((r_i+0.5)/(R-r_i+0.5)/((n_i-r_i+0.5)/(N-n_i-R+r_i+0.5)) dot ((k_1+1)f_i)/(k_1B+f_i) dot ((k_2+1)q f_i)/(k_2 + q f_i))$


=== Components of BM25

==== RSJ Weight

$log ((r_i+0.5)/(R-r_i+0.5)/((n_i-r_i+0.5)/(N-n_i-R+r_i+0.5)))$

This component is known as the *Robertson–Sparck Jones (RSJ) weight*.

It considers knowledge about the number of relevant documents ($R$) and the number of relevant documents that contain term $i$ ($r_i$).  
$N$ and $n_i$ refer to the number of documents that have been judged to obtain $R$ and $r_i$.  
Thus, if no relevance information is provided, it becomes:

$log (N-n_i + 0.5)/(n_i+0.5)$

which is an approximation of the classical IDF.

==== Saturation Component

$((k_1+1)f_i)/(k_1B+f_i)$

This is known as the *term saturation component*.

The contribution of the occurrence of a term to a document score cannot exceed a saturation point.  
The parameter $k_1$ controls the saturation, where $k_1 > 0$:
- High $k_1$ → $f_i$ contributes significantly to the score.
- Low $k_1$ → additional contribution of further term occurrences tails off quickly.

Typically, $k_1$ is set to 1.2.

=== Effect of Term Frequency in BM25

Consider the query “president lincoln.”
Assume document length is constant and equal to 90% of the average document length, with $N = 500000$ documents.

- President occurs in 40,000 documents ($n_1 = 40000$).  
- Lincoln occurs in 300 documents ($n_2 = 300$).

==== Document Length Normalisation

$B = (1-b) + b dot (d l)/(a v d l)$

This is known as the document length normalisation component.

An author may write a shorter or longer document than the average.  
The reason could be:
- Verbosity → prefer shorter documents.
- Scope → prefer longer documents.

This component provides a soft normalisation of document length.

The parameter $b$ controls the level of normalisation ($0 <= b <= 1$):
- $b = 1$ → full normalisation  
- $b = 0$ → no normalisation  

Typically, $b = 0.75$.  
$B$ is used to normalise the term frequency $f_i$.

==== Query Component
// TODO format equations like this everywherein the doc
#figure([
  #v(10pt)
  #set text(12pt)
  $((k_2+1)q f_i)/(k_2 + q f_i)$ 
  #v(10pt)
], kind: "Equation", supplement: "Equation", caption: [Within Query Component])


This is known as the within-query component.

It is useful for longer queries where a term may occur multiple times.  
It introduces similar saturation behaviour as the within-document component and has its own constant $k_2$ (sometimes referred to as $k_3$).  

Experiments suggest that this term is not particularly important — meaning multiple occurrences of a term in a query can often be treated as separate terms.

= Embedding -based retrieval models (Dense Retrievers)
Based on transformer encoders:
- Single Vector based: DPR, ANCE, RepBERT, Contriever
- Multi Vector based: ColBERT

Based on Transformer Decoders:
- Single Vector based: E5-mistral, LLM2Vec, RepLlama

Embed document d or query q into a "low" dimensional vector and compute the similarity between the vector q and that of d, which ideally measures how relevant q and d are to each other.

For encoder only, the encoder encodes the text into vectors making one vector pre token.

DPR and ANCE use the [CLS] token vector to represent, RepBERT and Contriever use the mean vector to represent, and ColBERT use multiple vectors to represent text.

Sinlge Vector, i.e. DPR, ANCE, RepBERT and Contriever is simply just comparing the one vector from query and the one vector from the text and comparing it, whereas the multi vector based approach is taking every token from the query and comparing it to every token from the passage, and summing / averaging those together for the final score.

Decoder only, the decoder predict the next token given input, but we can discard the next token prediction part and instead we get the vector of the last token, we can then use this vector to embed / represent text.

*Summary*

Encoder-based approaches:
- Available for either Single-vector-based: DPR, ANCE, RepBERT, Contriever; or multi-vector-based: ColBERT.
- Uses transformer-encoders as backbone architecture.

Decoder-based approaches:
- LLM-family models (E5-mistral, LLM2Vec, RepLlama)
- Uses transformer-decoders as backbone architecture.
- Originally for generation, adapted for retrieval

== Encoder Only
=== DPR <dpr>
Uses the CLS token embedding to represent a query or a document

*Similarity function:* Dot product

*Training (fine-tuning) Loss:* Negative log likelihood (InfoNCE loss)

$
L = -log ((e^("sim"(q,p^+)))/((e^("sim"(q,p^+))) + Sigma^n_(j=1)(e^("sim"(q,p^-_j))))) \
"where" \
q = "query embedding" \
p^+ = "positive (relevant) document embedding" \
p^- = "negative (irrelevant) document embeddings" \
"sim"(q, p) = q dot p = "dot product similarity"
$

*Negative Sampling for fine-tuning:* random sampling + BM25 hard negatives + In batch negatives (for efficiency)

=== RepBERT <repbert>
Mean of token embeddings

*Similarity function:* Inner product

*Loss:* Multi-Label Margin Loss

$
1/n dot Sigma_(1<=i<=m, m < j <=n) max(0, 1 - ("Rel"(q, d_i^+) - "Rel"(q, d_j^-)))
$

*Negative Sampling for fine-tuning:* In batch negatives
=== ANCE <ance>
Uses the CLS token embedding to represent a query or a document

*Similarity function:* Inner product

*Training (fine-tuning) Loss:* Negative log likelihood (InfoNCE loss)

$
L = -log ((e^("sim"(q,p^+)))/((e^("sim"(q,p^+))) + Sigma^n_(j=1)(e^("sim"(q,p^-_j))))) \
"where" \
q = "query embedding" \
p^+ = "positive (relevant) document embedding" \
p^- = "negative (irrelevant) document embeddings" \
"sim"(q, p) = q dot p = "dot product similarity"
$

*Negative Sampling for fine-tuning:* Negative Contrastive Estimation
- Choose hard negatives (for model) with regular checkpoints
=== Contriever <contriever>
Mean of token embeddings

*Similarity function:* Cosine Similarity

*Loss:* Negative log likelihood (InfoNCE loss)

$
L = -log ((e^("sim"(q,p^+)))/((e^("sim"(q,p^+))) + Sigma^n_(j=1)(e^("sim"(q,p^-_j))))) \
"where" \
q = "query embedding" \
p^+ = "positive (relevant) document embedding" \
p^- = "negative (irrelevant) document embeddings" \
"sim"(q, p) = q dot p = "dot product similarity"
$

*Innovation:* Unsupervised training (No labeled data needed)
=== ColBERT <colbert>
All token embeddings

*Similarity function:* MaxSim (max similarity between each query token and one of the document tokens and then summed all of those together)

*Loss:* negative log likelihood (InfoNCE loss)

$
L = -log ((e^("sim"(q,p^+)))/((e^("sim"(q,p^+))) + Sigma^n_(j=1)(e^("sim"(q,p^-_j))))) \
"where" \
q = "query embedding" \
p^+ = "positive (relevant) document embedding" \
p^- = "negative (irrelevant) document embeddings" \
"sim"(q, p) = q dot p = "dot product similarity"
$

*Innovation:* 
- Compute fine-grained interactions between ALL token vector pairs of query and document- Aggregate via MaxSim operations.
- As it captures richer amount of information; it can be more effective than other single-vector based retrievers.

*Problems:*
- Massive storage overhead: ~100-500 vectors per document vs 1 vector for bi-encoders need to be stored.
- Scoring complexity and inefficiency:
  - O(m×n) vs O(1) for bi-encoders
  - m = query length, n = document length
  - More expensive similarity computation (MaxSim operations)
  
== Decoder Only
=== E5 with LLMs <e5>
Originally E5 embedding are BERT-based, new E5 embedding are decoder-only LLM.

E5-mistral uses LLM in two ways:
- Synthetic data generation: 
  + prompt LLM to "brainstorm" a list of potential retrieval tasks 
  + generate \<query, positive, hard negative? for each task.
- Embedding backbone: 
  + modify query to: q+ inst = Instruct: {task_definition} \\n Query: {q+} 
  + append [EOS] token to end of query and document feed them into LLM to get embeddings by taking last layer of [EOS] vector

Training in E5Mistral:
- Contrasive pre-training: weakly supervised data from filtered text pairs
- Contrasive fine-tuning: LLM generated synthetic data + MS MARCO labelled data

Observation: contrastive pre-training has negligible impact on model quality
- extensive auto-regressive pre-training enables LLMs to acquire good text representations, and only fine-tuning required to obtain effective embeddings.

=== LLM2Vec <llm2vec>
Transform any decoder-only LLM into a strong text encoder.

Three key steps:
+ enable bidrectional attention,
+ masked next token prediction,
+ unsupervised contrastive learning
- 4. can add supervised contrastive learning
=== RepLlama <repllama>
Same as LLM2Vec but uses Llama instead of GPT.

[EOS] token embeddings

*Similarity function:* Dot Product

*Loss:* negaticve log likelihood (InfoNCE)

$
L = -log ((e^("sim"(q,p^+)))/((e^("sim"(q,p^+))) + Sigma^n_(j=1)(e^("sim"(q,p^-_j))))) \
"where" \
q = "query embedding" \
p^+ = "positive (relevant) document embedding" \
p^- = "negative (irrelevant) document embeddings" \
"sim"(q, p) = q dot p = "dot product similarity"
$

Train an Lora adaptor instead of the full model for training-time efficiency.

== Learned Sparse Retrievers
The main point behind learned sparse retrievers is that they combine the efficiency of lexical models and the effectiveness of neural approaches.

The main benefit of lexical models is that they only care about term overlap and hence, only go through documents which contain the query terms, these models have weighting from solely overlapping terms and defined formulas. Whereas, dense embedding-based retrievers are much more effective as they semantically match documents and query.

Learned sparse models in contrast learn optimal term weights from training and use transformers to predict term importance, thereby improving high effectiveness, but maintain sparsity for efficiency.

What is sparsity?
Sparsity is the fact that sparse models output vectors across the vocabulary but only have a small subset of terms learned to be non-zero.

Consider
```
Query: "gene therapy risks"
Sparse vector (V=50k terms):

gene: 1.7
therapy: 2.3
risk: 1.8
side: 0     ← zero
football: 0 ← zero
...
```

In this case the model has learned that gene, therapy, risk are all important terms, whereas the remaining 50,000 terms may be zero. If a model had all non-zero weights, a dense vector would be required as opposed a traditional vector.

Often sparse models use L1 and RELu (SPLADE/TILDEv2/TILDE) or explicit clamping to ensure this never occurs. This is why learned sparse retrievers are mode efficient as they use inverted index where each posting list has only a few terms from each doc.

Generally, learned sparse retrievers are less effective than dense retrievers but are more efficient.

=== TILDE <tilde>
==== High level
Learned spare model:
- learn optimal term weights from training data
- use powerful transformer to predict term importance and high effectiveness
- maintain sparsity of efficiency

First way to do it:
- We remove the query encoding and use only a tokeniser to create term-based lexical sparse vector for query; but keep the dense vector for the document

Document indexing and encoding time still the same as a dense model, but the query does not need to be encoded only tokenised so the query encoding time is substantially lower. The query vector is sparse so scoring cost is substantially lower.

==== More Details

TILDE or Term independent likelihood model uses BERT tokeniser to obtain sparse query encoding. It uses CLS token to encode documents and project CLS token embeddings to |V| vector. It creates an inner produce between the sparse query vector and document vector in the |V| space.

It still uses a bi-encoder architecture and BERT is used to precompute the document representations offline.

Each element in the sparse vector is a token appearance in the query. Since the query is a simple tokeniser we can encode without a GPU.

==== Loss function

The training loss function for TILDE is bi-directional query-document likelihood loss (BiQDL). This assumes that query tokens are independent. It is formed of two likelihood components: the document likelihood and the query likelihood.

#align(center)[
  $cal(L)_(B i Q D L) (D) = (cal(L)_(Q L)(D) + cal(L)_(D L)(D))/2$
]

Where: \
  $cal(L)_(Q L) (D) = - sum_(q,d in D) 1/(|V|) sum_i^(|v|)y log (P_theta (t_i | d)) + (1-y) log (1-P_theta (t_i|d))$

  $cal(L)_(D L) (D) = - sum_(q,d in D) 1/(|V|) sum_i^(|v|)y log (P_theta (t_i | q)) + (1-y) log (1-P_theta (t_i|q))$

==== Ranking function

TILDE can rank passages based on query likelihood alone: \ \
$"TILDE-QL"(q|d^k) = sum_i^(|q|) log(P_theta (q_i|d^k))$

TILDE can rank passages based on document likelihood only (TILDE-DL): \ \
$"TILDE-DL"(d^k|q) = 1/(|d^k|)sum_i^(|q|) log(P_theta (d^k|q_i))$

TILDE can rank passages based on query and document
likelihood (TILDE-QDL): \ \
$"TILDE-QDL"(q, d^k) = alpha dot "TILDE-QL"(q|d^k) + (1 - alpha) "TILDE-DL"(d^k|q)$

=== TILDEv2 <tildev2>

TILDEv2 expands upon TILDE with contextualised exact term matching and passage expansion. It uses the BERT tokeniser to obtain sparse query encoding, and uses BERT token embeddings for exact term matching.

TILDEv2's main innovation was expanding the document representations with docT5query (better doc2query) or TILDE. This improves the document representations and overcomes document vocabulary mismatches.

How is TILDE used in TILDEv2? TILDE produces a distribution over the query vocabulary representing how likely each token is to appear in a query given a passage. In other words, TILDE predicts which query tokens are most likely to be associated with it. This effectively is an estimation of the term importance in the passage context.

For document expansion, TILDE sorts this likelihood distribution in descending order and examines the top-m (usually 128, 150, or 200) ranked tokens based on their likelihood. Any token that is not already in the passage and not a predefined stop word is appended to the passage.

So why TILDE over docT5query (doc2query): it's quicker to train and cheaper (7.33 hours vs 320 hours for docT5query) and achieves similar performance. 

=== SPLADE


Given a query or document; if we can predict the importance of highly related terms in the vocabulary for its contribution to a ranking task; we could create a better lexical term-based retriever.

We can use a transformer encoder to do this.

Remember for the pertaining tasks of BERT (transformer encoder); it tried to predict [MASK] token given all the input text around it?

We get a probability of possible tokens from this MLM head and use that aggregate probability as the weight.

#align(center)[
#image("images/mlm head prediction.png", width: 50%)
]

However, this process creates a dense vector representation and hence, we ensure that the majority of the term weights are zero to maintain the sparsity.

We use a Relu activation function, FLOPS regularisation, and L1 regularisation to keep the sparsity, this keeps most term weights at zero and only keeps important terms active.

=== SPLADEv2

#table(columns: (0.5fr, 1fr, 1fr))[Model][*SPLADEv1*][*SPLADEv2*][Pooling][Sum pooling to aggregate token representations][Max pooling to aggregate token representations][Training Loss ][Basic contrastive training loss][Distillation training loss][Weighting and expansion][Query and Doc token weighting and expansion][Only doc token weighting and expansion (simiilar to TILDE)]


=== doc2query
Query / document expansion augments the user query or document in the corpus with additional relevant terms.

A variety of automatic or semi-automatic query/ document expansion techniques have been developed
  - goal is to improve effectiveness by matching related terms
  - sem-automatic techniques require user interaction to select best expansion terms

  Many Lexical-based retrieval models with learned term weight rely on query / document expansion for better effectiveness.

Query / Document Rewriting is a related technique.

doc2query is supervised training where paris of query and relevant document are made into a predicted query where that document is relevant.

For example, if the input document talks about cinnamon reducing blood sugar levels, the predicted query would be along the lines of "dpes cinnamon lower blood sugar?"

This is the added on to the document as the 'expanded document' which is the document and then the possible query added on to it. In practice, 5 - 40 queries are sampled with top-k or nucleus sampling. 

This gives significantly better results in most cases.

=== PromptReps

E5 gets embedding from last layer of [EOS] token; requires (supervised) contrastive fine-tuning

LLM2Vec gets embeddings from mean pooling of sequence tokens; requires (unsupervised/supervised) contrastive fine-tuning.

Can we engineer LLMs prompt to obtain an effective representations without the need for constrastive fine-tuning?

PromptReps is a "zero shot" generation of representation using LLMs.

PromptReps prompts the LLM to represent given token (doc/query) using one word. It uses the last token hidden layer to obtain a dense representation and uses the logits associated with the last token hidden layer to obtain a sparse representation over the LLM vocabulary. By combining the two of these it creates a hybrid retriever.

#align(center)[
  #image("images/promptreps performance.png", width: 70%)
]

= Rerankers

== Learning to rank
=== Offline Learning to Rank
Definition: Learning to rank models trained on pre-collected labelled datasets

Three main approaches:
- Pointwise: Treats ranking as regression / classification (predict relevance score for each document independently)
- Pairwise: Learns relative preferences between document pairs
- Listwise: Optimises the entire ranking list as a whole

Characteristics:
- Uses human-labelled relevance judgments
- Training happens before deployment
- No real-time user feedback during training
- Foundation for modern ranking systems

=== Online Learning to Rank
Defintion: Learning to rank models that adapt based on real-time user interactions

Key difference to offline: Use implicit feedback (clicks) instead of explicit labels

Main approaches:
  - Dueling Bandit Gradient Descent (DBGD): Compares ranking functions through user interactions
  - Counterfactual Onlein Learning to Rank (COLTR): Learns fro mbiased click data

Advantages:
- Adapts to changing user preferences
- Uses abundant click data
- Continuously improves with user feedback

=== Traditional Retrieval Methods
Traditional ranking functions in IR use a very small number of features:
  - Term Frequency
  - Inverse document frequency
  - Document length
  - Priors (e.g. PageRank)

It is easily possible to tune weighting coefficients by hand

But many more features are available, for example
- Log frequency of query word in anchor text
- Query word in color on page
- \# of images on page
- \# of (out) links on page
- PageRank of page
- URL length
- URL contains “~”?
- Page edit recency
- Page loading speed
- Arbitrary useful features – not a single unified model
- CoordinationMatch, VSM, BM25, Language model
- Spam score
- URL depth
- Number of query terms that match document d

== The basics
Model ranking as a linear model to be learnt:
$
"Score"(q, d) = sum w_i times f_i(q, d) = f(x, w)
$

$f_i(q,d)$ is a feature that typically describes a relationship between q and d

$ f_1 = "BM25" (q, d), f_2 = "LM" (q, d), f_3 = "Pagerank(..., d)" $

$w_i$ is the weight for feature $f_i(q, d)$

$x$ is a vector of features, w is a vector of weights

Treat each type of evidence as a feature
  - VSM, BM25, Indri, PageRank, url depth, ...

Use a machine learning model to combine evidence

LTR or LeToR (Learning to rank) are hypermodels built upon prior retrieval models.

Similar to other machine learning tasks,
- Given training data, X → Y, learn a model $Y = f(X,w)$;
- For new data $y$, apply the model to get $y = f(x, w)$;
	- y: desired value, x: feature vector, w: weight vector
 
*LTR is Supervised Learning*

Classification: Find $f: X arrow.r Y; Y in {-1, 1}$
- Naive Bayes, SVM

Regression: Find: $f: X arrow Y; Y in R$
- e.g. Linear regression

The IR ranking task
- Finding the best ranking of given documents
- Usually done by finding ranking scores: $f: X arrow Y; Y in R$
	- But we don’t really care about the scores, only the order

=== Where does ML & learning to rank fit in a large search engine?
- Query classification to determine the type of query
  - Perhaps use a template to form a structure query
  - Make decisions about how the query will be scored
- Retrieval is done by a sequence of retrieval models
  - Exac-match Boolean -> Form a set of documents
  - BM25 -> Re-rank the set, select the top n
  - Learned models -> Re-rank the top n, select the top k

The search architecture uses cascading architectture of rankers where each layer ranks, prunes and returns results. Layered evaluation gives control over search costs. This allows simpler models to be applied to massive data and sophisticated models to deal with little data, boosting efficiencies while still giving a good ranking.

LTR methods are described along three main dimensions
- The document representation (the features)
- The type (or style) of training data
- The machine learning algorithm

For a query q, model document d as feature vector x. It uses training date to learn a model h(x), h(x) generates a real-valued score that is used to rank d for q, where q and d are implicit in the feature values.

*Using search data to train a learning algorithm*
Binary assessments, either relevant or not relevant, document scores as an initial rank, preferences of users.

== Approaches to LTR
Three main LTR approaches

- Pointwise
- Pairwise
- Listwise

The similarities between them are that each uses a trained model h to estimate the score of x, where x describes how well document d matches query q, and the documents are then ranked by the score h(x).

They differ by the approaches to training the model h and the different types of training data used.

=== Pointwise LTR
Map items of a certain relevance rank to a subinterval
- this is a regression problem (though could simplify to classification)

*Approach:* Train a model using individual documents
- Training data: x -> score
- Learned model: h (x) -> score

*What is the document score?*
- Binary assessments { -1, +1 }, { not relevant, relevant }
- Graded assessments { 4, 3, 2, 1, 0 }, { perfect, excellent, very good, good, poor }

*Training Data*
#table(columns: (0.3fr, 0.1fr, 0.3fr))[Relevance][$arrow$][Training Data][$d_1$, relevant][$arrow$][$d_1$, 1][$d_2$, not relevant][$arrow$][$d_2$, -1][$d_3$, not relevant][$arrow$][$d_3$, -1]

Directly apply existing learning algorithms on ranking data
  - Given a document, predict the relevance label

Regression (e.g. linear regression)
  - Scores are { -1, +1 } or  { 4, 3, 2, 1, 0 }

Classification (e.g. SVM)
  - Categories are { -1, +1 } or { 4, 3, 2, 1, 0 }

*Problems with Pointwise LTR*

Score-learning approaches (e.g. regression) assume that training data has desired scores for each document
Usually relevance scores are arbitary values
  - { -1, +1 } or { 4, 3, 2, 1, 0 }
  - They could just as easily be { 0, 1 } or  { 189, 57, 42, 16, 1 }
The goal of IR though is a ranking of document not the scores, as as longa s the order of relevance is correct the scores are unimportant

=== Pairwise LTR
Input is a pair of results for a query, and the class is the relevance ordering relationship between them
- i.e. aim is to classify instance pairs as correctly ranked or incorrectly ranked: turns an ordinal regression problem back into a binary classification problem

*Approach:* Train a model using pairs of documents
- Training data: prefer (x1, x2)
- Learned model: h (x1) > h (x2)

*What is the pair value?* Binary assessments { >, < }

*Loss function:* 0 if a pair is ordered correctly, otherwise 1

Minimise the number of misclassified document pairs: focus on preference, not raw scores/labels

*Training Data*
#table(columns: (0.3fr, 0.1fr, 0.3fr))[Relevance][$arrow$][Training Data][$d_1$, relevant][$arrow$][$d_1$, $d_2$, $>$][$d_2$, not relevant][$arrow$][$d_2$, $d_1$, $<$][$d_3$, not relevant][$arrow$][$d_1$, $d_3$, $>$][][$arrow$][$d_3$, $d_1$, $<$]

*Problems with Pairwise LTR*
Queries with an even balance of relevant and non-relevant documents dominate the training data
- Number of pairs = $|R| times |"NR"|$
  - $q_1 = 5R times 5"NR" = 25"pairs" $
  - $q_2 = 9R times 1"NR" = 9"pairs" $
  - $q_3 = 2R times 8"NR" = 16"pairs" $
- The pairwise approach is more sensitive to noisy labels
- Usually the ranking position is ignored (as with the pointwise approaches)

=== Listwise LTR
Train a model using a list of documents
• Training data: $x_1 > x_2 > ... > x_n$
• Learned model: $h (x_1) > h (x_2) > ...$

What is the value of a particular ranking?
• Some metric over the ranking
• NDCG\@n, with n=1, 3, 5, 10, …
• MAP\@n
The goal is to maximize the value of the metric
• Directly align the model with the ranking target

*Training Data*
#table(columns: (0.3fr, 0.1fr, 0.3fr))[Binary Relevance][$arrow$][Multi-Valued Relevance][$d_1$, relevant][$arrow$][$d_4$, perfect][$d_4$, relevant][$arrow$][$d_1$, excellent][$d_3$, not relevant][$arrow$][$d_3$, poor]

*Problems with Listwise LTR*
Directly optimizing some metrics is hard
- Popular metrics (NDCG\@n) are not continous, nor convex; This very hard for optimization algorithms

Two common strategies in Listwise approaches:
+ Find another metric that is intuitive and easy to optimise
  - e.g. Likelihood of 'best' rankings in training data
+ Directly optimise ranking evaluation metrics, with approximation.

==== Listwise LTR with MLE

Intuition:

- Construct the probably of a ranking p ( x 1 , x 2 , … , x n )
- Find the model h ( x ) that maximises the probability (likelihood - MLE) of best rankings in training data
- Use h x in ranking

The key steps is construction p ( x 1 , x 2 , … , x n )

- It is impossible to define p ( x 1 , x 2 , … , x n )
  - The sample space is all possible ranks: n!
- ListMLE defines a generative process with much smaller space
  - With the help of independence assumptions

Direction Optimisation of a Metric in Listwise LTR

It would be natural to directly the metric of interest

- However, it may be difficult to do
- Some metrics are not continuous or differentiable
- Example: Metrics that depend upon rank position

Simpler possibilities

- Optimise an approximation of the metric
- Bound the objective function
- Optimise the objective function directly (a good result is not guaranteed)

=== LTR Summary
- Pointwise is the weakest of the three approaches
- Pairwise and listwise are about equally useful
  - Relative effectiveness: Listwise ≈ Pairwise > Pointwise
- Pairwise has an imperfect learning target, but is easy to achieve
  - Minimizes pairwise errors, but we want the best ranking
  - A simpler learning problem with theoretical guarantees
- Listwise has a perfect learning target, but is harder to achieve
  - Learning target is exactly the same as what we want
  - A harder learning problem
  - Listwise algorithms may be more effective (eventually), but there are fewer off-the-shelf solution

==== Limitations of LTR
- Most LTR methods produce linear models over features
- This contrasts with most of the clever ideas of traditional IR, which are nonlinear scaling and combinations of basic measurements
  - log term frequency, idf, tf.idf, pivoted length normalisation
- At present, ML is good at weighting features, but not as good at coming up with nonlinear scalings
  - Designing the basic features that give good signals for ranking remains the domain of human creativity
  - Or maybe we can do it with deep learning
- Much of the LTW literature uses lots of training data
- Research is driven by web companies that have a lot od data
  - But... you may not have a lot of data
  - Their conclusion smay not apply to your situation
- Use wth caution
  - Start with a small set of high-quality features then grow it
  - Need to manually design features carefully
- So far we have discussed the application o LTR to an offline setting
  - A training dataset is used, annotated with relevance labels (explicit relevance signal)
  - There are expensive to create: lots of editorial effort
  - Unethical to create in privacy-sensitive settings, e.g. email search
  - Difficult for small-scale problems, e.g. personalisation
  - User preferences change over time, e.g. because of temporal aspects query "powerball" in 2018 vs 2019 or "uss carl vinson" in Gulf War vs recent North Korea deployment
  - Not necessarily aligned with user preferences, e.g. annotators disagree, why wouldn't users

*Directly learning fro user interactions solves the problem of annotations.*
- Interactions such as clicks ar every easy to collect.
- User behaviour is indicative of their preferences.

*Problems with users interactions*
Consider clicks on search results
- Noise: users often click for unexpected reasons. Also there may be malicious behaviour
- Bias: some documents are more likely to be clicked for other reasons.

Methods to deal with these interactions should be robust to noise and bias: unbiased learning to rank.

- Position Bias
  - Documents placed higher up in the ranking are more likely to be considered
- Selection Bias
  - Users will only click on documents you present them
  - If you are not shown a result, you will never be able to click on it
- Presentation Bias
  - Attractive snippets may receive more clicks
    - Information Scent
    - Click baits

*How to learn a ranker from implicit feedback*
+ Learning from user preferences
  - Infer pairwise preferences between document from clicks in historical interaction logs and optimise a model to predict them correctly.
  - Does not deal with selection bias, only minimally with position bias.
  - It is performed offline, so does not adapt to changes in intent, unless batching updates every t-time intervals
+ Counterfactual learning
  - Offline
  - Pointwise LTR that accounts for position bias assumes the position bias is known or learned and independent from displayed documents.
    - Note, other LTR methods could be used, e.g. listwise
  - Estimating the positon bias from interaction data, active area of research
+ Online learning to rank (OLTR)
  - Infer preferences between rankers / documents from clicks as they occcur
  - OLTR methods have control over what to displau to the user.
    - So, the ranking is learnt in an online manner, not offline
  - For each user query, they perform two actions:
    + Decide what results to display to user
    + Learn from user interactions with chosen results
  - Different ways to de-bias the signal (some don't)
  
== Neural and LLM Rerankers

We can use either encoder or decoder based transformers as rerankers. To do so:

=== Encoder 

- Encoder encodes text into vectors
- It can split different parts of the text using [SEP]
- [CLS] token during BERT pretraining was used for next sentence prediction
- Split query and doc using [SEP], then use a softmax function to classify whether a document is relevant or irrelevant to query.

$s_i = "softmax"(T_"[CLS]" W+ b)_1$

=== Decoder

Decoder predicts next token given input

We can use it to know if documents is relevant to query or not

LLM-based reranking techniques: pointwise, pairwise, listwise, setwise

#image("images/transformer decoder only.png")

=== Using BERT for ranking

- Need to fine-tune BERT for the ranking task
  - relevance classification
  - monoBERT

==== Training monoBERT: loss function

$L = sum_(j in J_"pos") log(s_j) - sum_(j in J_"neg") log(1-s_j)$

Common practice:
- For positive judgement (relevant), use human label
- For negative judgement (negative), sample from BM25