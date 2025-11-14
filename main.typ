#import "@preview/hydra:0.6.2": hydra
#set text(font: "Noto Sans")
#set text(lang: "en", region: "au")
#[
  #set text(size: 50pt)
  INFS7410 Revision
]
#set text(9pt)
#show image: it => {
  align(center, it)
}

// )
Foreword:

These are notes written by a web-developer who has no idea on web-indexing, read at your own risk. If you have thoughts comment on this document. Input is welcome! 

#pagebreak()
#set page(numbering: "1")
#set page(
  number-align: center,
)
#outline()
#set page(
  header: [#context(emph(hydra(1))) #h(1fr) INFS7410],
  number-align: center,
)
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
 
* Term-based lexical models (Week 3)*
  - #link(<bm25>)[BM25]
  - Boolean Model
  - TF
  - TF-IDF
  - VSM
  - Binary Independence Model

* Learned-sparse retrievers (Week 6)*
  - #link(<tilde>)[TILDE]
  - #link(<tildev2>)[TILDEv2]

* Simple Traditional LTR Approaches (like point wise) (Week 7)*

* Traditional LTR (complex ones like pairwise/likewise) (Week 7?)*

* Single-vector embedding-based retrievers (Week 5)*
  - #link(<dpr>)[DPR]
  - #link(<ance>)[ANCE]
  - #link(<repbert>)[RepBERT]
  - #link(<contriever>)[Contriever]
  - #link(<e5>)[E5]
  - #link(<llm2vec>)[LLM2Vec]
  - #link(<repllama>)[RepLlama]

* Multi-vector embedding-based retrievers (Week 5)*
  - #link(<colbert>)[ColBERT]
  
* Simple Neural Reranker (Monobert, Pointwise LLM-based etc) (Week 8)*

* More Sofiscatred Neural Reranker (DuoBERT, Pairwise etc) (Week 8)*

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

#figure(image("images/transformer matrix.png",  width: 70%), caption: "Transformer Matrix showing Attention between Words in Sentance (Week 4 Lecture)")<transformer-matrix>

To use a transformer:
- first the text is tokenised (see @tokenisation) into token embeddings
- input embeddings go through the encoder to create encoded representations
- the decoder takes the start token, and adds encoded representation from encoders
- the decoder predicts the next token based on the output of the token probabilities
  - it predicts which token is most likely to be the next word
  - it picks the highest probability token, which becomes the word generated

#figure(image("images/probability.png",  width: 70%), caption: "Transformer Achitecture (Week 4 Lecture)")<transformer-use>


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
- Output: "the" $arrow$ "University" $arrow$ “of" $arrow$ ...
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
- High $k_1$ $arrow$ $f_i$ contributes significantly to the score.
- Low $k_1$ $arrow$ additional contribution of further term occurrences tails off quickly.

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
- Verbosity $arrow$ prefer shorter documents.
- Scope $arrow$ prefer longer documents.

This component provides a soft normalisation of document length.

The parameter $b$ controls the level of normalisation ($0 <= b <= 1$):
- $b = 1$ $arrow$ full normalisation  
- $b = 0$ $arrow$ no normalisation  

Typically, $b = 0.75$.  
$B$ is used to normalise the term frequency $f_i$.

==== Query Component
// TODO format equations like this everywherein the doc
#figure([
  #v(10pt)
  $((k_2+1)q f_i)/(k_2 + q f_i)$ 
  #v(10pt)
], kind: "Equation", supplement: "Equation", caption: [Within Query Component])


This is known as the within-query component.

It is useful for longer queries where a term may occur multiple times.  
It introduces similar saturation behaviour as the within-document component and has its own constant $k_2$ (sometimes referred to as $k_3$).  

Experiments suggest that this term is not particularly important — meaning multiple occurrences of a term in a query can often be treated as separate terms.

= Embedding-based retrieval models (Dense Retrievers)
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
- Given training data, X $arrow$ Y, learn a model $Y = f(X,w)$;
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

=== Evaluation with full information
We know the true relevance labels y for all candidate documents.

$
Delta (f_0, D, y) = sum_(d_i in D) lambda("rank"(d_i | f_0, D)) dot y(d_i) \
"where rank is the rank position of document " d_i "ranked by " f_0 "and " lambda "is a rank weighting function:" \

"Average Relevant Position (ARP)": lambda(r) = r  \
"Discount Cumulative Gain (DCG)": lambda(r) = 1/(log 2 (1+r)) \
"Percision at k (Prec@k):" Delta(R) = (1[r<=k])/k
$

#image("images/LTR evaluation example.png")

=== Evaluation with partial information
We often do not know the true relevant labels $y(d_i)$, only have implicit feedbacks from users, e.g., clicks

A missing click does not indicate non-relevance, and a click is a biased and noisy indicator of relevance.

We can compute the chance that a document is clicked on by

$
P(c_i = 1 ∧ o_i = 1 | y(d_i)) = P(c_i = 1 | o_i = 1,y(d_i)) ⋅ P(o_i = 1 | i)
$

==== Naive estimator
A naive way to estimate a ranking mode l is to assume clicks are relevance signals: 

$
Delta_"NAIVE"(f_0, D, c) = sum_(d_i in D) lambda("rank"(d_i) | f_0, D)) dot c_i
$

Even with no click noise, this estimator is biased by the examination probabilities

== Unbiased Estimator
Counterfactual evaluation accounts for position bias using Inverse propensity Scoring (IPS);

$
Delta_"IPS"(f_0, D, c) = sum_(d_i in D) (lambda("rank"(d_i | f_0, D)))/(P(o_i = 1)) dot c_i
$

This ends up being a decent estimation of the full information evaluation

== Online Evaluation
=== Balanced Interleaving
+ randomly choose one of the rankers to begin
+ then the rankers take turns:
  + chosen ranker places its next document unless it has already been placed
  + turn goes to the other ranker
  + repeat step i) until k documents are placed
+ display resulting interleaving to user, observe clicks

#image("images/balanced interleaving.png")

=== Inferring preferences from clicks
+ Determine the clicked document with the lowest displayed rank: dmax
+ Take the highest rank for dmax over the two rankers: imin
+ Count the clicked documents for each ranker at imin or above.
+ The ranker with the most clicks is preferred.

=== Dueling Bandit Gradient Descent
If online evaluation can tell us if a ranker is better than another, then we can use it to find an improvement of our system.

By sampling model variants and comparing them with interleaving, the gradient of a model w.r.t. user satisfaction can be estimated.

+ Start with the current ranker with feature vector $theta$.
+ for i to inf (i.e. lifetime of the search engine) repeat:
  + Sample a random unit vector $u$ from the unit sphere ($|u| = 1$)
  + Compute the candidate ranker $theta prime = theta + u$
  + Get the result list of $theta$ and $theta prime$;
  + Compare $theta$ and $theta prime$ using interleaving .
  + $theta prime$ wins the comparison, update $theta = theta + alpha * u$ (where $alpha$ is the learning rate).

#image("images/dbgd.png")

=== Advantages
- May be more efficient: because they have control over what data is gathered, by producing the ranking
- Learn the true preferences of users
- Responsive to changes in intents: they "immediately" respond to users

=== Potential problems
- Unreliable methods cn worsen the user experience
- At the start, the user experience may not be god and user may be exposed to bad rankers until close to convergence
- Exploration of the ranking space may mean exposing users to bad rankings, thus worsening user experience
- Click noise, bias, adversarial behaviours, serendipity may all negatively affect the learning process and worsen the user experience

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

#image("images/transformer decoder only.png", width: 70%)

=== Using BERT for ranking

- Need to fine-tune BERT for the ranking task
  - relevance classification
  - monoBERT

==== Training monoBERT: loss function

$L = sum_(j in J_"pos") log(s_j) - sum_(j in J_"neg") log(1-s_j)$

Common practice:
- For positive judgement (relevant), use human label
- For negative judgement (negative), sample from BM25

=== monoBERT

#image("images/monoBERT for ranking.png", width: 70%)

monoBERT is very effective, below are results on the TREC 2019 Deep Learning Track (passage retrieval).

#table(columns: (1fr, 1fr, 1fr, 1fr))[][nDCG\@10][MAP][Recall\@1k][BM25][0.506][0.377][0.739][+monoBERT][*0.738*][*0.506*][0.739][BM25 + RM3][0.518][0.427][0.788][+monoBERT][*0.724*][*0.529*][0.788]

==== monoBERT limitations
#image("images/bert.png", width: 70%)
Need separate embedding for every possible position, so we restrict the indices 0-511.
- Hence, we cannot input the entire document.

  To Improve Training:
  - Chunk documents
  - Transfer labels (although this is an approximation)
  To Improve Inference:
  - Aggregate evidence either over passage scores or over sentence scores.
  
- Computationally expensive layers.
  - e.g. 110+ million learned weights
  To Improve: 
  - We could use a multistage ranking pipeline with limited re-ranking
  - We could simplify the BERT interface & dense retrievers


=== monoBERT over passages scores
BERT-MaxP, FirstP, SumP either take the maximum, first or sum. 

#image("images/bert-max-p.png", width: 70%)

These methods deliver better results with MaxP showing the best results
// MAYBE remove????
#image("images/monobert-passages-scores-results.png", width: 70%)

=== monoBERT over sentence scores: Birch

$s_f =^Delta alpha dot s_d + (1 - alpha) dot sum^n_(i=1) w_i dot s_i$

Where: \
$alpha dot s_d$ is the first retrieval score \
$w_i dot s_i$ is the sentence scores
#import "@preview/hydra:0.6.2": hydra
Trained on sentence level judgements like tweets and interpolation weights are tuned on target dataset.
// Maybe add scores here??????

=== Representation Aggregate

#image("images/representation aggregation.png", width: 70%)

You can do this for either term embeddings or passage representations.

=== Over Passage Rep: PARADE

Aggregation approaches (in order of increasing complexity):

- Average feature value

- Max feature value

- Attention weighted average

- Two Transformer layers

#image("images/parade.png", width: 70%)


=== Multi-stage re-rankers

A multi-stage re-ranker will take the top k documents and then put it through another re-ranker. Usually this is from most efficient to least efficient with the the least efficient being the most effective. There is trade-off between effectiveness (quality of the ranked lists) and efficiency (retrieval latency). 

=== duoBERT
#image("images/duoBERT.png", width: 70%)

DuoBERT takes the token embeddings, segment embeddings, and position embeddings.

To train duoBERT

$L_"duo" = - sum_(i in J_"pos", j in J_"neg") log(p_i, j) - sum_(i in in J_"neg",j in J_"pos") log(1-p_i,j)$

Is doc di more relevant than doc dj to query q?

#image("images/duobert training.png", width: 70%)

=== Inference with duoBERT

#image("images/duobert inference.png", width: 70%)

=== monoBERT vs duoBERT

#image("images/monobert v duobert.png", width: 70%)


=== Takeaways of Multi-stage Rankers

Advantage:
- More tuning knobs $arrow$ more flexibility in effectiveness/efficiency tradeoff space


Disadvantage:
- More complexity

Reranking with transformers is new and not well studied

=== LLM as a Rankers

#image("images/llm-as-rankers.png", width: 70%)

All four main families are characterised by how documents are passed in the prompt an dhow relevance of document and query is determined.

All are "zero-shot": i.e. once you obtained the pre-trained, instructions tuned LLM, no need to do contrastive training

#image("images/llm-as-rankers-methods.png")

Where: \
a is pointwise, \
b is listwise, \
c is pairwise, \
d is setwise\

Setwise offers two advanages:
- Compared to listwise: it can rely on logits rather than generation so it's faster
- Compared to pairwise: it requires less comparisons (i.e. inferences with LLM)

#image("images/pairwise v setwise.png")

Prompts proposed for different LLM rankers vary largely. Not just in terms of instructions but adding additional wording such as a roleplaying and ordering of components (passage first, then query - or vice versa)

The effectiveness varies across these wordings, with LLMs finding ranking more complex than classifying. 

= Generative IR

As was previously discussed there are three transformer architectures:

#grid(columns: (0.5fr, 1fr))[][][
  #image("images/encoder.png", width: 75%)
][
  - Gets bidirectional context: can condition on future tokens!
  - Fine-tuning done via Masked Language Model
  - BERT uses this approach
  - Generating sequences: don't naturally lead to effective auto regressive (1-word-at-a-time) generation methods. \ \
][
#image("images/decoder.png", width: 75%)
][- Can generate text, but can't condition on future words
- GPT-stye][
  
#image("images/encoder-decoder.png")
][
  - Good parts of decoders and encoders? Not really.
  - Hard to scale (computationally complex)
  - Not data efficient
  - T0, T5
]

== Autoregression Decoders

In an autoregressive text generation model, at each time step $t$, the model takes a sequence of ${y}_(<t)$ as input, and outputs a new token $hat(y)_t$
#align(center)[
  #image("images/autoregressive-decoder.png", width: 50%)
]

== In-context Learning
pre-GPT-3 models are used in two ways:
+ Sample from the distributions they define (maybe providing a prompt)
+ Fine-tune them on a task, and take their predictions

Very large language models seem to perform some kind of learning without gradient steps (pre-training) simply from examples you provide within their contexts (input, a.k.a prompt).

Input (prefix within a single transformer decoder context): \
"thanks" $arrow$ "merci" \
"hello" $arrow$ "bonjour" \
"mint" $arrow$ "menthe" \
"otter" $arrow$ $...$ \
Output (conditional generations):
"loutre..."


#image("images/in-context-learning.png", width: 70%)

== Zero, one, few-shot learning

=== Zero Shot
The model predicts the answer given only a natural language description of the task. No gradient updates are performed.
```
Translate English to French
cheese =>
```

=== One Shot
In addition to the task description, the model sees a single example of the task. No gradient updates are performed.
```
Translate English to French
sea otter => loutre de mer
cheese =>
```

=== Few Shot
In addition to the task description, the model sees a few example of the task. No gradient updates are performed.
```
Translate English to French
sea otter => loutre de mer
peppermint => menthe poivree
pus girafe => girafa peluche
cheese =>
```

=== Fine Tuning
The model is trained via repeated gradient updates using a large corpus of example tasks.

```
sea otter => loutre de mer
```
_Gradient Update_

```
peppermint => menthe poivree
```

_Gradient Update_

Many other examples 

pus girafe => girafa peluche

_Gradient Update_
```
cheese =>
```

=== Chain of Thought Prompting

Chain of thought prompting explains your reasoning for an answer in a one-shot or a few-shot where you explain a previous answer to provide reasoning to the LLM.

== Retrieval Augmented Generation

LLMs encapsulate a vast amount of factual information within their pre-trained weights

This knowledge is inherently limited, relying heavily on the characteristics of the training data

Solution: use external datasets to incorporate new information or refine the capabilities of LLMs.

Two directions to do this:
+ unsupervised fine-tuning
+ retrieval-augmented-generation (RAG)

Unsupervised fine-tuning offers some improvement, but RAG consistently outperforms it, both for existing knowledge encountered during training and entirely new knowledge.
- LLMs struggle to learn new factual information through unsupervised fine-tuning.


#align(center)[
 #image("images/rag-pipline.png", width: 50%) 
]

The pre-retrieval and post-retrieval are not necessary.

The modularity of RAG allows for:
- Sequential processing and integrated end-to-end training across components
- New/Additional modules
- New Patterns (e.g. Rewrite-Retrieve-Read, Generate-Read, Recite-Read, Retrieve-Read-Retrieve-Read, Hypothetical Document Embeddings (HyDE), Demonstrate-Search-Predict (DSP))
- Adaptive RAG: flexible orchestration of RAG modules and flows, e.g. Self-RAG

== BlendFilter
BlendFilter enhances input queries through different augmentation strategies.
+ Use initial query to retrieve external knowledge.
+ Pass this to LLM prompt with few-shot examples and Chain of Thought
+ Generate answer
+ Concatenate answer and original query

Then it prompts for internal knowledge
#image("images/internal-knowledge-augmentation.png", width: 30%)

Then it eliminates the irrelevant knowledge using prompted LLM. It would provide the LLM the knowledge retrieved, the topic at hand and would ask it to categorise into a certain level of knowledge ("please check the relevance between the question and knowledges 0-4 on by one based on the context"). 

The LLM generates an answer based on the filtered knowledge and original query.


#image("images/blendfilter.png", width: 50%)

== Rewrite-Retrieve-Read Method

Queries are often ambiguous and underspecified

*Rewrite*: black-box LLM is prompted to re-write query

*Retrieve*: search based on the re-written query

*Read*: provide instruction, query, documents, generate answer

#image("images/rewrite retrieve read.png", width: 10%)

Few-shot (1-3) prompt in format [instruction, demonstrations, input]

Output can be none, one or more queries

#image("images/read-rewrite-read open vs multiple.png", width: 40%)

#image("images/rewrite retrieve read method.png", width: 50%)


Trainable Rewriter

Warm-up: training on pseudo-data
- Collect rewritten queries x, keep only those that generate correct end-to-end answers: use to form warm-up dataset.
- Fine-tune rewriter on warm-up dataset

Then continually trained by Reinforcement Learning with PPO

Reward obtained from end-to-end answers compared to gold + KL regularisation to prevent model doom deviating too far.

== Prompt Compression

Reduce the prompt (aka context) to consume less tokens than in its original form. 

Why prompt compression?
- Long prompts often confused LLMs  and yield lower effectiveness
- Latency of decoder-based LLMs is quadratic with respect to prompt length

Two main categories of approaches
- Lexical based: Compress prompts by removing tokens according to their information entropy obtained from LLM
- Embedding based: Compress prompt into special tokens (short compact memory slots) that can be directly conditioned on by the LLM.

== Model Knowledge and Retrieval Knowledge


The output of RAG is not guaranteed to be consistent with retrieved relevant passages

Because the models are not explicitly trained to leverage and follow facts from provided passages.

=== Inconsistent with RAG Evidence

Injecting Knowledge


#image("images/injecting knowledge 1.png",  width: 70%)


#image("images/injecting knowledge evidence.png", width: 70%)

== Power of Noise

Introducing semantically aligned yet non-relevant documents potentially misguides LLMs. 

=== Distracting Passage
Progressive accuracy degrades as the number of distracting documents included in the context. Adding just one causes a sharp reduction

=== Random Passage

A random passage _can_ cause an improvement in effectiveness but it is LLM dependant.

== Attribution

Attribution: the ability to generate evidence (in the form of references or citations) that supports claims the LLM makes in an answer

LLMs fail to correctly attribute and is consistently incorrect and evidence does not have evidence to support it's possition.

There are two ways to improve attribution:
- Direct generation attribution, and
- Retrieval-based attribution

However, direction generation attribution is prone to hallucination.

=== Retrofit Attribution using Research and Revision (RARR)

Automatically finds attribution for LLM output

Post-edits the output to fix unsupported content while preserving original output as much as possible.

#image("images/rarr 1.png")

== RAG Resources

Many platforms/frameworks for RAG
- Industrial-oriented, e.g. Llama-Index, LangChain
	- Some provided only/also as a service, e.g. Nuclia AI
- Research-orientated: Ragnarok, FlashRAG, BERGEN

Datasets
- Many datasets, not necessarily focusing on RAG
- Few recent data specifically designed for RAG, e.g.
	- TREC RAG 2024

=== BERGEN
End-to-end library for reproducible research standarising RAG experiments

Focus on QA

Implements different state-of-the-art retrievers, rerankers, and LLMs.

Also large scale analysis of RAG components (state-of-the-art retrievers, LLMs $arrow$ 500+ experiments)

Key findings:
+ need to go beyond commonly used surface-matching metrics (e.g. exact match, F1, Rouge-L, etc.)
+ Retrieval quality matters for RAG response generation
+ Need to improve current knowledge-intensive benchmarks to use them in RAG:
	+ Datasets evaluating general knowledge might not be suitable for RAG, as LLMs have acquired most such knowledge from Web/Wikipedia
+ LLMs of any size can benefit from retrieval.

=== TREC RAG 2024 & Ragnarok

TREC RAG 2024: dataset for RAG evaluation. Three tasks: Retrieve (R), Augmented Generation (AG), Retrieval Augmented Generation (RAG)

Ragnarok: open-source, reproducible, reusable framework implementing RAG pipeline, with 2 sequential modules: (1) R, (2) AG

=== Feb4Rag

Create pseudo search engine for RAG systems; Good-system $arrow$ pick the correct search engine with respect to user query and generate high-quality responses

Sample evaluation on RAG responses.

== LC-LLMs


Long-context LLMs enable promptss to be… long: this allows to pass for text (not just instructions) into the prompts

Examples include Gemini-1.5-Pro (1M tokens; internal version 10M tokens), GPT4o (128k tokens), GPT-3.5-Turbo (16k tokens)

Long context prompting is expensive due to quadratic computation cost of transformers regarding to the input token numbers

Methods have been proposed to reduce cost
- Prompt compression
- Model distillation
- LLM cascading

Most LLM APIs charge based on token count.

=== Advantages

Potentially allows for consolidating complex pipelines into a unified model, thus reducing issues like:
- Cascading erros
- Cumbersome optimisation
Streamline end-to-end approach to model development

=== As a Search Engine

This is currently an active area of study for Google Deepmind.

Corpus-in-Context (CiC, pronounced "seek") direct ingestion and processing of entire corpa within their context window.

#image("images/cic.png")

==== CiC Prompting

#image("images/CiC prompt.png", width: 60%)

In the instruction they used funny wording such as “read carefully and understand”

They hand crafted the prompt to get the model to perform well

#image("images/cic prompt docids.png", width: 60%)

Should use the same formatting over all the documents

==== Efficiency

As it uses token as cost, sending a ton of documents as a prompt will be expensive!

CiC prompting is compatible with prefix-caching

The corpus only needs to be encoded once (i.e. indexing phase)

#image("images/cic limitations.png", width: 40%)

==== Is it any good?

#image("images/cic prompts.png")

That's not bad!

=== Position Bias

#image("images/position bias.png", width: 50%)

performance drops as gold documents of the test queries are more towards the end of the corpus
- ? reduced attention in later sections of the prompt
placing gold documents of few-shot queries at the end improves recall

co-locating gold documents of few-shot and test queries consistently boosts performance

=== Few Shot

#image("images/few shot.png", width: 40%)

=== Extreme Model Based IR


Compared to RAG: no multi-component framework to maintain/trainrun

Compared to DSI: no training! (indexing)


#image("images/lc llm extreme ir.png", width: 50%)


== Vision Language Models

#image("images/screenshot retrievers.png")

Current: extract each element then encode

Screenshot: take screen then encode that entire thing

=== Vision Language Models

#image("images/vision language models.png", width: 60%)

= Hyperlink Information in Retrieval

Documents often contain metadata that explicitly relate them to other links.
- Many document collections contain explicit links (web hyperlinks, citations in papers/legal cases).
- Links can convey authority (a vote) and provide additional topical signal via anchor text.
- Uses in IR: scoring/ranking, link-based clustering, features for classification, crawl prioritisation.

== Anchor text
Anchor text on links pointing to page $D$ can be indexed with $D$ (weighted by the authority of the source page). Useful signal: e.g., many anchors saying “ibm” pointing to `www.ibm.com` strongly indicate the target is about IBM.
- Risks: spam, manipulation — weight anchors by source authority when possible.

== PageRank (intuition)
Model: web as directed graph (nodes = pages, edges = hyperlinks).

Random-surfer process:
  - Start at a random page.
  - At each step: with probability $d$ follow a uniformly-chosen outlink; with probability $1-d$ teleport to a random page.
  - In steady state, each page has a long-run visit probability = PageRank.

=== Transition / equation (compact)
Let $"PR"(u)$ be PageRank of $u$, $B_u$ the set of pages linking to $u$, $L_v$ the out-degree of $v$, $N$ the total pages, and $lambda$ the teleport prob.

$"PR"(u) = (1-lambda)/N + lambda * sum_(v ∈ B_u) "PR"(v) / L_v$

- Typical damping value: $lambda approx 0.85$ (so teleport prob $(1-lambda) approx 0.15$).
- Intuitions for high PR:
  - Many in-links.
  - In-links from high-PR pages.
  - In-links from pages with few out-links (less dilution).

=== Practical considerations
Handle sinks (dead-ends) via teleportation.
Variants: site-internal links weighting, seeding new pages’ PR, combating link farms.

PageRank is topic-independent — high PR doesn't guarantee topical relevance for a query.

== Integrating PageRank into retrieval
Treat PageRank as a query-independent feature (document prior).

=== In Language Models
Replace uniform prior $P(D)$ with priors proportional to PageRank, spam score, URL depth, etc.

=== In Vector Space / Learning-to-rank
Combine signals outside the pure vector space: \
  $"score" = w_"vsm" * "sim"(q,d) + w_"pr" * "PageRank"(d) + w_"spam" * "SpamScore"(d) + ...$

=== In BM25
Model document as text $T$ + features $F$; weight and combine feature functions $F_(i(d))$ with tunable weights.

Document priors can improve ranking but don't always guarantee better effectiveness; effects vary by model and query length.

== Search-result diversification
=== Motivation)
Queries can be ambiguous (multiple)at maximises coverage of possible intents while minimising redundancy.

=== Probability Ranking Principle (PRP)
PRP: rank docs by decreasing probability of relevance (to maximise expected effectiveness).

PRP assumes: (A1) relevance probabilities are estimated accurately (no uncertainty), (A2) relevance of a document is independent of other returned documents.

Both assumptions are often violated in practice (ambiguity and interdependence).

=== Diversification formulation
Treat query as a mixture of possible information needs (aspects). Seek coverage across aspects and novelty among selected documents.

=== Maximal Marginal Relevance (MMR)
Balances relevance and novelty. For candidate `d`:

$"score"(d) = alpha * P("relevant" | d) - (1-alpha) * max_(d prime in "ranked") "sim"(d,d prime)$

- $alpha$ controls trade-off: higher $arrow$ more relevance; lower $arrow$ more diversity.

== Evaluation & other topics
Diversity-aware measures: intent-aware nDCG, a-nDCG, subtopic recall, MRR variants, ERR variants.


Resources and testbeds: TREC, NTCIR collections for diversification tasks.

Additional tasks: query ambiguity detection, query aspect mining, diversification across verticals.

= Relevance Feedback (RF)
Provide to the search engine information about the relevance of a document
 - e.g. a user may tick one of the results on a SERP and say “this is relevant”
- Explicit RF: the user explicitly tell the system one or more documents are relevant (both positive&negative)
  - “More like this”, query-by-example
- Implicit RF: use some of the signals from users to attempt to infer a relevance signal (both positive&negative)
  - Clicks (though more often used in click-models/LTR)
- Pseudo RF: just assume the top retrieved results are relevant, and recompute ranks (just positive) (rarely people also consider negative

== VSM Extension for RF: Rocchio
Attempts to construct the Optimal Query
- Maximises the difference between the average vector representing the relevant documents and the average vector representing the non-relevant documents
- Modifies query according to
$
q^prime_j = alpha dot q_j + beta dot 1/(|"Rel"|) sum_(D_i in "Rel") D_("ij") - gamma dot 1/(|"Nonrel"|) sum_(D_i in "Nonerel" d_("ij"))
$

== Query expansion
Two main approaches:
+ Thesaurus / knowledge base
  - Adding synonyms or more specific terms using query operators base don thesarusus
  - Contradicting evidence on whether it improves search effectiveness
+ analysis of term co-occurrence / statistics

#image("images/Screenshot from 2025-11-13 17-53-54.png")
#image("images/Screenshot from 2025-11-13 17-53-59.png")

Associated words are of little use for expanding the query "tropical fish " if considering "tropical" and "fish" on their own. Expansion based on whole query takes context into account (n-grams) though it is impractical for all possible queries.

To actually deal with query expansion in your models you can either
+ Run the expansion terms through the same retrieval model you would use without expansion
  - possibly weighting them differently to original terms
+ Formally define a retrieval model that represents the expansion process
  - Use the model to determine: which terms are selected for expansion and their weight.

== Query reduction
Removes from a verbose query those terms that may be out of focus and hurt the retrieval process, you can use POS tagging, IDF-r or query performance predictors to determine terms that may not benefit the process, and remove them from the query.

== Rank Fusion
Combines many methods into one to boost performance.

Methods:
- Voting: based on the number of lists that support the retrieval of a document
- Rank aggregation: rank of each document within each list is considered (Borda, RRF)
- Score aggregation: rank and score of each document within each list considered (min-max, sum norm)



#pagebreak()

= Index Compression

== Indexes and Memory
Goal: fast access to the index for quick query processing. Faster access $arrow$ index stored closer to CPU. Memory high in the hierarchy (cache, main memory) is limited and expensive. A full index with positions and extents can be as large as the original collection. Compression techniques help manage the memory hierarchy efficiently.

=== Advantages of Index Compression
Reduces disk and memory requirements.
Allows index data to move up the memory hierarchy (closer to CPU). If compression achieves a factor of 4, we can store $4 times$ more data in CPU cache. Compression avoids fragmentation and paging issues.

This Enables sharing of memory bandwidth across multiple CPUs however, requires decompression thus, it must be fast and efficient.

== Modelling Compression
CPU can process $p$ postings per second.
Memory can supply CPU with $m$ postings per second.
Effective processing rate: $min(m, p)$.

Cases:
- If $p > m$: CPU is idle (waiting for memory).
- If $m > p$: memory is idle (waiting for CPU).

With compression rate $r$ (r postings stored in place of one):
- Memory supplies $m times r$ postings per second.
- CPU processes $d times p$ postings per second, where $d$ is the decompression factor.

Effective processing = $min("mr", "dp")$.

== Goal of a Compression Algorithm
We want to maximise $min("mr", "dp")$.

- No compression: $r = 1$, $d = 1$.
- With compression: $r > 1$, $d < 1$.
- Compression only helps when CPU is faster than memory ($p > m$).
- Ideally, choose compression such that $"mr" = "dp"$.

Only lossless compression methods are used in IR (we can’t lose data in posting lists).

== Basic Idea of Compression
Common data elements use short codes whereas, uncommon data elements use longer codes.

Example: encoding numbers
- If we encode 0 as just “0”, total bits are reduced (e.g., 10 bits instead of 14), but decoding becomes ambiguous.
- An encoding scheme must be unambiguous.

== Delta Encoding
Assumption: small numbers occur more often than large ones.
- True for word frequencies: many words appear once or twice.
- False for document IDs (some small, some large).

Solution: store differences between consecutive document IDs $arrow$ *delta encoding*.

Example:
```
DocIDs: 1, 5, 9, 18, 23
Diffs:  1, 4, 4, 9, 5
```
These are called d-gaps.

Produces smaller numbers (good for compression).

=== Frequent vs Rare Words
- Frequent words $arrow$ long posting lists, small increments (small d-gaps).
- Rare words $arrow$ short lists, large increments (large d-gaps).
- Frequent words compress better.

== Unary Coding
Encode integer $k$ as $k$ ones followed by a zero.

Example: $k = 3 arrow 1110$

- Works in base 1.
- The trailing 0 makes it unambiguous.
- Bit-aligned $arrow$ code boundaries can appear after any bit.

Unary vs Binary:
- Unary good for small numbers.
- Binary better for large numbers.
- Example: encoding 1023 $arrow$ unary requires 1024 bits, binary needs only 10.


== Elias-Gamma Codes
Combines unary and binary coding.

Steps to encode $k$:
1. Compute $floor(log_2(k))$  $arrow$ number of bits required for binary encoding.
2. Encode this length in unary.
3. Encode `k` in binary using that many bits.

- Unary part indicates how many bits to read for the binary part.
- Total bits: $2 floor(log_2(k)) + 1$

Efficient for small numbers, inefficient for large ones (binary alone would use $floor(log_2(k))$ bits).

== Elias-Delta Codes
Steps to encode $k$:
1. Encode the length of $k$’s binary representation ($log_2(k)$) in Elias-gamma.
2. Encode `k` in binary.

- Slightly worse for small numbers but better for large ones.
- Bit cost $approx 2log_2(log_2k) + log_2k$.

== Auxiliary Structures
The inverted index is the primary data structure of a search engine.

But there are also auxiliary structures:
- Vocabulary/Lexicon: lookup table from term $arrow$ byte offset of inverted list in the index file.
- Collection statistics: stored separately (term frequencies, document counts, etc.).

Vocabulary typically fits in memory.

== Data Structures
To decide which structures to use, consider time complexity and space complexity.

== In-Memory Index Construction
A simple algorithm:
1. Parse documents and store postings in memory.
2. When memory fills up, write current index to disk as a partial index.
3. Start a new in-memory index.

Result: many partial indexes on disk $arrow$ must merge.

=== Merging Process
- Merge partial indexes ($I_1, I_2, …, I_n$) into a single sorted index.
- Must merge in small chunks (can’t load all into memory).
- Each list stored in alphabetical order $arrow$ merge efficiently.

Distributed systems: use message-passing or MapReduce for large-scale merging.

== Using Indexes for Retrieval
Two main strategies for scoring:

=== Document-at-a-Time (DAAT)
- Process all query terms for one document before moving to the next.
- Scores computed per document.
- Advantage: low memory usage; good for top-k retrieval.

=== Term-at-a-Time (TAAT)
- Process all documents for one term before moving to the next.
- Scores accumulated across documents.
- Advantage: efficient disk access (sequential list reading).

Both methods have optimisations for faster retrieval.

