#set text(font: "Noto Sans")
#set text(lang: "en", region: "au")
#[
  #set text(size: 50pt)
  INFS7410 Revision
]

Foreword:

These are notes written by a web-developer who has no idea on web-indexing, read at your own risk. If you have thoughts comment on this document via Typst.app or via the Github discussion. Input is welcome! 

#pagebreak()

#outline()

#pagebreak()

= The Basics
#set quote(block: true)
#show quote: set pad(x: 5em)
#quote(attribution: [Gerald Salton, IR pioneer])["Information Retrieval is the field concerned with the structure, analysis, organisation, storage, searching and retrieval of information"]

Typically, you interact with IR by entering a query (which expresses what information a user needs) into a specific vertical (different search engines into one, i.e. image search, or news search). It will return documents (which is an ordered list of information according to the relevance of the query) which are ranked by some criteria or model. SERP or search engine results page is the entire page of results.

A search engine retrieves documents according to their relevance to the query. The term document is used to describe the unit of information to retrieve. Documents are not necessarily just text, they could be images or songs as well and are unstructured or semi-structured. Some parts of documents are important than others: 
- document parsers often recognise structure such as markup, HTML, and more.
- head, nachor text, and bold text are all likely to be more important
- links within the document can be used for link analysis.

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

== Tokenizing
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

= Term Based Lexical Models

== BM25 (Best Match 25)
= BM25 Ranking Function

BM25 is a popular and effective ranking algorithm based on the binary independence model.

It expands upon the tf-idf idea (under its most general form with no relevance information provided), with two main differences:
- Term frequency saturation
- Document length normalisation

Empirically, it has been shown to be a quite reliable and robust model, which works well out-of-the-box in most situations.  
It is also the default retrieval model in many open-source search engines such as Lucene and Elasticsearch.

For each subsequent section, we will separate the BM25 formula into its components and analyse them.  
The full equation is as follows:

$sum_(i=1)^(|Q|) log ((r_i+0.5)/(R-r_i+0.5)/((n_i-r_i+0.5)/(N-n_i-R+r_i+0.5)) dot ((k_1+1)f_i)/(k_1B+f_i) dot ((k_2+1)q f_i)/(k_2 + q f_i))$


= Components of BM25

== RSJ Weight

$log ((r_i+0.5)/(R-r_i+0.5)/((n_i-r_i+0.5)/(N-n_i-R+r_i+0.5)))$

This component is known as the *Robertson–Sparck Jones (RSJ) weight*.

It considers knowledge about the number of relevant documents ($R$) and the number of relevant documents that contain term $i$ ($r_i$).  
$N$ and $n_i$ refer to the number of documents that have been judged to obtain $R$ and $r_i$.  
Thus, if no relevance information is provided, it becomes:

$log (N-n_i + 0.5)/(n_i+0.5)$

which is an approximation of the classical IDF.

== Saturation Component

$((k_1+1)f_i)/(k_1B+f_i)$

This is known as the *term saturation component*.

The contribution of the occurrence of a term to a document score cannot exceed a saturation point.  
The parameter $k_1$ controls the saturation, where $k_1 > 0$:
- High $k_1$ → $f_i$ contributes significantly to the score.
- Low $k_1$ → additional contribution of further term occurrences tails off quickly.

Typically, $k_1$ is set to 1.2.

= Effect of Term Frequency in BM25

Consider the query “president lincoln.”
Assume document length is constant and equal to 90% of the average document length, with $N = 500000$ documents.

- President occurs in 40,000 documents ($n_1 = 40000$).  
- Lincoln occurs in 300 documents ($n_2 = 300$).

== Document Length Normalisation

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

== Query Component

$((k_2+1)q f_i)/(k_2 + q f_i)$

This is known as the within-query component.

It is useful for longer queries where a term may occur multiple times.  
It introduces similar saturation behaviour as the within-document component and has its own constant $k_2$ (sometimes referred to as $k_3$).  

Experiments suggest that this term is not particularly important — meaning multiple occurrences of a term in a query can often be treated as separate terms.


= Neural Retrievers