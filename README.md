# Assignment 1 -  Sparse Retrieval

## Overview
Syllabus modules covered: Retrieval framework · Sparse models (Boolean, vector-space, probabilistic,
unigram/bigram language modelling) · Ecosystem/TREC collections · Evaluation metrics and protocols.

**Difficulty Level** : 1 of 4 (basics)
**Format** : Inidividual, continuously updated leaderboard

You will build a from-scratch inverted-index retrieval engine and tune it to outrank every other submission in the class — and the instructor's own reference system — on nDCG@10 over a held-out query set. This is the assignment where the difference between a B-grade IR system and an A-grade one is entirely in the details: stemming choices, smoothing constants, term-weighting design, and how you handle short queries and rare terms.

## Learning Objectives
* Implement an inverted index from raw documents through tokenisation, stopword handling, and postings construction.
* Implement and compare at least three ranking functions: a Boolean/vector-space model, BM25 (Robertson & Walker et al., 1992; Robertson & Zaragoza, 2009), and a query-likelihood language model
with smoothing.
* Understand, in a hands-on way, why parameter choices (BM25's k1, b; LM's Dirichlet/Jelinek–Mercer smoothing constant) materially change ranking quality on real query sets.
* Correctly implement and interpret standard TREC-style evaluation metrics (MAP, nDCG@k, MRR, Precision@k) using trec_eval-compatible qrels.
* Experience, first-hand, how leaderboard-driven tuning can overfit to a public dev set — and why a heldout private evaluation set exists.
