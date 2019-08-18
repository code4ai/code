## Seq2seq Model Used for Rewriter

*Requirements* 
- bert-base-uncased.30522.768d.vec ([download here](https://github.com/google-research/bert). Put bert-base-uncased.30522.768d.vec in the main directory.)
- Also, put source.txt and target.txt in the main directory.

Then, run the following:

`python train.py --input==source.txt --output==target.txt`

### Unknown Token Handling
Since unknown tokens affect the result, we use an approach described in [Unsupervised Sentence Compression using Denoising Auto-Encoders](http://aclweb.org/anthology/K18-1040) to generate rare words that are not in the vocabulary.
