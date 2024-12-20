# gw-pdf-extractor
Extract data from a pdf using **open source LLMs** for QA. If the data cannot be found directly in the pdf, use **sentiment analysis** on relevant content. Created to analyse **greenwashing** data but intended to be versatile.
## Notes for improvement

 - Currently using *deepset/roberta-base-squad2* for QA. Improve accuracy by using a **larger model**, e.g. [ModernBERT](https://github.com/AnswerDotAI/ModernBERT). Necessary to fine tune this model for QA.
 
 - **Improve PDF parsing**. Most accurate results found when using LlamaParse to generate markdown, however there likely exists a better way to do this that results in more accurate output from QA.
 
 - **Prune sections of the PDF** that are very unlikely to contain relevant content. This will result in better performance, reducing unnecessary work.
