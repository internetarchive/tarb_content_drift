GSoC'23 @Internet Archive | TARB - Content Drift

Introduction
Hello, I am Darahaas, a Computer Science major at Birla Institute of Technology and Science, Goa. I have been working on the “Turn All References Blue" (TARB) project at the Internet Archive through Google Summer of Code 2023. This has been both a valuable learning experience and a key step in my open-source journey. The TARB project identified two existing big challenges and advertised them for GSoC contributors: detection of soft-404s and analysis of content drift. I have chosen to focus on the latter to be the major part of my GSoC project.
The primary objective was to develop a comprehensive analytical framework that enables researchers, archivists, Wikipedia editors, and general users to ascertain if there has been a drift in the content of a referenced Wikipedia hyperlinks. This is crucial for ensuring the integrity and historical accuracy of linked web pages, as well as providing valuable data on the fluidity of digital information.
The Problem Statement
The problem focuses on "content drift" in hyperlinks within Wikipedia articles, a phenomenon where linked webpages change over time and may lose relevance to the original article. This dynamic nature of web content poses challenges for both Wikipedia's reliability and archival systems like the Internet Archive. Given the high volume and diversity of such hyperlinks, manual tracking is impractical, necessitating an automated system for scalable and accurate detection of content drift. The aim is to either update or annotate these drifting hyperlinks regularly, ensuring the contextual integrity of Wikipedia articles and maintaining them as reliable resources for future research and public use.

Detection of content drift
Below we have an example which showcases a case of content drift, on two pages which are versions of a page in the references of wikipedia - National Stockpile.

April 2nd, 2020:

Fig 1. SNS website on April 2, 2020

Fig 2. Semantic Relevance - April 2, 2020

Here we have a reference from the wikipedia page on the Strategic National Stockpile (Fig 1), which has a brief explanation of what the the term means. We have generated a semantic relevancy metric (Fig 2) that will only be used in comparison with another score.

April 6th, 2020:

Fig 3. SNS website on April 6, 2020

Fig 4. Semantic Relevance - April 2, 2020

For the same reference, we see that at a later date, the explanation of the term Strategic National Stockpile has been cut down much shorter (Fig 3), thereby indicating a decrease in content relevancy. We have generated a score for this as well, and we can see that since it still talks about the same topic, it is relevant to the wikipedia context, but the score is slightly lesser than the above generated one (Fig 4), hence indicating a decrease in content relevancy.
The CLIs, APIs and web interfaces being talked about in this article can be found at https://github.com/internetarchive/tarb_gsoc23_content_drift.

Approach to the problem
Dataset Creation
The Internet Archive maintains a specialized collection called the Wikipedia EventStream Page Links Change(wikipedia-eventstream-page-links-change). This collection serves as a comprehensive log, capturing all modifications and new additions to hyperlinks within Wikipedia articles. It is an invaluable resource for tracking the dynamic nature of web content as it relates to Wikipedia.
Given the project's aim to detect content drift using advanced language models and contextual summarization techniques like BERT and BART, an initial dataset was created specifically targeting the English Wikipedia domain (“https://en.wikipedia.org”). This dataset comprises changes or additions to hyperlinks within English Wikipedia articles, serving as the foundational data layer for subsequent machine learning tasks.
By focusing on English Wikipedia pages, the dataset offers a manageable yet rich scope for analysis. It allows for the application of NLP techniques to assess the contextual relevance of these hyperlinks over time, thereby providing a robust framework for detecting and understanding content drift.

https://web.archive.org/web/20181220050616/help.twitch.tv/customer/portal/articles/2964218-ending-support-for-voice-chat-video-chat-group-messaging-and-servers
https://en.wikipedia.org/wiki/Twitch_(service)
https://www.usatoday.com/story/sports/olympics/2016/08/03/ioc-approves-addition-of-5-sports-for-2020-tokyo-olympics/88035700/
https://en.wikipedia.org/wiki/List_of_Olympic_medalists_in_softball
https://www.inaturalist.org/taxa/1198287
https://www.inaturalist.org/taxa/1198287
https://www.dropbox.com/sh/k9o2q7p7o4awhqx/AAAkMyzrlbRAICtQLNZhuPuma/Apr%25202023%2520Single%2520Accreds.pdf
https://en.wikipedia.org/wiki/Oliver_Tree
…
…

The initial dataset that I showed above was created in the following way:
Get the eventstream data (which consisted of the link, wikipedia page link, timestamp and user)
Drop the columns timestamp and user
Filter the data in such a way so that
The wikipedia page link column contains only english wikipedia pages
There are not too many rows with the same wikipedia link
The next step involved web scraping to enrich the data with additional contextual information from the Wikipedia pages. Specifically, I extracted the following attributes for each hyperlink:
Anchor Text: The clickable text that is used in the hyperlink on the Wikipedia page. This provides insight into how the link is presented to the user and its intended context within the article.
Page Heading: The heading or section-heading under which the hyperlink is located. This offers a higher-level context, helping to understand the section of the article where the link is placed.
Surrounding Paragraph: The text paragraph that surrounds the hyperlink. This captures the immediate context in which the hyperlink is embedded, offering a more granular level of detail.

The final dataset has the columns: link, wikipedia_page, anchor_text, page_heading and surrounding_paragraph.

link
wikipedia*page
anchor_text
page_heading
surrounding_paragraph
https://www.wshu.org/2023-05-01/wshu-podcast-nominated-for-peabody-award
https://en.wikipedia.org/wiki/Talk:List_of_Peabody_Award_winners*(2020%E2%80%932029)
WSHU podcast nominated for Peabody Award
Talk:List of Peabody Award winners (2020–2029)
"Sources:2023 Peabody Awards Nominations Include ‘The Territory’ and ‘George Carlin’s American Dream’|IndieWire Peabodys Unveil Documentary And News Nominees Including ‘Lucy And Desi’, ‘We Need To Talk About Cosby…
https://ahluwalia.world/pages/about
https://en.wikipedia.org/wiki/User_talk:310001_art
Ahluwalia
User talk:310001 art
British fashion designer Priya Ahluwalia is the UK-based creator of the clothing line Ahluwalia. She has been repeatedly praised for her ethical and environmentally friendly fashion design methods as well as for incorporating …
http://www.essentialvermeer.com/history/neighbours-slager.pdf
https://en.wikipedia.org/wiki/Talk:Maria_de_Knuijt
http://www.essentialvermeer.com/history/neighbours-slager.pdf
Talk:Maria de Knuijt
"@CaroleHenson and Drmies:, I don't know how reliable this is, but according to H. G. Slager at http://www.essentialvermeer.com/history/neighbours-slager.pdf: "
https://finderhub.com.au/things-to-do-in-great-ocean-road/
https://en.wikipedia.org/wiki/Lorne,_Victoria
The Great Ocean Road Museum
Lorne, Victoria
"The town has two pubs (The Grand Pacific Hotel and Lorne Hotel) and a number of cafes, restaurants, and bakeries, mostly located along Mountjoy Parade. The town is serviced by one supermarket with a reasonable range of products…

Architecture Overview

Initial Architecture overview:

Data Layer:
Dataset: Wikipedia Event Stream Data with columns such as link, Wikipedia page, anchor text, section-heading, surrounding paragraph, LLM relevance score, and BERT relevance score.

Analytics Layer:
Language Models and Algorithms
LLM (GPT 3.5) - Accurate Determination
Role: Provides a binary metric of "Relevant" or "Not-relevant."
Limitations:
Inconsistent percentage values.
Limited by a 4096 character limit.
Data Preprocessing:
Truncation
Iterative Summarization
LDA (Latent Dirichlet Allocation) - Language Agnostic
Role: Scoped analysis for relevance in different languages and specific parts of the page. Language agnostic metric shown below:

Limitations:
Resource-intensive
Slow in iterative analysis

BERT (Transformers) - Constant Relevance Metric
Role: Provides a constant relevance metric using vector embeddings and cosine similarity.
Limitations:
Resource-heavy

Integration Layer:
How LDA and LLM Complement Each Other:
Scoped Analysis: LDA narrows down the context that LLM will evaluate, enhancing the accuracy of relevance determination.
Where to Use BERT
Threshold Determination: BERT can be used to set a relevancy threshold for detecting content drift, complementing LLM's binary relevance metric.

Pros and cons of the initial architecture:
Initially, we were creating a summary of both the wikipedia page as well as the link content using the LLM, and comparing those two to get our required relevancy scores.

After testing this on the dataset we prepared, the summarized content for comparison within the Wikipedia page (which the hyperlink is a part of) is mostly similar to the surrounding_paragraph section of the dataset, unless the hyperlink is a part of the references section(in which case we will have to summarize the page accordingly). If this metric is available (say > 200 words, for example), we need not summarize the wikipedia page but directly compare with this itself. Also gives us three independent metrics, BERT, LDA and LLM to get content relevance.

While these metrics are robust enough for detecting content drift, LDA as well as LLMs are not adequately sensitive when the changes in content are subtle or minute. These traditional methods may overlook fine-grained shifts in the context or semantics, thereby failing to detect what is known as "content drift." Content drift refers to changes in the subject matter or context over time. Therefore, a more advanced summarization system might be needed—one that can capture such nuances and offer a more accurate representation of the original content.

Final architecture overview:
The system for checking content relevance has the following stages:
Content extraction - Beautifulsoup4
Tokenization - tokenizer - BERT tokenizer
Summarization - BART
Similarity - BERT, RoBERTa, and XLNet
Reporting

Calculation of a combined metric:

The BART + BERT system for checking content relevance employs a multi-stage pipeline that starts with Content Extraction. At this stage, web content is scraped and relevant segments are isolated for analysis. If there is no context to compare against for a link and its corresponding wikipedia page, we are using a get_relevant_content function (Fig 5) that locates an anchor tag matching the provided text and finds its parent element among "p", "div", or "li" tags. It then extracts text content from this parent element and adjacent siblings, aiming for a larger set of text surrounding the anchor text.

This raw content is then tokenized to convert it into a format that can be fed into machine learning models. The Tokenization stage is crucial for preparing the data for the subsequent Summarization stage, where BART (facebook/bart-base) is used to condense the content into a more manageable size. This is helpful when we don’t have a context from the wikipedia page that we can compare right off the bat (like a paragraph) and need to shorten the context of the wikipedia page to have a better comparison.

The Similarity stage is where the heavy lifting occurs. BERT is used alongside other transformer models like XLNet and RoBERTa to generate embeddings for the summarized content. These embeddings are then compared to measure the cosine similarity, providing a metric for content relevancy. The system also incorporates a reporting stage, which has helper functions that are useful for showcasing the metrics in the form of a CLI. By combining these stages, the system achieves a robust and comprehensive analysis of content relevance, leveraging the strengths of multiple state-of-the-art NLP models.

Our similarity segment demonstrates a sophisticated approach to text similarity by leveraging multiple models and techniques. It defines a function combined_similarity that takes two texts and computes their similarity using three different metrics: BERT embeddings, TF-IDF vectors, and Jaccard similarity. Each of these metrics captures different aspects of text similarity, making the combined metric more robust and comprehensive.
In the function, embeddings for the two texts are first generated using BERT (bert_similarity), RoBERTa, and XLNet models. An ensemble score is generated from these three models. Cosine similarity is then computed between these combined embeddings. Additionally, TF-IDF vectors are generated for the texts, and their cosine similarity (tfidf_similarity) is calculated. Finally, Jaccard similarity (jaccard_similarity) is computed based on the intersection and union of the words in the two texts. These three similarity scores are then combined into a single metric using a weighted sum, where the weights are determined based on the amount of sensitivity we want for contextual analysis. Giving higher weight for BERT implies detection of change in relevancy for texts that still are about the same topic but slightly different from each other, but will perform rather poorly when comparing against texts that are not at all relevant to each other, which is why we have this ensemble mode (TF-IDF, and Jaccard)l to also account for that This weighted sum produces the final combined similarity score, capturing semantic, lexical, and set-based aspects of the texts.

Fig 5. System for measuring relevance scores

Once the relevance score has been generated, we can run the same system on previously archived versions of the hyperlink on the Wayback Machine, to see if any of these links have a higher relevance score than the one that exists currently in the wikipedia page.
Conclusion
After an intensive period of development, the project aimed at detecting content drift in Wikipedia hyperlinks for the Internet Archive has made significant strides. Leveraging state-of-the-art language models like BERT, BART, and LLM, the system provides a multi-layered approach to assess hyperlink relevance over time. While the system is robust for most use-cases, it faces challenges in detecting subtle changes, indicating room for further optimization.
The architecture is modular, consisting of content extraction, tokenization, summarization, and similarity assessment stages. It employs a combined metric system that utilizes BERT embeddings, TF-IDF vectors, and Jaccard similarity to offer a comprehensive relevance score. Despite its complexity, the system is user-friendly, with a web UI that allows for easy interaction, and a set of APIs for easy integration.
The codebase is well-structured, separated into different modules for web UI, scripts, and APIs, each serving specific functionalities. The framework not only assesses content relevance but also enables the monitoring of content drift by comparing current versions of a page against its older iterations.
In conclusion, the project successfully addresses the problem of content drift in Wikipedia hyperlinks, offering a scalable and automated solution. However, it also highlights the need for more nuanced algorithms to capture subtle changes, setting the stage for future enhancements.

Future Goals:
Enhancing BERT with LLM Summaries
One of the key areas for future development involves addressing the limitations of our current BERT + BART summarization model. We aim to fine-tune the BERT model using summaries generated by the Language Model (LLM) with a specific temperature setting. The inclusion of the temperature parameter (A lower temperature (e.g., 0.2) makes the output more focused and deterministic, while a higher temperature (e.g., 0.8) introduces more variability) is intended to modulate the stochasticity of the LLM's output. This fine-tuning strategy aims to stabilize the LLM's output over multiple invocations, ensuring deterministic and consistent summaries for identical input text. By doing so, we can achieve a more reliable and stable metric for content relevance, mitigating some of the challenges we've encountered in detecting subtle content drift. We could also focus on using less computationally heavy models such as DistilBERT.

Integration with Soft-404 Detection
Another ambitious goal on this roadmap is the integration of Soft-404 detection mechanisms. Soft-404 errors, which are essentially "Page Not Found" errors disguised as actual content, pose a significant challenge to the quality and reliability of Wikipedia references. By incorporating Soft-404 detection, we aim to further enhance the overall page quality of Wikipedia. This will not only improve the user experience but also contribute to the platform's credibility as a reliable source of information.

By focusing on these two key areas, we aim to build a more robust, accurate, and comprehensive system for tracking and managing content drift in Wikipedia hyperlinks. Overall, this system can be used to detect if a link’s content relevance to the wikipedia context has deteriorated over time, and possibly be replaced with a more relevant version from a web archive, such as the Wayback Machine.

Acknowledgements
I extend my deepest gratitude to all the mentors who have guided me throughout this project. Mark Graham and Dr. Sawood Alam have been very welcoming, and have helped me through the coding period for Google Summer of Code. A special acknowledgement goes to Dr. Sawood Alam, whose expertise and timely advice have been invaluable whenever I encountered challenges. Being a part of this project has been an enriching experience, and I am thankful for the opportunity to contribute and learn.

Reading material used for this project:
"The Decay and Failures of Web References" by Steve Lawrence, David Pennock, Gary William Flake, Robert Krovetz, Frans Coetzee, Eric Glover, Finn Årup Nielsen, Andries Kruger, and C. Lee Giles
“Detecting Content Drift on the Web Using Web Archives and Textual Similarity (short paper)” by Brenda Reyes Ayala, Qiufeng Du and Juyi Han
"Attention Is All You Need" by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin
"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" by Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova
"Abstractive Text Summarization using Sequence-to-Sequence RNNs and Beyond" by Ramesh Nallapati, Bowen Zhou, Cicero dos Santos, Caglar Gulcehre, and Bing Xiang
