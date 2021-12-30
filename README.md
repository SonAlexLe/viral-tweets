# Viral tweets MLOps project

<p align="center">
<figure>
  <img src="docs/images/architectures.png">
  <figcaption><i>ML model and pipeline architectures.</i></figcaption>
</figure>
</p>

Forked from [the original repository](https://github.com/Mallonbacka/viral-tweets).

Please check out [our project documentation](https://sonalexle.github.io/viral-tweets/) for more details.

## Introduction

* Machine learning (ML) pipeline for predicting the virality of Twitter posts, or tweets.

* Scraping a large number of tweets with [Twint](https://github.com/twintproject/twint),

* performing exploratory data analysis,

* selecting a final ML model: BERT [1] (67% accuracy),

* and deploying it on a set of host machines capable of continuously *fetching new tweets, retraining and redeploying the model if necessary*.

* Completed as a part of the CS-C3250 Data Science Project course at Aalto University, under the supervision of a [Futurice](https://futurice.com/) company representative.

* Main goal: to learn the MLOps framework.

* Instead of using ready made tools from cloud providers, we write code that can be set up on any machine.

* Key tools and libraries: PyTorch [2], HuggingFace Transformers [3] for model training; MLflow for model monitoring; Flask for model deployment and pipeline integration.

## Contributors

We had an agile and flexible development process.

* Son ([@sonalexle](https://github.com/sonalexle)): helped the group navigate through the project; researched cloud options (GCP); wrote PyTorch scripts for data preprocessing and training BERT; coordinated model development with pipeline development.

* Matthew ([@Mallonbacka](https://github.com/Mallonbacka)): continuous and fault-tolerant data scraping.

* Pawel ([@Taikelenn](https://github.com/Taikelenn)): researched cloud options (Azure); mainly implemented and integrated the pipeline components: data downloading, model (re)training, model (re)deployment.

* Sergey ([@zakuraevs](https://github.com/zakuraevs)): researched cloud options (AWS); prototyped on AWS; researched pipeline components; drafted the pipeline.

* Long ([@normsie](https://github.com/normsie)): exploratory data analysis (EDA); researched ML models with TF-IDF [6] text vectorization; tried BERT hyperparameter tuning.

* Hafsa ([@salehi-Hafsa](https://github.com/salehi-Hafsa)): EDA; researched ML models with Doc2Vec [3] text vectorization.

* Binh ([@pdtbinh](https://github.com/pdtbinh)): prototyped BERT code in TensorFlow; EDA; helped Long, Hafsa, and Son with their approaches.

## References

1. Devlin, J., Chang, M.W., Lee, K. and Toutanova, K., 2018. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers).

2. Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L. and Desmaison, A., 2019. PyTorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems (NeurIPS), 32, pp. 8026-8037.

3. Wolf, T., Chaumond, J., Debut, L., Sanh, V., Delangue, C., Moi, A., Cistac, P., Funtowicz, M., Davison, J., Shleifer, S. and Louf, R., 2020. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations (pp. 38-45).