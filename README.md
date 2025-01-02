# Designing a Recommender System for Articles Using Implicit Feedback

**Reference Publication**: [Designing a Recommender System for Articles Using Implicit Feedback](https://link.springer.com/chapter/10.1007/978-981-16-8225-4_2)

This project focuses on building and evaluating a recommendation system using various collaborative filtering techniques such as Alternating Least Squares (ALS), Bayesian Personalized Ranking (BPR), and Logistic Matrix Factorization (LMF).

## Overview
The recommendation system predicts user preferences for content based on historical interactions. This implementation includes methods for data preprocessing, model training, and evaluation using key metrics such as AUC (Area Under the Curve).

## Dataset
Two datasets are used in this project:
- **`shared_articles.csv`**: Contains details of articles shared, including attributes like `contentId`, `title`, and `eventType`.
- **`users_interactions.csv`**: Contains user interactions with articles, such as views, likes, bookmarks, follows, and comments.

## Methodology
1. **Preprocessing**:
   - Merging the datasets based on `contentId`.
   - Mapping interaction types to numerical strengths (`VIEW=1`, `LIKE=2`, `BOOKMARK=3`, `FOLLOW=4`, `COMMENT=5`).
   - Creating sparse matrices for `content-user` and `user-content` interactions.

2. **Algorithms**:
   - **Alternating Least Squares (ALS)**: Collaborative filtering method to factorize user-item interaction matrices.
   - **Bayesian Personalized Ranking (BPR)**: Optimizes ranking of items for each user.
   - **Logistic Matrix Factorization (LMF)**: Combines logistic regression with matrix factorization for implicit feedback.

3. **Evaluation**:
   - AUC scores are computed to evaluate the performance of each algorithm.

## Key Findings
   - Collaborative filtering methods like ALS and LMF provide robust recommendations when paired with implicit feedback.

## Visualizations
- **Interaction Distribution**: Visualized using bar plots to highlight the frequency of each interaction type.
- **AUC-ROC Curves**: Used to validate model predictions against ground truth.

## Requirements
Ensure the following libraries are installed:
```bash
pip install pandas numpy scipy implicit seaborn matplotlib scikit-learn
```

## Usage
1. Clone this repository.
2. Place the datasets (`shared_articles.csv` and `users_interactions.csv`) in the same directory as `ALS_recSystem.py`.
3. Run the script:
   ```bash
   python ALS_recSystem.py
   ```
4. View the generated metrics and recommendations.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.
