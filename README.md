## Problem Statement
The INN Hotels Group is facing a critical rise in booking cancellations, with inventory loss reaching an all-time high of 18% and resulting \
in approximately $0.25 million in annual revenue loss. This trend has significantly impacted profit margins and operational efficiency. The group’s current \
heuristic, rule-based methods have proven ineffective, inefficient, and unscalable.The issue is further fueled by the growing ease of online and offline bookings \
offering flexible cancellation policies, often allowing last-minute cancellations. These are typically driven by unforeseen changes in travel plans such as personal \
emergencies or flight delays. Cancellations not only lead to unsold rooms (inventory loss) but also increase costs through higher distribution commissions, last-minute \
discounts, and reduced profit margins.
To combat this, the INN Hotels Group seeks a data science-based solution to proactively predict booking cancellations, aiming to minimize revenue loss, \
optimize inventory management, and improve overall operational efficiency.

## Dataset
* **Past Data:** Past booking records including cancellation status and booking details.
* **New Data:** New bookings data where cancellation predictions are applied.

## Workflow
1. Data Overview
* Loaded and inspected the dataset for shape and basic statistics.
* Gained initial insight into the distribution of cancellations.

2. Exploratory Data Analysis (EDA)
* Visualized cancellation rates to understand class distribution.
* Explored key categorical variables such as room type, market segment, and booking channel.
* Analyzed numerical features like lead time and average daily rate.
* Studied cancellation trends across time and customer segments.

3. Data Preprocessing
* Dropped irrelevant columns (e.g., booking ID).
* Handled missing values appropriately.
* Applied statistical models to check for irrelevant columns.
* Applied encoding technique on for categorical features.
* Building up new features from the existing features.
* Used PowerTransformer to reduce skewness.
* Scaled numerical features for modeling.

4. Model Development
* Defined the classification target as booking_status.
* Trained models including logistic regression and decision trees.
* Evaluated performance using metrics like accuracy, precision, recall, and F1 score.
* Tackled class imbalance where necessary.

5. Predictions on New Data
* Applied the final model to predict cancellations on new records.
* Exported predictions for business use.

### Key Features Analyzed
* Lead time before check-in
* Booking channel
* Rebooking indicator
* Stay duration analysis

## Tools & Technologies
* Python (pandas, numpy, scikit-learn, statsmodels, matplotlib, seaborn)
* Jupyter Notebook

## Future Work
- Extend analysis with time series modeling for trends.
- Build a real-time dashboard for monitoring cancellation risk.
