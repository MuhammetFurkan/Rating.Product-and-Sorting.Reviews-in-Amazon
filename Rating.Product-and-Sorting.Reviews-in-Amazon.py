
###################################################
# PROJECT: Rating Product & Sorting Reviews in Amazon
###################################################

#####################################################
# Business Problem
#####################################################

"""

One of the most important problems in e-commerce is the correct calculation of the points given to the products after sales.
The solution to this problem means providing greater customer satisfaction for the e-commerce site,prominence of the product for the sellers and a seamless shopping experience for the buyers.
Another problem is the correct ordering of the comments given to the products.
Since misleading comments will directly affect the sale of the product, it will cause both financial loss and loss of customers.
In the solution of these 2 basic problems, e-commerce site and sellers will increase their sales, while customers will complete their purchasing journey without any problems.

"""

# This data set, which includes Amazon product data, includes product categories and various metadata.
# The product with the most comments in the electronics category has user ratings and comments.

# Features:

# reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
# asin - ID of the product, e.g. 0000013714
# reviewerName - name of the reviewer
# helpful - helpfulness rating of the review, e.g. 2/3
# reviewText - text of the review
# overall - rating of the product
# summary - summary of the review
# unixReviewTime - time of the review (unix time)
# reviewTime - time of the review (raw)
# day_diff - Number of days since evaluation
# helpful_yes - The number of times the review was found helpful
# total_vote - Number of votes given to the review


############################################
# Importing Library and Functions
############################################

import matplotlib.pyplot as plt
import pandas as pd
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', 10)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


df = pd.read_csv(r"C:\Users\Furkan\Desktop\vbo_bootcamp\amazon_review.csv")
df["overall"].mean()


###################################################
# Calculate the Weighted Average of Score by Date.
###################################################


df["reviewTime"] = pd.to_datetime(df["reviewTime"])
current_date = pd.to_datetime((df["reviewTime"]).max())
df["day_diff"] = (current_date - df["reviewTime"]).dt.days

df.info()

# determination of time-based average weights
df.loc[df["day_diff"] <= 280, "overall"].mean()
df.loc[(df["day_diff"] > 280) & (df["day_diff"] <= 430), "overall"].mean()
df.loc[(df["day_diff"] > 430) & (df["day_diff"] <= 600), "overall"].mean()
df.loc[(df["day_diff"] > 600) & (df["day_diff"] <= 1063), "overall"].mean()



def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[dataframe["day_diff"] <= 280, "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 280) & (dataframe["day_diff"] <= 430), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 430) & (dataframe["day_diff"] <= 600), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 600) & (dataframe["day_diff"] <= 1063), "overall"].mean() * w4 / 100

time_based_weighted_average(df)


###################################################
# Generate the helpful_no variable
###################################################


df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

df.head()

###################################################
# Calculate score_pos_neg_diff, score_average_rating and wilson_lower_bound Scores and Add to Data
###################################################

def score_up_down_diff(up, down):
    return up - down

def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

def wilson_lower_bound(up, down, confidence=0.95):
    """
Calculate Wilson Lower Bound Score

     - The lower limit of the confidence interval to be calculated for the Bernoulli parameter p is accepted as the WLB score.
     - The score to be calculated is used for product ranking.
     - Note:
     If the scores are between 1-5, 1-3 are marked as negative, 4-5 as positive and can be adjusted to Bernoulli.
     This brings with it some problems. For this reason, it is necessary to make a bayesian average rating.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


# score_pos_neg_diff

df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down_diff(x["helpful_yes"], x["helpful_no"]), axis=1)

# score_avarage_rating

df["score_avarage_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["total_vote"]), axis=1)

# wilson_lower_bound

df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

# Identify the Top 20 Comments

df.sort_values("wilson_lower_bound", ascending=False).head(20)

