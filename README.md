# AMoS.24
2nd Iteration of Awesome Model Of Statistics (AMoS), TeenPact National Convention (2024 Target)

## Contents

[Introduction, Methodology](#introduction-and-methodology)

[Schema](#data-schema)

## Introduction and Methodology

AMoS is a project aimed at accurately and quickly simulating thousands of nationwide 
elections to create a comprehensive prediction about a winning presidential ticket. It is an
ongoing experiment on the best AI/ML testing a training practices, as well as the 
most computationally efficient methods of simulation, data generation, visualization, etc.

AMoS.24, the 2nd iteration of this experiment, hopes to improve on the accuracy and performance
of the original experiment. To accomplish this, it relies on a fundamentally different approach
than conventional election simulations, opting for a classification-based approach,
rather than a regression-based approach.

### Primary Challenges

AMoS.24's classification-based approach is prompted by a challenge encountered in
AMoS.23, which followed traditional, regression-based election simulation techniques like 
those proposed by FiveThirtyEight's US Election Simulator. These regression based
approaches can be effective at simulating two-party elections at large scale, but they 
presented 2 fundamental issues for AMoS.23's scope:

- Simulating a three-party election under this framework required some large mathematical 
assumptions when generating simulation data. In particular, when a party's vote share
was selected from it's normal distribution, there was no fair way of determining how these
votes were reallocated from among two opponents. 
- FiveThirtyEight's approach benefits from statistically significant sample sizes in every
state, allowing it to draw from it's probability distribution with much more confidence. 

AMoS.24's primary goal is to improve prediction accuracy by addressing these two challenges.
Rather than a purely regression-based approach of simulation, AMoS.24 will test the 
efficacy of training a classification tree to predict a winner out of three possible party 
representatives (compassion, vision, perseverance).


## Data Schema

This section will outline how the training/testing data is structured. The first iteration of 
AMoS.24 is based on the following schema:

| state       | candidate_a_vote_share | candidate_b_vote_share | candidate_c_vote_share | a_home_state | b_home_state | c_home_state | party_a | party_b | party_c | independent | absentee |
|-------------| ---------------------------- | --------------------------- | ----------------------------- | --------------- | ----------------- | ---------------- | --------- | --------- | -------- | -------------- | ----------- |
  | postal code | % | % | % | Yes/No | Yes/No | Yes/No | % | % | % | % | % | 

