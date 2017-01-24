# Probability Estimation for Online Education System

Machine Learning and Computational Statistics

Felipe N. Ducau, Michael Higgins, Sebastián Brarda (New York University)

Project Advisor, Ph.D. Kush Varshney (IBM Research)

## Introduction and Problem Definition

Online education has become one of the main education formats globally. It has many advan-
tages over traditional education like lower costs, scalability and convenience. However it lacks
the personalization of traditional education.

It is very challenging to design a system that adapts to the different student skills and weaknesses.
The pace at which different students learn varies considerably across different topics and
exercise types. By training a machine to understand how well a particular student has learned a
particular topic, we could choose which is the most appropriate question to show next in order
to optimize her rate of learning. Thus we strive to predict as accurately as possible the prob-
ability that a student answers correctly any given question given his/her past performance. If
the probability of answering correctly a certain question is very high, the problem would be too
easy - showing that problem to the student would be a waste of time. On the other hand, if the
problem had a very low probability of success, it would be too difficult, generating frustration
without assisting learning.

In this project, we focused on generating calibrated probability estimates for the mentioned
task and proposing a personalization system based on these probabilities. The goal was to create
a strong probabilistic model using novel feature creation and leverage this model to create an
effective recommendation system.

Results and technical description [here](/DS-GA-1003_Final_Project.pdf)
