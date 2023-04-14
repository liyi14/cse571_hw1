# Homework 1 - EKF and Particle Filter for Localization

This assignment focuses on implementing a localization system using odometry-based motion models and landmark-based sensor models. You will be required to complete both writing and programming tasks. 
The main document for this assignment can be found [here](https://courses.cs.washington.edu/courses/cse571/23sp/homeworks/CSE571_HW1.pdf).

For additional reference, consult the following lecture slides:
- [Lecture 04: Extended Kalman Filter and Unscented Kalman Filter](https://courses.cs.washington.edu/courses/cse571/23sp/slides/L04/Lecture04_EKF_UKF.pdf)
- [Lecture 05: Particle Filter](https://courses.cs.washington.edu/courses/cse571/23sp/slides/L05/Lecture05_ParticleFilters_Updated.pdf)
- [Lecture 06: Sensor and Motion Models](https://courses.cs.washington.edu/courses/cse571/23sp/slides/L06/Lecture06_SensorMotion_Updated.pdf)

## Writing Task

In the writing portion of this assignment, you are asked to derive the Jacobian of the motion and observation models. These Jacobians will later be utilized in the programming assignment.

## Programming Task

The programming portion of this assignment consists of three parts:

### 1. Implement Extended Kalman Filter (EKF)

Refer to the [Lecture 04: EKF](https://courses.cs.washington.edu/courses/cse571/23sp/slides/L04/Lecture04_EKF_UKF.pdf) for implementing the Extended Kalman Filter algorithm.

### 2. Implement Particle Filter

Refer to the [Lecture 05: PF](https://courses.cs.washington.edu/courses/cse571/23sp/slides/L05/Lecture05_ParticleFilters_Updated.pdf) and [Lecture 06: PF & Sensor Motion](https://courses.cs.washington.edu/courses/cse571/23sp/slides/L06/Lecture06_SensorMotion_Updated.pdf) for implementing the Particle Filter algorithm.

### 3. Train a CNN-based Observation Model on Colab

Train a Convolutional Neural Network (CNN) as an observation model using [Google Colab](https://colab.research.google.com/).

## Submission Guidelines

Please submit your solutions for both the writing and programming tasks in the specified format. Ensure that your code is well-documented and includes comments to explain your implementation.
