# Causal Imitation Learning (mSBD)

This repository contains the implementation of a sequential imitation learning model under multi-outcome SBD conditions.

---

## Experiments
The experiment compares the imitation accuracy in the mSBD environment between:  
1. Applying Kumor's sequential pi-BD policy separately at each time step (baseline).  
2. Deriving the sequential pi-BD policy from a projected graph where the unobserved variable $Y$ has been projected (proposed method).  

The experiments were conducted on the following two graphs, with 1,000 SCMs and a sample size of 1,000 for each SCM.
The evaluation metric is the mean squared error (MSE) of \( Y_1 \) and \( Y_2 \), averaged across both.


## Usage
### Graph 1
```
python run_graph1_mSBDIL.py
```

### Graph 2
```
python run_graph2_mSBDIL.py
```