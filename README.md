# Unexpected Attributed Subgraphs: a Mining Algorithm

UnexPatterns is a an algorithm that mines unexpected patterns from large attributed graphs.

It relies on the combination of information-theoretic based measure of *Unexpectedness* [1] as an interestingness measure to select relevant patterns and pruning techniques to reduce the search space.

[1] Dessalles, J.L.: [Algorithmic simplicity and relevance.](https://perso.telecom-paristech.fr/jld/papers/Dessalles_11061001.pdf) *Algorithmic Probability and Friends - LNAI 7070*, pp. 119–130, Springer (2013).

## Install

Clone this project and install dependencies:
```
git clone
cd UnexPatterns
pip install -r requirements.txt
```
## Data
The five real-world datasets used in our work are provided either in `data/` directory or accessible through [Netset](https://netset.telecom-paris.fr/): 
* `'wikivitals'` 
* `'wikivitals-fr'`
* `'wikischools'`
* `'sanFranciscoCrimes'` 
* `'ingredients'`

## Pattern mining

### Parameter file

Use `parameters.txt` to specify the dataset(s) and parameter(s) value. As an example:
```
datasets: wikivitals, wikivitals-fr
s: 8
beta: 4
delta: 0
patterns_path: output
```
where:
*  `datasets`: Dataset name (if several datasets, names are coma-separated)
* `s`: Minimum number of nodes in pattern
* `beta`: Minimum number of attributes in pattern
* `delta`: Minimum amount of unexpectedness difference between two patterns
* `patterns_path`: Path to output directory 

### Usage
To mine patterns according to `parameters.txt`, use the following command:
```python
python unexpatterns.py parameters.txt
```

### Output format
Patterns are stored in binary format, within `output/patterns/` directory. To extract them, you can use:
```Python
with open(f'output/patterns/<patterns>.bin', 'rb') as data:
    patterns = pickle.load(data)
```
The naming convention of each file is the following: `result_<dataset>_<beta>_<s>_<delta>.bin`.  

The variable `patterns` then contains a list of patterns, each in the form of a tuple (X<sub>i</sub>,Q<sub>i</sub>) where X<sub>i</sub> is the set of node indexes and Q<sub>i</sub> the set of attribute indexes of the i<sup>th</sup> pattern.

## Experiments
Code for experiments and comparisons with baseline and state-of-the-art aglorithms are provided in `experiments/` directory.
