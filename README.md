<!--
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
 -->


<!-- PROJECT LOGO -->
<br />
<h3 align="center">French real estate prices prediction</h3>
<p align="center">Are the DVF data set usefull for prices prediction?</p>
<p align="center">
  <a href="https://executive-education.dauphine.psl.eu/formations/executive-master-diplome-universite/ia-science-donnees" target="_blank">
    <img src="images/image_1.gif" alt="Logo" width="500" height="300">
  </a>




<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2> Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#description">Description</a></li>
      </ul>
      <ul>
        <li><a href="#datasets">Datasets</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>

  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
  
DVF datasets are annual data of real estates transactions published each year by the French Ministry of Economy and Finance. These datasets cover
7 years, from 2014 to 2020 and contains more than 14 millions rows (more or less 2 millions transaction by year), for several type of properties (residential, commercial, industrial, raw land..). 
They have been maid available with the aim to increase tranparency of the real estate market.  

**My main objective was to check if these dataset could be used for predicting residential properties' price, with good performance.**


### Description
**Much of the code has been stored in my own package and modules to make the Jupyter Notebook more readable.**


The project contains:

```sh
- 1 Jupyter Notebooks as the main files:
	* Note_Book.ipynb
	
- 1 package: mypackage
	* module : mydataloader.py
	* module : data_processor.py	
	* module : ploter.py	
```




<a href="https://github.com/DanielOmola/Data_Science_Portfolio/tree/main/Real_Estate_Price_Prediction" target="_blank">Project Link</a>
	

### Datasets

<a href="https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/" target="_blank">DVF datasets from 2007 to 2020 with more than 14 millions total entries</a>

<!-- GETTING STARTED -->
## Getting Started


### Prerequisites
*  Jupyter Notebook
*  Numpy
*  Pandas
*  Pickle
*  Plotly
*  Python3
*  Sklearn
*  sqlite3


### Installation

If you chose the first installation method, make sure the prerequisites are available in your system.

#### Method - 1
1. Clone the repo
```JS
   git clone https://github.com/DanielOmola/Data_Science_Portfolio/tree/main/Real_Estate_Price_Prediction
```
2. Open one of the file below in Jupyter Notebook
```JS
   Notebook.ipynb
```
<!-- -->

#### Method - 2
(the easiest way if docker is already installed in your system)
1. Clone the repo
```JS
   git clone https://github.com/DanielOmola/Data_Science_Portfolio/tree/main/Real_Estate_Price_Prediction
```
2. Open the terminal and move to the cloned directory 
```JS
   cd PATH/TO/THE/DIRECTORY
```
3. Create a Docker image from the terminal
```JS
   docker build . --no-cache=true -f Dockerfile.txt -t realestate
```
4. Run the Docker image
```JS
 docker run -it -p 8888:8888 realestate
```



<!-- USAGE EXAMPLES -->
## Usage

Play with it as you want. Change hyperparameter to see how the model performs on test set.


<!-- CONTACT -->
## Contact

Daniel OMOLA - daniel.omola@gmail.com


<!-- Recommended links -->
## Recommended links

* <a href="https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/" target="_blank">Demandes de valeurs foncières</a>