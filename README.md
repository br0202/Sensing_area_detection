## [Detecting Sensing Area of A Laparoscopic Probe in Minimally Invasive Cancer Surgery](https://arxiv.org/abs/2208.08407) (MICCAI 2023)
[![DOI](https://img.shields.io/badge/DOI-10.1007%2F978--3--031--16449--1__2-darkyellow)](https://doi.org/10.1007/978-3-031-16449-1_2)
[![arXiv](https://img.shields.io/badge/arXiv-2208.08407-b31b1b.svg)](https://arxiv.org/abs/2208.08407)

By [Baoru Huang](https://baoru.netlify.app/), Yicheng Hu, [Anh Nguyen](https://www.csc.liv.ac.uk/~anguyen), Stamatia Giannarou, Daniel S. Elson

![image](https://github.com/br0202/Sensing_area_detection/blob/master/figure/main_network2.png "Network")

### Contents
1. [Requirements](#requirements)
2. [Dataset](#dataset)
3. [Training&Testing](#Training & Testing)
4. [Notes](#Citing)


### Requirements

1. Installation:
	- `cd $Sensing_area_detection`
	- `conda env create -f environment.yml`

### Dataset
<p align="center">
  <img src="https://github.com/br0202/Sensing_area_detection/blob/master/figure/Picture4.png" width="370" />
  <img src="https://github.com/br0202/Sensing_area_detection/blob/master/figure/probe.jpeg" width="360" /> 
</p>
1. We newly acquired two datasets: 1) Jerry dataset and 2) Coffbea dataset
	- Jerry dataset includes: Stereo laparoscopic images with standard illumination, Stereo laparoscopic images with laser on and laparoscopic light off, laser segmentation mask, laser center point ground truth, and PCA line points txt files. 
	- Coffbea dataset includes: everything included in Jerry dataset, and the ground truth depth map of every frames.

2. Labelling:
	- Example data. (a) Standard illumination left RGB image; (b) left image with laser on and laparoscopic light off; same for (c) and (d) but for right images
![image](https://github.com/br0202/Sensing_area_detection/blob/master/figure/dataset-all.png "dataset")
	- Problem Definition. (a) The input RGB image, (b) The estimated line using PCA for obtaining principal points, (c) The image with laser on that we used to detect the intersection ground truth	
![image](https://github.com/br0202/Sensing_area_detection/blob/master/figure/label.png "PCA")
	
### Training & Testing

1. Training:
	- Change the data directory to the folder of data
	- `cd $Sensing_area_detection`
	- `python main.py --mode train`
	
2. Test:
    - `cd $Sensing_area_detection`
    - `python main.py --mode test`

3. Results:
	- Qualitative results. (a) and (c) are standard illumination images and (b) and (d) are images with laser on and laparoscopic light off. The predicted intersection point is shown in blue and the green point indicates the ground truth, which are further indicated by arrows for clarity
![image](https://github.com/br0202/Sensing_area_detection/blob/master/figure/vis.png "results")
	

### Citing 

If you find our paper useful in your research, please consider citing:

        

### License
MIT License

### Acknowledgement
1. This work was supported by the UK National Institute for Health Research (NIHR) Invention for Innovation Award NIHR200035, the Cancer Research UK Imperial Centre, the Royal Society (UF140290) and the NIHR Imperial Biomedical Research Centre.
