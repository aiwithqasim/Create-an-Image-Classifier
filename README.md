# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

<h1>AI Programming with Python Project by Udacity<h1>
  <h3>By : Qasim Hassan</h3>

<h2>Introduction</h2>
<p>Final project files for creating an image classifer for 102 different flowers. This is part of the requirements for completing the Udacity's AI Programming with Python Nanodegree. Here, I build an image classifier using PyTorch, then convert it into a command line application. The model can be saved to disk and loaded for usage at a later stage using the pytorch library.following are the Sample of Flowers</p>

<img src = "https://github.com/qasim1020/Create-an-Image-Classifier/blob/test/Assets/Flowers.png">

<h2>Requirements</h2>
This program requires the following modules to execute:
<ol>
  <li>python 3 and later</li>
  <li>pytorch 1.2</li>
  <li>numpy</li>
  <li>pandas</li>
  <li>IMAGE PIL</li>
  <li>torchvision</li>
</ol>
Altenatively, you can use anconda to manage your AI or data science environment.

<h2>GuideLIne</h2>
The user has the option to train the model using the GPU or the CPU just by passing in the arguement of choice for example starting the program with:

    python train.py --gpu --epochs 10 --save_dir dir
Starts the training program using the gpu if it exists on the host. 10 iterations are made and dir is used as the save and load directory for the model. While starting the program with:

    python predict --input file.jpg --catergory_names category_names.json
Performs a prediction on the given input. It uses the trained model on disk to detemine what class this image belongs to. The catergory names file conatins a mapping of the indexes to the catergory names.

<h2>Contribution & Acknowledgement</h2>
<p>Feel free to Contribute in that If you like Kindly hit that Star and show your love.if wanna meet me i'm always open to all and my
  <a href ="https://github.com/qasim1020">GitHub</a> & <a href = "https://www.linkedin.com/in/qasim-hassan/" >LinkedIN</a> </p>
